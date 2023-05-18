import io
import sys
import queue
import shutil
import tempfile
import traceback
import multiprocessing
from ..tracer import Tracer
from .utils import (
    create_tempdir,
    reliability_guard,
    swallow_io,
    TimeoutException,
    time_limit,
)

def indent_code(code: str) -> str:
    """Indent code by 4 spaces."""
    return "\n".join(["    " + line for line in code.split("\n")])

TRACER_LINE_SHIFT = -1

def get_exec_traceback_info(
    exc_type,
    exc_value,
    exc_traceback,
    cur_code: str,
    line_no_shift: int = TRACER_LINE_SHIFT,
) -> str:
    splited_code = cur_code.split("\n")
    
    # Only retain the traceback in the executed code
    while exc_traceback:
        if exc_traceback.tb_frame.f_code.co_filename != "<string>":
            exc_traceback = exc_traceback.tb_next
        else:
            break
    # exc_traceback will be None in the case of SyntaxError
    # since there will be no traceback in the executed code "<string>"
    
    stack = traceback.StackSummary.extract(
        traceback.walk_tb(exc_traceback),
        lookup_lines=False,
        capture_locals=False
    )
    # If stack is NOT empty, correct the line number and _line
    for frame_summary in stack:
        # frame_summary.filename # <string>
        # frame_summary.name # <module>
        # Modify the line number to match the original code
        if frame_summary.filename == "<string>":
            # this removes the initial line "with tracer:" from counting
            frame_summary.lineno += line_no_shift
            # minus 1 because the line number starts from 1 and we're using 0-based indexing
            frame_summary._line = splited_code[frame_summary.lineno - 1]

    if exc_type and issubclass(exc_type, SyntaxError):
        exc_value.lineno += line_no_shift
        # set to None to avoid printing the offset
        # which is probably not useful to feed into the model
        exc_value.offset = None

    traceback_str = "Traceback (most recent call last):\n"
    traceback_str += "\n".join(stack.format())
    traceback_str += "\n".join(list(
        traceback.TracebackException(
            exc_type, exc_value, None, limit=None, lookup_lines=False, capture_locals=False
        ).format_exception_only()
    ))

    traceback_info = {
        "exc_type": exc_type.__name__,
        "exc_value": str(exc_value),
        "str": traceback_str,
    }
    if exc_type and issubclass(exc_type, SyntaxError):
        traceback_info["extra_info"] = {
            "lineno": exc_value.lineno,
            "offset": exc_value.offset,
            "text": exc_value.text,
            "msg": exc_value.msg,
        }
    return traceback_info

def exec_code(
    cur_code: str,
    do_trace: bool = False,
    return_globals: bool = False,
    extra_headers: str = "",
) -> dict:
    tracer_log_io = io.StringIO()
    
    n_extra_header_lines = len(extra_headers.split("\n"))
    line_no_shift = TRACER_LINE_SHIFT - n_extra_header_lines
    try:
        # stringIO
        tracer = Tracer(
            source_code=cur_code,
            line_no_shift=line_no_shift,
            do_trace=do_trace,
            file=tracer_log_io
        )
        exec_globals = {"tracer": tracer}
        exec_code = extra_headers + "\n" + f"with tracer:\n{indent_code(cur_code)}"
        exec(exec_code, exec_globals)

        traceback_info = None
        success = True
    except Exception:
        # Get traceback object
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_info = get_exec_traceback_info(
            exc_type,
            exc_value,
            exc_traceback,
            cur_code,
            line_no_shift=line_no_shift
        )
        success = False
    tracer_log = tracer_log_io.getvalue()
    tracer_log_io.close()
    ret = {
        "success": success,
        "traceback": traceback_info,
        "tracer_log": tracer_log,
        "reason": "exception" if not success else None,
    }

    if return_globals:
        ret["globals"] = exec_globals
    return ret

def _unsafe_execute(
    check_program,
    tempdir: str,
    timeout: float=5.0,
    do_trace: bool = False,
    extra_headers: str = "",
    debug: bool = False
):
    if debug:
        try:
            res = exec_code(
                check_program,
                do_trace=do_trace,
                extra_headers=extra_headers
            )
        except TimeoutException:
            res = {
                "success": False,
                "reason": "timeout",
            }
        return res

    import os
    # tempdir is a temporary directory, will be cleaned up by the caller
    os.chdir(tempdir)

    # Disable functionalities that can make destructive changes to the test.
    reliability_guard()

    # Run program.
    try:
        with swallow_io():
            with time_limit(timeout):
                res = exec_code(
                    check_program,
                    do_trace=do_trace,
                    extra_headers=extra_headers
                )
    except TimeoutException:
        res = {
            "success": False,
            "reason": "timeout",
        }

    return res

def unsafe_execute(
    check_program,
    tempdir: str,
    timeout: float = 5.0,
    do_trace: bool = False,
    extra_headers: str = "",
):
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """
    try:
        return _unsafe_execute(
            check_program,
            tempdir,
            timeout,
            do_trace,
            extra_headers=extra_headers
        )
    except Exception as e:
        return {
            "success": False,
            "reason": "execution driver exception",
            "traceback": {
                "exc_type": e.__class__.__name__,
                "exc_value": str(e),
                "str": traceback.format_exc(),
            }
        }

def unsafe_execute_mp(
    check_program,
    timeout: float = 5.0,
    do_trace: bool = False,
    extra_headers: str = "",
) -> dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """
    # Create a temporary directory for the execution.
    tempdir = tempfile.mkdtemp()

    q = multiprocessing.Queue()
    def unsafe_execute_mp_wrapper(check_program, tempdir, timeout, do_trace, extra_headers, q):
        res = unsafe_execute(
            check_program,
            tempdir,
            timeout,
            do_trace,
            extra_headers=extra_headers,
        )
        q.put(res)

    # Start the process.
    process = multiprocessing.Process(
        target=unsafe_execute_mp_wrapper,
        args=(check_program, tempdir, timeout, do_trace, extra_headers, q)
    )
    process.start()

    # Wait for the process to finish (w/ return value) or timeout.
    try:
        result = q.get(block=True, timeout=timeout)
    except queue.Empty:
        result = {
            "success": False,
            "reason": "process timeout",
        }

    # Kill the process if it's still running.
    process.join(timeout=timeout + 1)
    if process.is_alive():
        process.kill()

    # Clean up the temporary directory.
    shutil.rmtree(tempdir)

    return result
