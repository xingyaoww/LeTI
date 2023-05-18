import os
import jax
import time
import threading
import numpy as np
from tqdm import tqdm
from queue import Queue
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    TimeoutError,
)

from typing import List, Callable, Iterable
from leti.utils.execute import unsafe_execute_mp

class Executor:
    """Executor for executing code in parallel as a consumer."""
    def __init__(
            self,
            process_id: int,
            timeout: int = 5,
            extra_headers: str = "",
            num_workers=min(32, os.cpu_count() + 4),
            disable_execution=False,
        ):
        print(f"Executor: Using {num_workers} workers (process {process_id})")
        self.process_id = process_id
        self.timeout = timeout
        self.extra_headers = extra_headers
        self.disable_execution = disable_execution
        if process_id == 0:
            # Only the main process should do work
            self.num_workers = num_workers
            self.executor = ProcessPoolExecutor(max_workers=num_workers)
            self.results: List[Future] = []
        else:
            print("Executor: Not using executor since process_id != 0")
        
        # Due to all-gather operation required,
        # all the processes should run future_tasks_iter (e.g., decode the sentence)
        self.future_tasks_fn_queue = Queue()
        # only 1 thread is needed, more threads will probably mess up the all-gather operation
        # NOTE: we have to maintain the order of the tasks, so we can't use ThreadPoolExecutor
        # otherwise, it might cause deadlock
        self.future_task_exec_thread = threading.Thread(
            target=self._future_task_exec_thread_fn,
            daemon=True,
        )
        self.future_task_exec_thread.start()

    def _future_task_exec_thread_fn(self):
        while True:
            tasks_iter = self.future_tasks_fn_queue.get()

            for task in tasks_iter():
                if not self.disable_execution:
                    self.add_task(task)

            self.future_tasks_fn_queue.task_done()

    @staticmethod
    def _execute_task(task: dict, timeout: int, extra_headers: str) -> dict:
        cur_code = task["prompt"] + task["generation"] + "\n" + task["reference"]
        exec_results = unsafe_execute_mp(
            cur_code,
            timeout=timeout,
            do_trace=False,
            extra_headers=extra_headers
        )
        return {
            "execution_result": exec_results,
            **task,
        }

    def add_task(self, task: dict):
        """Add a task to the executor."""
        if self.process_id != 0:
            # Only the main process should add tasks
            return
        future = self.executor.submit(
            self._execute_task,
            task,
            timeout=self.timeout,
            extra_headers=self.extra_headers,
        )
        self.results.append((task, future))

    def add_future_tasks(
        self,
        task_iter_fn: Callable[[], Iterable[dict]],
        delay_execution=True
    ):
        """Add a future task to the future_tasks_fn_queue (ThreadPoolExecutor)."""
        # We add future tasks for all the processes, 
        # since we might need to execute all_gather for task_iter_fn

        # Call the tasks_iter in a separate thread to avoid blocking
        if delay_execution:
            self.future_tasks_fn_queue.put(task_iter_fn)

        else:
            for task in task_iter_fn():
                if not self.disable_execution:
                    self.add_task(task)

    def get_results_and_reset(self) -> List[dict]:
        """Get the results of all tasks and reset the executor."""
        if self.process_id != 0:
            # Only the main process should get the results
            return []

        # kill the thread if it takes too long
        print(f"Waiting for future_tasks_fn_queue to be empty... queue size: {self.future_tasks_fn_queue.qsize()}")
        while self.future_tasks_fn_queue.qsize() > 0:
            tasks_iter = self.future_tasks_fn_queue.get()
            for task in tasks_iter():
                if not self.disable_execution:
                    self.add_task(task)
            self.future_tasks_fn_queue.task_done()

        print(f"future_task_thread finished: {not self.future_task_exec_thread.is_alive()}")
        self.future_tasks_fn_queue.join()

        if self.future_tasks_fn_queue.unfinished_tasks > 0:
            print(f"WARNING: End with unfinished tasks: {self.future_tasks_fn_queue.unfinished_tasks}")

        if self.disable_execution:
            return None

        results = []
        pbar = tqdm(total=len(self.results), desc="Waiting for code execution results...")
        for task, future in self.results:
            try:
                _res = future.result(timeout=self.timeout + 3)
            except TimeoutError as e:
                print(f"TimeoutError @ Executor: {e}")
                _res = {
                    "execution_result": {
                        "success": False,
                        "reason": "process timeout",
                    },
                    **task,
                }
            results.append(_res)
            pbar.update(1)
        self.results = []
        pbar.close()
        return results

    def shutdown(self, wait=True):
        if self.process_id != 0:
            # Only the main process should shutdown
            return
        self.executor.shutdown(wait=wait)


def future_tasks_iter_fn(
    cur_generation_ids: jax.Array,
    batch: dict,
    tokenizer,
    task_callback_fn: Callable[[dict], None] = None,
    postprocess_fn: Callable[[str], str] = None,
    prompt_processing_fn: Callable[[str], str] = lambda x: x,
    postprocess_lang_feedback_fn: Callable[[str], str] = None,
) -> Iterable[dict]:
    """
    This function is called by the executor to get the next batch of tasks to execute.
    
    This function should return an iterable of tasks, where each task is a dictionary.
    It is designed to work with jax's asynchronous dispatch, so this should be run in a separate thread to avoid blocking the main thread.
    """
    # block until all computations are done
    cur_generation_ids = np.array(cur_generation_ids)

    # Remove the prompt from the generated solutions
    input_seq_len = batch["input_ids"].shape[1]
    cur_generation_ids = cur_generation_ids[:, input_seq_len:]

    # Decode the generated solutions
    cur_generations = tokenizer.batch_decode(
        cur_generation_ids,
        skip_special_tokens=True
    )

    prompt_indices = batch["prompt_idx"]
    generation_indices = batch["generation_idx"]

    for i, cur_gen in enumerate(cur_generations):
        cur_prompt_idx = prompt_indices[i]
        cur_generation_idx = generation_indices[i]

        if cur_prompt_idx is None or cur_generation_idx is None:
            # This is a padding element
            continue
        else:
            cur_prompt_idx = str(cur_prompt_idx)
            cur_generation_idx = int(cur_generation_idx)
        
        if postprocess_fn is not None:
            original_gen = cur_gen
            cur_gen = postprocess_fn(cur_gen)

        # Send the generated solutions to the solution executor
        cur_task = {
            "prompt_idx": cur_prompt_idx,
            "generation_idx": cur_generation_idx,
            # if GSM, don't include the prompt (will cause syntax errors)
            "prompt": prompt_processing_fn(batch["text"][i]),
            "generation": cur_gen,
            "reference": batch["reference"][i],
        }

        if postprocess_fn is not None:
            cur_task["original_generation"] = original_gen

        if postprocess_lang_feedback_fn is not None:
            assert original_gen is not None
            cur_task["additional_feedback"] = postprocess_lang_feedback_fn(
                original_gen, cur_gen
            )

        if task_callback_fn is not None:
            task_callback_fn(cur_task)

        yield cur_task # to be sent as future task to the executor
