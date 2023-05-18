import re
import random
from .utils import (
    INCORRECT_LANG_FEEDBACK,
    CORRECT_LANG_FEEDBACK,
    CORRECT_LANG_FEEDBACK_POSTPROCESS_HEURISTIC,
    dump_offset_mapping_to_json
)

def extract_error_spans(content, traceback_str):
    # _tb_str = _sample["execution_result"]["traceback"]["str"]
    content_lines = content.splitlines(keepends=True)

    # extract the line numbers from the traceback
    relevant_line_nums = re.findall(r"File \"<string>\", line (\d+)", traceback_str)
    relevant_line_nums = [int(x) for x in relevant_line_nums]
    # filter out the line numbers that are out of range
    relevant_line_nums = [x for x in relevant_line_nums if x <= len(content_lines)]

    error_spans = []
    start_char_idx = 0
    for i, line in enumerate(content_lines):
        if i+1 in relevant_line_nums:
            # print(f"-> {i+1:4d}| {line}", end="")
            error_spans.append((start_char_idx, start_char_idx + len(line)))
        # else:
        #     print(f"   {i+1:4d}| {line}", end="")
        start_char_idx += len(line) # we don't use += 1 since we already keep the newline character
    
    # print the error spans
    # print(f"Detected error spans: {error_spans}")
    # for start, end in error_spans:
    #     print(f"{start:4d}, {end:4d} | {content[start:end]}")

    return error_spans

def find_non_error_spans(content, error_spans, start_idx=0):
    non_error_spans = []
    start_char_idx = start_idx
    if error_spans:
        for start, end in error_spans:
            assert start_char_idx <= start, f"start_char_idx={start_char_idx} > start={start}"
            if start_char_idx < start:
                non_error_spans.append((start_char_idx, start))
            start_char_idx = end

    if start_char_idx < len(content):
        non_error_spans.append((start_char_idx, len(content)))
    
    return non_error_spans

def build_data_instance_markline_heuristic(
    row,
    include_input=False,
    cur_df_path=None,
    add_reference=True,
):
    """Builds a data instance for the reward model.
    
    NOTE: this build data instance using the mark-line heuristic on top of the post-processing results.
    REQUIRES: the input `row` is a row from the dataframe with execution results after post-processing.

    The general idea is that, for errors, we want to mark the line that caused the error as incorrect.
    """
    assert "original_generation" in row, "The input row must be from the dataframe with execution results after post-processing."

    success = row["execution_result"]["success"]
    
    # Inputs
    problem = row["prompt"]
    generated_solution = row["generation"]
    generated_solution_w_extra_code = row["original_generation"] # not post-processed

    if not success:
        if "traceback" not in row["execution_result"]:
            language_feedback = INCORRECT_LANG_FEEDBACK + row["execution_result"]["reason"] # skip this example
        else:
            language_feedback = INCORRECT_LANG_FEEDBACK + row["execution_result"]["traceback"]["str"]
        language_feedback_w_extra_code = language_feedback
    else:
        # randomly select one of the correct language feedback
        language_feedback = random.choice(list(CORRECT_LANG_FEEDBACK))
        language_feedback_w_extra_code = random.choice(list(CORRECT_LANG_FEEDBACK_POSTPROCESS_HEURISTIC))
        

    # different versions of text_a, text_b
    text_a = language_feedback
    text_b = problem + generated_solution

    text_a_w_extra_code = language_feedback_w_extra_code
    text_b_w_extra_code = problem + generated_solution_w_extra_code
    assert text_b_w_extra_code.startswith(text_b), "The post-processed solution must be a prefix of the original solution."
    
    if add_reference:
        text_b += "\n" + row["reference"]
        text_b_w_extra_code += "\n" + row["reference"]

    data_instances = []
    if success:

        # mark the post-processed version of text_b as correct with a positive reward
        if add_reference: # text_b has additional "reference" added
            text_b_reward_offset_mapping = {
                (0, len(problem)): 0.0, # no reward for the problem (don't want to encourage the model to judge the problem)
                (len(problem), len(problem + generated_solution)): 1.0, # reward for the generated solution (potentially with test cases) - POSITIVE
                (len(problem + generated_solution), len(text_b)): 0.0, # no reward for the reference solution
            }
        else:
            text_b_reward_offset_mapping = {
                (0, len(problem)): 0.0, # no reward for the problem (don't want to encourage the model to judge the problem)
                (len(problem), len(text_b)): 1.0 # reward for the generated solution (potentially with test cases) - POSITIVE
            }
        post_processed_instance = {
            "success": success,
            
            "problem": problem,
            "generated_solution": generated_solution,
            "language_feedback": language_feedback,
            "reward_offset_mapping": dump_offset_mapping_to_json(text_b_reward_offset_mapping),

            # concated text
            "text_a": text_a,
            "text_b": text_b,
        }
        data_instances.append(post_processed_instance)

        # mark the extra code as incorrect with a negative reward
        if add_reference: # text_b has additional "reference" added
            text_b_w_extra_code_reward_offset_mapping = {
                (0, len(problem)): 0.0, # no reward for the problem (don't want to encourage the model to judge the problem)
                (len(problem), len(problem + generated_solution)): 1.0, # reward for the generated solution
                (len(problem + generated_solution), len(problem + generated_solution_w_extra_code)): -1.0, # reward for the extra code - NEGATIVE
                (len(problem + generated_solution_w_extra_code), len(text_b_w_extra_code)): 0.0 # reward for the reference - NEUTRAL
            }
        else:
            text_b_w_extra_code_reward_offset_mapping = {
                (0, len(problem)): 0.0, # no reward for the problem (don't want to encourage the model to judge the problem)
                (len(problem), len(text_b)): 1.0, # reward for the generated solution (after post-processing) - POSITIVE
                (len(text_b), len(text_b_w_extra_code)): -1.0 # reward for the extra code - NEGATIVE
            }
        extra_code_instance = {
            "success": success,

            "problem": problem,
            "generated_solution": generated_solution_w_extra_code,
            "language_feedback": language_feedback_w_extra_code,
            "reward_offset_mapping": dump_offset_mapping_to_json(text_b_w_extra_code_reward_offset_mapping),
            # concated text
            "text_a": text_a_w_extra_code,
            "text_b": text_b_w_extra_code,
        }
        data_instances.append(extra_code_instance)
    else:

        # Use heuristic to find reward offset mapping
        if row["execution_result"].get("traceback", None) is not None \
            and row["execution_result"]["traceback"]["exc_type"] != "AssertionError":
            # only use heuristic to find error spans for non-assertion errors
            text_b_error_spans = extract_error_spans(text_b, row["execution_result"]["traceback"]["str"])
            text_b_non_error_spans = find_non_error_spans(text_b, text_b_error_spans, start_idx=len(problem))
            text_b_reward_offset_mapping = {
                (0, len(problem)): 0.0,
                **{
                    (start, end): -1.0 for start, end in text_b_error_spans
                    # mark the error span with a large negative reward
                },
                **{ 
                    (start, end): -0.5 for start, end in text_b_non_error_spans
                    # mark the non-error causing span with a small negative reward
                },
            }
            
            text_b_w_extra_code_error_spans = extract_error_spans(
                text_b_w_extra_code, row["execution_result"]["traceback"]["str"]
            )
            text_b_w_extra_code_non_error_spans = find_non_error_spans(
                text_b_w_extra_code, text_b_w_extra_code_error_spans, start_idx=len(problem)
            )
            text_b_w_extra_code_reward_offset_mapping = {
                (0, len(problem)): 0.0,
                **{ 
                    (start, end): -1.0 for start, end in text_b_w_extra_code_error_spans
                    # mark the error span with a large negative reward
                },
                **{  
                    (start, end): -0.5 for start, end in text_b_w_extra_code_non_error_spans
                    # mark the non-error causing span with a small negative reward
                },
            }
        else:
            text_b_reward_offset_mapping = {
                (0, len(problem)): 0.0,
                (len(problem), len(text_b)): -1.0
            }
            text_b_w_extra_code_reward_offset_mapping = {
                (0, len(problem)): 0.0,
                (len(problem), len(text_b_w_extra_code)): -1.0
            }

        # mark both version as incorrect with a negative reward
        post_processed_instance = {
            "success": success,

            "problem": problem,
            "generated_solution": generated_solution,
            "language_feedback": language_feedback,
            "reward_offset_mapping": dump_offset_mapping_to_json(text_b_reward_offset_mapping),

            # concated text
            "text_a": text_a,
            "text_b": text_b,
        }
        data_instances.append(post_processed_instance)

        assert language_feedback_w_extra_code == language_feedback, \
            "The language feedback should be the same for both versions when the post-processed version is incorrect."
        assert text_a_w_extra_code == text_a, \
            "The text_a should be the same for both versions when the post-processed version is incorrect."

        extra_code_instance = {
            "success": success,

            "problem": problem,
            "generated_solution": generated_solution_w_extra_code,
            "language_feedback": language_feedback_w_extra_code,
            "reward_offset_mapping": dump_offset_mapping_to_json(text_b_w_extra_code_reward_offset_mapping),
            # concated text
            "text_a": text_a_w_extra_code,
            "text_b": text_b_w_extra_code,
        }
        data_instances.append(extra_code_instance)

    if include_input:
        for instance in data_instances:
            instance["input"] = row.to_dict() if not isinstance(row, dict) else row
            instance["data_path"] = cur_df_path
    return data_instances
