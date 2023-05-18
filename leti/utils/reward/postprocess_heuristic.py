import random
from typing import Mapping, Tuple, Union
from .utils import (
    INCORRECT_LANG_FEEDBACK,
    CORRECT_LANG_FEEDBACK,
    CORRECT_LANG_FEEDBACK_POSTPROCESS_HEURISTIC,
    dump_offset_mapping_to_json
)


def build_data_instance_postprocess_heuristic(row, include_input=False, cur_df_path=None):
    """Builds a data instance for the reward model.
    
    NOTE: this build data instance using the post-processing heuristic.
    REQUIRES: the input `row` is a row from the dataframe with execution results after post-processing.

    The general idea is that, when post-processing code is executed sucessfully, this means the extra code is not needed.
    We want to train the reward model to suppress the extra code by assigning a negative reward to the extra code,
    while assigning a positive reward to the post-processed (success) solution.
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
    
    data_instances = []
    if success:
        # mark the post-processed version of text_b as correct with a positive reward
        post_processed_instance = {
            "success": success,
            
            "problem": problem,
            "generated_solution": generated_solution,
            "language_feedback": language_feedback,
            "reward_offset_mapping": dump_offset_mapping_to_json({
                (0, len(problem)): 0.0, # no reward for the problem (don't want to encourage the model to judge the problem)
                (len(problem), len(text_b)): 1.0 # reward for the generated solution - POSITIVE
            }),

            # concated text
            "text_a": text_a,
            "text_b": text_b,
        }
        data_instances.append(post_processed_instance)

        # mark the extra code as incorrect with a negative reward
        extra_code_instance = {
            "success": success,

            "problem": problem,
            "generated_solution": generated_solution_w_extra_code,
            "language_feedback": language_feedback_w_extra_code,
            "reward_offset_mapping": dump_offset_mapping_to_json({
                (0, len(problem)): 0.0, # no reward for the problem (don't want to encourage the model to judge the problem)
                (len(problem), len(text_b)): 1.0, # reward for the generated solution (after post-processing) - POSITIVE
                (len(text_b), len(text_b_w_extra_code)): -1.0 # reward for the extra code - NEGATIVE
            }),
            # concated text
            "text_a": text_a_w_extra_code,
            "text_b": text_b_w_extra_code,
        }
        data_instances.append(extra_code_instance)
    else:
        # mark both version as incorrect with a negative reward
        post_processed_instance = {
            "success": success,

            "problem": problem,
            "generated_solution": generated_solution,
            "language_feedback": language_feedback,
            "reward_offset_mapping": dump_offset_mapping_to_json({
                (0, len(problem)): 0.0, # no reward for the problem (don't want to encourage the model to judge the problem)
                (len(problem), len(text_b)): -1.0 # reward for the generated solution - NEGATIVE
            }),

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
            "reward_offset_mapping": dump_offset_mapping_to_json({
                (0, len(problem)): 0.0, # no reward for the problem (don't want to encourage the model to judge the problem)
                (len(problem), len(text_b_w_extra_code)): -1.0, # reward for the generated solution (after post-processing) - NEGATIVE
            }),
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
