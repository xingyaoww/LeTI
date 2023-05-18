import random
from typing import Mapping, Tuple, Union
from .utils import (
    INCORRECT_LANG_FEEDBACK,
    CORRECT_LANG_FEEDBACK,
    CORRECT_LANG_FEEDBACK_POSTPROCESS_HEURISTIC,
    dump_offset_mapping_to_json
)

def build_data_instance_training(
    row,
    correct_lang_feedback: str,
    include_input=False,
    cur_df_path=None,
    add_reference=False,
):
    """Builds a data instance for the reward model.
    
    NOTE: this build naive reward value which should be ignored.
    """
    success = row["execution_result"]["success"]
    
    # Inputs
    problem = row["prompt"]
    generated_solution = row["generation"]
    reference = row["reference"]
    if not success:
        if "traceback" not in row["execution_result"]:
            language_feedback = INCORRECT_LANG_FEEDBACK + row["execution_result"]["reason"] # skip this example
        else:
            language_feedback = INCORRECT_LANG_FEEDBACK + row["execution_result"]["traceback"]["str"]
    else:
        # randomly select one of the correct language feedback
        language_feedback = correct_lang_feedback

    text_a = language_feedback
    text_b = problem + generated_solution
    if add_reference:
        text_b += "\n" + reference

    # Reward only applies to the generated solution
    reward_offset_mapping: Mapping[Tuple[int, int], Union[float, None]] = {
        (0, len(problem)): 0.0, # no reward for the problem (don't want to encourage the model to judge the problem)
        (len(problem), len(text_b)): 1.0 if success else -1.0 # reward for the generated solution
    }

    ret = {
        "success": success,
        
        "problem": problem,
        "generated_solution": generated_solution,
        "language_feedback": language_feedback,
        "reward_offset_mapping": dump_offset_mapping_to_json(reward_offset_mapping),

        # concated text
        "text_a": text_a,
        "text_b": text_b,
    }
    if include_input:
        ret["input"] = row.to_dict() if not isinstance(row, dict) else row
        ret["data_path"] = cur_df_path
    return ret


def convert_execution_result_to_train_instance(row):
    """Convert execution result to a data instance for reward-conditioned model training.
    """
    success = row["execution_result"]["success"]
    
    # Inputs
    problem = row["prompt"]
    generated_solution = row["generation"] # will always be the solution we want to use
    if not success:
        if "traceback" not in row["execution_result"]:
            language_feedback = row["execution_result"]["reason"]
        else:
            language_feedback = row["execution_result"]["traceback"]["str"]
    else:
        # randomly select one of the correct language feedback
        language_feedback = None

    if "additional_feedback" in row and row["additional_feedback"]:
        # add additional feedback
        if language_feedback is not None:
            language_feedback += "\n" + row["additional_feedback"]
        else:
            language_feedback = row["additional_feedback"]

    text_a = language_feedback
    text_b = problem + generated_solution
    ret = [{
        "success": success,
        
        "problem": problem,
        "generated_solution": generated_solution,
        "language_feedback": language_feedback,

        # concated text
        "text_a": text_a,
        "text_b": text_b,
    }]
    return ret
