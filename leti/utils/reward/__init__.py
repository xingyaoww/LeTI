import json
import random
import logging
import numpy as np
from collections import deque
from typing import List, Tuple, Deque, Mapping, Union
from transformers import PreTrainedTokenizer

logger = logging.getLogger("leti.utils.reward")

def is_cur_span_match_reward_span(start, end, rw_mapping) -> Tuple[bool, float]:
    if len(rw_mapping) == 0:
        return False, np.nan

    (rw_start, rw_end), rw_val = rw_mapping[0] # first reward span
    assert rw_start <= rw_end, f"rw_start {rw_start} must be <= rw_end {rw_end}"

    # we only allow the cases
    # 1. current token span is before the first reward span
    # 2. current token span is within the first reward span
    # 3. current token span is after the first reward span
    # but not the case where current token span is partially within the first reward span
    # e.g., current token span is (70, 80) and first reward span is (69, 180)
    token_span_len = end - start
    if (start < rw_start and end > rw_start):
        # start ------------ end ----------------
        # ------ rw_start ------ rw_end -------
        old_span = (start, end)
        assert end <= rw_end, f"end {end} must be <= rw_end {rw_end}"
        overlap = end - rw_start
        assert overlap < token_span_len, f"overlap {overlap} must be < token_span_len {token_span_len}"
        if overlap > token_span_len // 2: # break the tie by choose the left half
            start = rw_start
        else:
            end = rw_start
        logger.warning(
            f"current token span {old_span} is partially within the first reward span {rw_start, rw_end}."
            f" Attempting to fix to token span {start, end}..."
        )

    if start < rw_end and end > rw_end:
        # -------- start ------- end ---
        # rw_start ------ rw_end -------
        old_span = (start, end)
        assert start >= rw_start, f"start {start} must be >= rw_start {rw_start}"
        overlap = rw_end - start
        assert overlap < token_span_len, f"overlap {overlap} must be < token_span_len {token_span_len}"
        if overlap >= token_span_len // 2: # break the tie by choose the left half
            end = rw_end
        else:
            start = rw_end
        logger.warning(
            f"current token span {old_span} is partially within the first reward span {rw_start, rw_end}."
            f" Attempting to fix to token span {start, end}..."
        )

    if end <= rw_start: # current token is before the first reward span
        return False, np.nan
    elif start >= rw_start and end <= rw_end: # current token is within the first reward span
        return True, rw_val
    elif start >= rw_end: # current token is after the first reward span
        rw_mapping.popleft()
        return is_cur_span_match_reward_span(start, end, rw_mapping)
    else:
        raise ValueError(
            f"current token span {start, end} hits the corner case. First reward span {rw_start, rw_end}"
        )

def get_reward_mask_and_value(
        shape: Tuple[int, int],
        token_offset_mapping: List[List[Tuple[int, int]]],
        reward_offset_mapping: List[Mapping[Tuple[int, int], Union[float, None]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
    # no labels for reward model, but reward val for regression
    has_reward_mask = np.zeros(shape, dtype=np.bool_)
    reward_val = np.zeros(shape, dtype=np.float32)
    # set has_reward_mask to 1 for tokens that are part of a span specified in reward_offset_mapping
    for i, offset_mapping in enumerate(token_offset_mapping):
        rw_mapping = deque(reward_offset_mapping[i].items())
        # e.g., {(69, 180): 0.0, (180, 276): 1.0}                
        for j, (start, end) in enumerate(offset_mapping):
            assert start <= end, f"start {start} must be <= end {end}"
            # e.g., (0, 13), ..., (69, 70), ...
            has_reward_mask[i, j], reward_val[i, j] = is_cur_span_match_reward_span(
                start, end, rw_mapping
            )
    # check has_reward_mask is consistent with reward_val
    assert (~np.isnan(reward_val) == has_reward_mask).all(), \
        "has_reward_mask is not consistent with reward_val"
    return has_reward_mask, reward_val

def _annotate_str_with_color(s, reward_val):
    # if reward_val is nan, then it is not part of a reward span
    if np.isnan(reward_val):
        return s
    elif reward_val < 0:
        # red
        return f"\033[91m{s}\033[0m"
    elif reward_val > 0:
        # green
        return f"\033[92m{s}\033[0m"
    else:
        # gray (=0)
        return f"\033[90m{s}\033[0m"


def visualize_rm_example(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    reward_mask: np.ndarray,
    reward_val: np.ndarray,
    tokenizer: PreTrainedTokenizer,
    **kwargs
):
    # decode to get original text
    _tokens = tokenizer.convert_ids_to_tokens(input_ids)
    _tokens = [
        tokenizer.convert_tokens_to_string([t])
        for t in _tokens
    ]
    # convert tokens to actual text
    _tokens_w_reward = [
        # annotate t with color green if it is part of a reward span
        _annotate_str_with_color(t, r) if has_reward else t
        for t, has_reward, r, attn_mask in zip(
        _tokens, reward_mask, reward_val, attention_mask
        )
        if attn_mask == 1 # only consider tokens that are not padding
    ]
    print("".join(_tokens_w_reward))

from .training import build_data_instance_training
from .coarse_grain import build_data_instance_coarse_grained
from .postprocess_heuristic import build_data_instance_postprocess_heuristic
from .markline_heuristic import build_data_instance_markline_heuristic
