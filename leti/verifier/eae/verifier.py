from .utils import convert_event_obj_to_dict, match_pred_gold_sets

import requests
from typing import Dict, Any, List

def construct_pred_set_via_api(
    predicted_args: Dict[str, Any],
    cur_event: Dict[str, Any],
    tokens: List[str],
    api_url: str = "http://localhost:5000/api",
) -> Dict[str, Any]:
    # This is a solution to circumvent the issue of importing spacy in the pseudo-sandbox
    # which involves calling subprocess.Popen() which is dangerous.
    response = requests.post(
        api_url,
        json={
            "predicted_args": predicted_args,
            "cur_event": cur_event,
            "tokens": tokens,
        }
    )
    response.raise_for_status()
    ret = response.json()
    return {
        "predicted_set": eval(ret["predicted_set"]),
        "not_matched_pred_args": eval(ret["not_matched_pred_args"]),
    }


class EAESolutionVerifier:
    def __init__(
        self,
        gold_set: set,
        cur_event: Dict[str, Any],
        tokens: List[str],
    ) -> None:
        self.gold_set = gold_set
        self.cur_event = cur_event
        self.tokens = tokens

    def verify(self, event_obj):
        predicted_args = convert_event_obj_to_dict(
            event_obj,
            raise_error=True
        ) # 1. Convert event_obj to a dict

        res = construct_pred_set_via_api(
            predicted_args,
            self.cur_event,
            self.tokens,
        )
        predicted_set, not_matched_pred_args = res['predicted_set'], res['not_matched_pred_args']

        match_result = match_pred_gold_sets(
            predicted_set,
            self.gold_set,
            tokens=self.tokens,
            raise_error_when_not_matched=True
        )
