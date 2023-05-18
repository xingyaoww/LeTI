import os
import re
import json
import spacy
import argparse
import pandas as pd
import tensorflow as tf

from spacy.tokens import Doc
from collections import defaultdict
from typing import Set, List, Tuple
from leti.utils.execute import exec_code
from leti.verifier.eae.utils import (
    convert_event_obj_to_dict,
    construct_gold_set,
    construct_pred_set,
    match_pred_gold_sets
)
from leti.verifier.eae.nlp_utils import nlp

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=str)
parser.add_argument("--testset-filepath", type=str, default="data/processed/finetune/eae/test_prompts.json")
args = parser.parse_args()

# Load the data
with tf.io.gfile.GFile(args.testset_filepath, "r") as f:
    testset_df = pd.read_json(f, orient="records", lines=True)

generation_json_filepath = os.path.join(args.input_dir, "generations.json")
reference_json_filepath = os.path.join(args.input_dir, "references.json")
with tf.io.gfile.GFile(generation_json_filepath, "r") as f:
    generations = pd.read_json(f)
generations = generations.rename(
    columns={0: "prompt_idx", 1: "generation_idx", 2: "generation"}
)
with tf.io.gfile.GFile(reference_json_filepath, "r") as f:
    references = pd.read_json(f)
references = references.rename(
    columns={0: "prompt_idx", 1: "generation_idx", 2: "reference"}
)
output_results_filepath = os.path.join(args.input_dir, "results.json")

assert generations.shape == references.shape, "The generations and references should have the same shape"

# Merge the generations and references
data = generations.merge(references, on=["prompt_idx", "generation_idx"])
data = data.sort_values(by=["prompt_idx", "generation_idx"])
# Combine data with testset_df
data = data.merge(testset_df.rename(columns={"id": "prompt_idx"}), on="prompt_idx")


def postprocess_fn(instance_code: str):
    # Remove code after """, class, print, or #
    STOP_WORDS = ['"""', 'class', 'print', '#']
    instance_code = re.split('|'.join(STOP_WORDS), instance_code)[0]
    return instance_code


data["full_code"] = data["full_prompt"] + data["generation"].apply(lambda x: postprocess_fn(x))

def parse_through_execution(full_code: str, cur_cls_name: str):
    """

    Example instance_code:
    ```
    nominate_event = Nominate(
    agent=[
        GPE('British Chancellor of the Exchequer Gordon Brown on Tuesday named the current head of the country '),
        ORG('s energy regulator as the new chairman of finance watchdog the Financial Services Authority ( FSA )'),
    ],
    person=[
        PER('British Chancellor of the Exchequer Gordon Brown on Tuesday named the current head of the country '),
    ],
    )
    ```
    """
    variable_name = f"{cur_cls_name.lower()}_event"

    exec_res = exec_code(
        full_code,
        do_trace=False,
        return_globals=True,
        extra_headers="from __future__ import annotations"
    )

    if exec_res["success"]:
        gvars = exec_res["globals"]
        event_var = gvars[variable_name]

        # each key is a role, each value is a list of object
        # convert to a list of (role, [(entity_type, list of tokens), ...])
        try:
            event_dict = convert_event_obj_to_dict(event_var, split_role_value=True)
        except Exception as e:
            event_dict = {}
    else:
        event_dict = {}

    exec_res["event_dict"] = event_dict
    return exec_res


data["exec_res"] = data.apply(
    lambda ex: parse_through_execution(ex["full_code"], ex["cur_cls_name"]),
    axis=1
)
data["predicted_args"] = data["exec_res"].apply(lambda x: x["event_dict"])

data["tokens"] = data.apply(
    # replace a word that looks like "**word**" (trigger) with "word"
    lambda ex: [re.sub(r"\*\*(.+)\*\*", r"\1", token) for token in ex["sentence"].split()],
    axis=1
)

data["cur_event"] = data.apply(lambda ex: ex["event_mentions"][ex["event_idx"]], axis=1)


_predset = data.apply(
    lambda x: construct_pred_set(
        x["predicted_args"],
        x["cur_event"],
        x["tokens"],
        nlp(' '.join(x['tokens'])),
        head_only=True,
        nlp=nlp
    ),
    axis=1
)
data["predicted_set"], data["not_matched_pred_args"] = zip(*_predset)

_goldset = data.apply(
    lambda x: construct_gold_set(
        x["tokens"],
        nlp(' '.join(x['tokens'])),
        x["cur_event"],
        x["entity_mentions"],
        head_only=True
    ),
    axis=1
)
data["gold_set"], data["gold_canonical_set"] = zip(*_goldset)

data = pd.concat([
    data,
    data.apply(
        lambda x: match_pred_gold_sets(x["predicted_set"], x["gold_set"]),
        axis=1
    ).apply(pd.Series).fillna(0)
], axis=1)

data["pred_arg_num"] = data["predicted_set"].apply(len)
data["gold_arg_num"] = data["gold_set"].apply(len)

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1

arg_i_precision, arg_i_recall, arg_i_f1 = compute_f1(
    data["pred_arg_num"].sum(),
    data["gold_arg_num"].sum(),
    data["arg_idn_num"].sum()
)

arg_c_precision, arg_c_recall, arg_c_f1 = compute_f1(
    data["pred_arg_num"].sum(),
    data["gold_arg_num"].sum(),
    data["arg_class_num"].sum()
)

results = {
    "arg_i_precision": arg_i_precision,
    "arg_i_recall": arg_i_recall,
    "arg_i_f1": arg_i_f1,

    "arg_c_precision": arg_c_precision,
    "arg_c_recall": arg_c_recall,
    "arg_c_f1": arg_c_f1,
}

print(f"Argument Identification: Precision: {arg_i_precision:.4f}, Recall: {arg_i_recall:.4f}, F1: {arg_i_f1:.4f}")
print(f"Argument Classification: Precision: {arg_c_precision:.4f}, Recall: {arg_c_recall:.4f}, F1: {arg_c_f1:.4f}")

with tf.io.gfile.GFile(output_results_filepath, "w") as f:
    json.dump(results, f)
