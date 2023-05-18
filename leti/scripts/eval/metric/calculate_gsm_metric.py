import re
import os
import json
import argparse
import pandas as pd
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=str)
args = parser.parse_args()

args.is_cot = "cot" in args.input_dir
print("Is COT: ", args.is_cot)

generation_json_filepath = os.path.join(args.input_dir, "generations.json")
reference_json_filepath = os.path.join(args.input_dir, "references.json")
output_results_filepath = os.path.join(args.input_dir, "results.json")

with tf.io.gfile.GFile(generation_json_filepath, "r") as f:
    generations = pd.read_json(f)
with tf.io.gfile.GFile(reference_json_filepath, "r") as f:
    references = pd.read_json(f)

generations = generations.rename(
    columns={0: "prompt_idx", 1: "generation_idx", 2: "generation"}
)
references = references.rename(
    columns={0: "prompt_idx", 1: "generation_idx", 2: "reference"}
)

assert generations.shape == references.shape, "The generations and references should have the same shape"

# Merge the generations and references
data = generations.merge(references, on=["prompt_idx", "generation_idx"])

def extract_answer(generation):
    # https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    ANS_RE = re.compile(r"The answer is (\-?[\$0-9\.\,]+)")
    match = ANS_RE.search(generation)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        match_str = match_str.replace("$", "")
        match_str = match_str.rstrip(".") # remove trailing period
        return match_str
    else:
        return None

def check_match(predicted_answer, reference) -> bool:
    # gt_answer = extract_answer(gt_example["answer"])
    # assert gt_answer != INVALID_ANS
    # return extract_answer(model_completion) == gt_answer
    assert reference and int(reference) == reference, "The reference should be an integer"
    if not predicted_answer:
        return False
    predicted_answer = int(float(predicted_answer))
    return predicted_answer == reference

n_examples_no_output_due_to_seq_len = (data["generation"].apply(lambda x: bool(x) == False)).sum()
assert n_examples_no_output_due_to_seq_len == 0, "There should be no examples with no output due to sequence length"

data["answer"] = data["generation"].apply(extract_answer)
data["match"] = data.apply(lambda row: check_match(row["answer"], row["reference"]), axis=1)

exact_match = data["match"].mean()

with tf.io.gfile.GFile(output_results_filepath, "w") as f:
    f.write(json.dumps({
        "exact_match": exact_match
    }))

print("Number of examples with no output due to sequence length: ", n_examples_no_output_due_to_seq_len)
print("Exact match: ", exact_match)
print(f"Results saved to {output_results_filepath}")
