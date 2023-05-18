import os
import json
import argparse
import pandas as pd
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=str)
parser.add_argument("--data_dir", type=str, default="data/processed/eval")
args = parser.parse_args()

args.is_cot = "cot" in args.input_dir
print("Is COT: ", args.is_cot)

generation_json_filepath = os.path.join(args.input_dir, "generations.json")
reference_json_filepath = os.path.join(args.input_dir, "references.json")
output_results_filepath = os.path.join(args.input_dir, "results.json")
prompts_filepath = os.path.join(args.data_dir, "bbh" if not args.is_cot else "bbh_cot", "test_prompts.json")

with tf.io.gfile.GFile(prompts_filepath, "r") as f:
    prompts = pd.read_json(f, orient="records", lines=True)
prompt_id_to_task_name = prompts[["id", "task_name"]].set_index("id").to_dict()["task_name"]

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

# check accuracy (i.e., whether the generation lstrip().startswith() the reference .lstrip())
def check_match(generation, reference) -> bool:
    generation = generation.lstrip()
    reference = reference.lstrip()
    return generation.startswith(reference)

def check_cot_match(generation, reference) -> bool:
    generation = generation.lstrip().split("Q:")[0].strip()
    reference = reference.strip()
    return reference in generation

n_examples_no_output_due_to_seq_len = (data["generation"].apply(lambda x: bool(x) == False)).sum()

if args.is_cot:
    data["match"] = data.apply(lambda row: check_cot_match(row["generation"], row["reference"]), axis=1)
else:
    data["match"] = data.apply(lambda row: check_match(row["generation"], row["reference"]), axis=1)

data["task_name"] = data["prompt_idx"].apply(lambda x: prompt_id_to_task_name[x])

exact_match_by_task = data.groupby("task_name")["match"].mean()
exact_match = data["match"].mean()
exact_match_exclude_no_output = data.loc[data["generation"].apply(bool), "match"].mean()
n_examples_no_output_due_to_seq_len = int((data["generation"].apply(lambda x: bool(x) == False)).sum())
# prompt_idx where generation is empty
empty_generation_prompt_idxs = data.loc[data["generation"].apply(lambda x: bool(x) == False), "prompt_idx"].unique()

with tf.io.gfile.GFile(output_results_filepath, "w") as f:
    f.write(json.dumps({
        "exact_match": exact_match,
        "exact_match_by_task": exact_match_by_task.to_dict(),
        "exact_match_exclude_no_output": exact_match_exclude_no_output,
        "n_examples_no_output_due_to_seq_len": n_examples_no_output_due_to_seq_len,
        "empty_generation_prompt_idxs": empty_generation_prompt_idxs.tolist()
    }))

print("Number of examples with no output due to sequence length: ", n_examples_no_output_due_to_seq_len)
print("Exact match: ", exact_match)
print("Exact match by task: ", exact_match_by_task)
print("Exact match (exclude no output): ", exact_match_exclude_no_output)
print("Number of prompts with no output: ", len(empty_generation_prompt_idxs))
print(f"Results saved to {output_results_filepath}")
