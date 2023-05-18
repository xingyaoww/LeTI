import os
import json
import pathlib
import argparse
from tqdm import tqdm
from typing import List, Tuple
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=str, required=True)
args = parser.parse_args()

pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

import json

def generate_prompt(sample) -> str:
    # https://github.com/bigcode-project/bigcode-evaluation-harness/blob/d61afde130005ecc65cf800ad8eca790a9bc2115/lm_eval/tasks/humaneval.py#L49

    prompt = sample["prompt"].strip()
    
    # build reference (i.e., testcases)
    test_func = sample["test"]
    entry_point = f"check({sample['entry_point']})"
    reference = "\n" + test_func + "\n" + entry_point

    return {
        "text": prompt,
        "id": sample["task_id"],
        "reference": reference,
        # avoid overwriting the id and text fields
        **{k: v for k, v in sample.items() if k not in {"text", "id", "reference"}}
    }

def save_to_jsonl(data, path):
    with open(path, "w") as f:
        for example in data:
            f.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    splits = ["test"]
    ds = load_dataset(path="openai_humaneval")

    # Process test
    print(f"Processing {len(ds['test'])} test examples")
    test_prompts = []
    for sample in tqdm(ds["test"]):
        test_prompts.append(
            generate_prompt(sample)
        )

    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    # token_lens = []
    # for example in tqdm(test_prompts):
    #     token_lens.append(len(
    #         tokenizer(example["text"] + example["canonical_solution"])["input_ids"]
    #     ))
    # import pandas as pd
    # print(pd.Series(token_lens).describe())
    # import pdb; pdb.set_trace()

    # count    164.000000
    # mean     206.926829
    # std       97.946120
    # min       51.000000
    # 25%      139.000000
    # 50%      193.000000
    # 75%      258.250000
    # max      621.000000
    # dtype: float64
    # Should do 768 max tokens for humaneval

    save_to_jsonl(test_prompts, os.path.join(args.output_dir, "test_prompts.json"))
