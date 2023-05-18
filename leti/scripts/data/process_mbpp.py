import os
import json
import pathlib
import argparse
from tqdm import tqdm
from typing import List, Tuple
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=str, required=True)
parser.add_argument('--addtional-prompt', type=str, default="")
parser.add_argument('--addtional-feedback', type=str, default="")
args = parser.parse_args()

pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

import json

def generate_prompt(sample) -> str:
    # https://github.com/bigcode-project/bigcode-evaluation-harness/blob/d61afde130005ecc65cf800ad8eca790a9bc2115/lm_eval/tasks/mbpp.py#L57

    description = sample["text"]
    test_example = sample["test_list"][0]
    prompt = f'{args.addtional_feedback}"""\n{description}\n{test_example}\n{args.addtional_prompt}"""'

    return {
        "text": prompt,
        "id": sample["task_id"],
        "reference": "\n".join(sample["test_list"]),
        # avoid overwriting the id and text fields
        **{k: v for k, v in sample.items() if k not in {"text", "id", "reference"}}
    }

def generate_supervised_example(sample) -> List[dict]:
    prompt = generate_prompt(sample)["text"]
    solution = sample["code"]
    return [
        {
            "text": prompt + "\n" + solution,
            "id": sample["task_id"],
            **{k: v for k, v in sample.items() if k not in {"text", "id"}}
        }
    ]

def save_to_jsonl(data, path):
    with open(path, "w") as f:
        for example in data:
            f.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    splits = ["train", "validation", "test"]
    ds = load_dataset("mbpp")

    # Process Prompts for In-context Learning
    in_context_examples = []
    for sample in tqdm(ds["prompt"]):
        in_context_examples.extend(
            generate_supervised_example(sample)
        )
    save_to_jsonl(in_context_examples, os.path.join(args.output_dir, "in_context_examples.json"))

    # concatenate 1, 3, 5, 10 shots prompts
    k_to_prompt = {}
    Ks = [1, 3, 5, 10]
    for k in Ks:
        k_to_prompt[k] = ""
        for sample in in_context_examples[:k]:
            k_to_prompt[k] += sample["text"] + "\n\n"

    # from transformers import CodeGenTokenizer
    # tokenizer = CodeGenTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    # for k in Ks:
    #     print(f"k-shot {k}: {len(tokenizer(k_to_prompt[k])['input_ids'])} tokens")
    # import pdb; pdb.set_trace()
    # k-shot
    # 1-shot 301 tokens -> 896 tokens
    # 3-shot: 481 tokens -> max 1024 tokens
    # 5-shot: 770 tokens -> max 1280 tokens
    # 10-shot: 1260 tokens -> max 1792 tokens

    # Process train
    print(f"Processing {len(ds['train'])} training examples")
    train_examples = []
    train_prompts = []
    for sample in tqdm(ds["train"]):
        train_examples.extend(
            generate_supervised_example(sample)
        )
        train_prompts.append(
            generate_prompt(sample)
        )
    save_to_jsonl(train_examples, os.path.join(args.output_dir, "train.json"))
    save_to_jsonl(train_prompts, os.path.join(args.output_dir, "train_prompts.json"))

    # Process validation
    print(f"Processing {len(ds['validation'])} validation examples")
    validation_examples = []
    validation_prompts = []
    for sample in tqdm(ds["validation"]):
        validation_examples.extend(
            generate_supervised_example(sample)
        )
        validation_prompts.append(
            generate_prompt(sample)
        )
    save_to_jsonl(validation_examples, os.path.join(args.output_dir, "val.json"))
    save_to_jsonl(validation_prompts, os.path.join(args.output_dir, "val_prompts.json"))

    # Process test
    print(f"Processing {len(ds['test'])} test examples")
    test_prompts = []
    for sample in tqdm(ds["test"]):
        test_prompts.append(
            generate_prompt(sample)
        )
    save_to_jsonl(test_prompts, os.path.join(args.output_dir, "test_prompts.json"))

    # Generate k-shot test prompts (1, 3, 5, 10)
    for k in Ks:
        cur_prompts = []
        for sample in test_prompts:
            cur_prompts.append(
                {
                    "text": k_to_prompt[k] + sample["text"],
                    "id": sample["id"],
                    **{k: v for k, v in sample.items() if k not in {"text", "id"}}
                }
            )
        save_to_jsonl(
            cur_prompts,
            os.path.join(args.output_dir, f"test_prompts_{k}_shot.json")
        )
