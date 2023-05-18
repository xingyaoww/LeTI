
REFERNECE_PREFIX = """
import sys
sys.path.insert(0, "/root/lang-reward")
from leti.verifier.eae import EAESolutionVerifier
"""

if __name__ == '__main__':
    import os
    import re
    import json
    import pathlib
    import argparse
    from tqdm import tqdm
    from typing import List, Tuple
    from datasets import load_dataset
    from leti.verifier.eae.utils import construct_gold_set
    from leti.verifier.eae.nlp_utils import nlp

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', type=str, required=True)
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    import json

    def generate_prompt(sample, idx, fewshot=False) -> str:
        prompt = sample["full_prompt"]
        
        # Build gold_set for verifier
        tokens = [
            re.sub(r"\*\*(.+)\*\*", r"\1", token)
            for token in sample["sentence"].split()
        ]
        cur_event = sample["event_mentions"][sample["event_idx"]]
        gold_set, gold_canonical_set = construct_gold_set(
            tokens,
            nlp(" ".join(tokens)),
            cur_event,
            sample["entity_mentions"],
            head_only=True,
        )
        # gold_set_str = str(gold_set)

        var_name = f"{sample['cur_cls_name'].lower()}_event"

        reference = REFERNECE_PREFIX + "\n"
        reference += (
            f"verifier = EAESolutionVerifier(\n"
            f"    gold_set={str(gold_set)},\n"
            f"    cur_event={str(cur_event)},\n"
            f"    tokens={str(tokens)},\n"
            f")\n"
        )
        reference += f"verifier.verify({var_name})"

        return {
            "id": idx,
            "text": prompt,
            "reference": reference,
            # avoid overwriting the id and text fields
            **{k: v for k, v in sample.items() if k not in {"text", "id", "reference"}}
        }

    def generate_supervised_example(sample, idx) -> List[dict]:
        prompt = generate_prompt(sample, idx)["text"]
        gt_completed_code = sample["gt_instance_code"].replace(sample["instantiation_prompt"], "")
        text = prompt + gt_completed_code
        return [
            {
                "text": text,
                "id": idx,
                **{k: v for k, v in sample.items() if k not in {"text", "id"}}
            }
        ]

    def save_to_jsonl(data, path):
        with open(path, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")

    # load jsonl file
    train_ds = load_dataset("json", data_files=args.train_file)["train"]
    test_ds = load_dataset("json", data_files=args.test_file)["train"]

    # Sequence length statistics
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    # # tokenize "full_prompt"
    # ds = ds.map(lambda x: tokenizer(x["full_prompt"]), batched=False, remove_columns=["full_prompt"])
    # seq_lengths = list(map(len, ds["input_ids"]))
    # import pandas as pd
    # print(pd.Series(seq_lengths).describe())
    # # count     403.000000
    # # mean      888.004963
    # # std       129.022039
    # # min       651.000000
    # # 25%       737.500000
    # # 50%       927.000000
    # # 75%       993.500000
    # # max      1066.000000
    # # dtype: float64
    # import pdb; pdb.set_trace()
    # We should use 1280 as the max sequence length (1024 + 256)

    # Process train
    print(f"Processing {len(train_ds)} train examples")
    train_prompts = []
    train_examples = []
    for idx, sample in tqdm(enumerate(train_ds)):
        train_prompts.append(
            generate_prompt(sample, idx)
        )
        train_examples.extend(
            generate_supervised_example(sample, idx)
        )
    save_to_jsonl(train_prompts, os.path.join(args.output_dir, "train_prompts.json"))
    save_to_jsonl(train_examples, os.path.join(args.output_dir, "train.json"))

    # Process test examples
    print(f"Processing {len(test_ds)} test examples")
    test_examples = []
    for idx, sample in tqdm(enumerate(test_ds)):
        test_examples.extend(
            generate_supervised_example(sample, idx)
        )
    save_to_jsonl(test_examples, os.path.join(args.output_dir, "test.json"))

    # Process test
    print(f"Processing {len(test_ds)} test examples")
    test_prompts = []
    for idx, sample in tqdm(enumerate(test_ds)):
        test_prompts.append(
            generate_prompt(sample, idx)
        )
    save_to_jsonl(test_prompts, os.path.join(args.output_dir, "test_prompts.json"))
