
if __name__ == '__main__':
    import os
    import json
    import glob
    import pathlib
    import argparse
    from tqdm import tqdm
    from typing import List, Tuple, Mapping, Dict
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-repo-dir', type=str, required=True)
    # input-repo-dir: git clone https://github.com/suzgunmirac/BIG-Bench-Hard.git
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--cot', action='store_true')
    args = parser.parse_args()

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    import json

    def generate_prompt(sample, idx, cot=False) -> str:
        prompt = sample["prompt"]
        # add task
        prompt += "\n" + "Q: " + sample["input"]
        if cot:
            prompt += "\nA: Let's think step by step."
        else:
            prompt += "\nA:"

        return {
            "id": idx,
            "text": prompt,
            "reference": sample["target"],
            # avoid overwriting the id and text fields
            **{k: v for k, v in sample.items() if k not in {"text", "id", "reference"}}
        }

    def save_to_jsonl(data, path):
        with open(path, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")

    if __name__ == "__main__":
        task_json_files = glob.glob(os.path.join(args.input_repo_dir, "bbh", "*.json"))

        name_to_task: Mapping[str, Tuple[str, str]] = {}
        for task_json_file in task_json_files:
            # Get names
            task_name = os.path.basename(task_json_file).replace(".json", "")

            # Read files
            with open(task_json_file, "r") as f:
                task_json = json.load(f)

            # Get prompt
            if args.cot:
                # directly read cot prompt
                cot_prompt_file = os.path.join(
                    args.input_repo_dir, "cot-prompts", f"{task_name}.txt"
                )
                assert os.path.exists(cot_prompt_file), f"Cannot find {cot_prompt_file}"
                with open(cot_prompt_file, "r") as f:
                    prompt = f.read()

                # remove canary in the prompt
                canary = task_json["canary"]
                prompt = prompt.replace(canary + "\n-----\n", "")
                prompt += "\n"
            else:
                # trying to find few-shot examples from codex file
                codex_template_file = os.path.join(
                    args.input_repo_dir,
                    "code-davinci-002-outputs",
                    "code-davinci-002-direct",
                    f"{task_name}_few_shot_template_0-255000.json",
                )
                assert os.path.exists(codex_template_file), \
                    f"Cannot find {codex_template_file}"
            
                with open(codex_template_file, "r") as f:
                    codex_template_json = json.load(f)
                    assert codex_template_json["canary"] == task_json["canary"]
                    # find the first few-shot prompt
                    _first_input = codex_template_json["outputs"][0]["input"]
                    prompt = _first_input[:_first_input.rfind("\nQ: ")]
                    # verify that the few-shot prompt is indeed a few-shot prompt for all
                    for _ex in codex_template_json["outputs"][1:]:
                        assert _ex["input"].startswith(prompt)
                # we've found the prompt!

            # e.g., [..., {'input': "Which of the following is a humorous edit of this artist or movie name: 'system of a down'?\nOptions:\n(A) system mof a down\n(B) shstem of a down\n(C) syster of a down\n(D) system of a gown", 'target': '(D)'} ...]
            name_to_task[task_name] = [
                {
                    "input": cur_example_dict["input"],
                    "target": cur_example_dict["target"],
                    "prompt": prompt,
                    "task_name": task_name,
                } for cur_example_dict in task_json["examples"]
            ]

        # Process test
        test_prompts = []
        id_counter = 0
        for task_name, task_info in name_to_task.items():
            for cur_task_info in task_info:
                test_prompts.append(
                    generate_prompt(cur_task_info, id_counter, cot=args.cot)
                )
                id_counter += 1
        save_to_jsonl(test_prompts, os.path.join(args.output_dir, "test_prompts.json"))

        # # analyze token length
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        # token_lens = []
        # for example in tqdm(test_prompts):
        #     token_lens.append(len(
        #         tokenizer(
        #         example["text"] + example["target"],
        #         )["input_ids"])
        #     )
        # import pandas as pd
        # print(pd.Series(token_lens).describe())
        # import pdb; pdb.set_trace()

        # For non-cot: 1536 should be more than enough
        # count    6511.000000
        # mean      407.677930
        # std       243.422712
        # min        79.000000
        # 25%       205.000000
        # 50%       378.000000
        # 75%       554.000000
        # max      1335.000000

        # For cot: (definitely need max token length of 2048)
        # count    6511.000000
        # mean      919.864844
        # std       350.497175
        # min       231.000000
        # 25%       773.000000
        # 50%       895.000000
        # 75%       989.000000
        # max      2070.000000
