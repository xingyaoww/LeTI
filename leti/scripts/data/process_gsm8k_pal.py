# Following https://github.com/reasoning-machines/pal/blob/main/pal/prompt/math_prompts.py

MATH_PROMPT = '''
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
# solution in Python:
def solution():
    """Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
# solution in Python:
def solution():
    """Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"""
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
# solution in Python:
def solution():
    """There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"""
    computers_initial = 9
    computers_per_day = 5
    num_days = 4  # 4 days between monday and thursday
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
# solution in Python:
def solution():
    """Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"""
    toys_initial = 5
    mom_toys = 2
    dad_toys = 2
    total_received = mom_toys + dad_toys
    total_toys = toys_initial + total_received
    result = total_toys
    return result
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
# solution in Python:
def solution():
    """Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"""
    jason_lollipops_initial = 20
    jason_lollipops_after = 12
    denny_lollipops = jason_lollipops_initial - jason_lollipops_after
    result = denny_lollipops
    return result
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
# solution in Python:
def solution():
    """Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"""
    leah_chocolates = 32
    sister_chocolates = 42
    total_chocolates = leah_chocolates + sister_chocolates
    chocolates_eaten = 35
    chocolates_left = total_chocolates - chocolates_eaten
    result = chocolates_left
    return result
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
# solution in Python:
def solution():
    """If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"""
    cars_initial = 3
    cars_arrived = 2
    total_cars = cars_initial + cars_arrived
    result = total_cars
    return result
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
# solution in Python:
def solution():
    """There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"""
    trees_initial = 15
    trees_after = 21
    trees_added = trees_after - trees_initial
    result = trees_added
    return result
Q: {question}
# solution in Python:
'''.strip() + '\n\n\n'

# consumes 1181 tokens for Salesforce/codegen-350M-mono
# total 512 tokens budget should be enough?

if __name__ == '__main__':
    import os
    import json
    import pathlib
    import argparse
    from tqdm import tqdm
    from typing import List, Tuple
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    # input-file should either be gsm.jsonl or gsmhardv2.json
    # from https://github.com/reasoning-machines/pal
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    import json

    def generate_prompt(sample, idx, fewshot=False) -> str:
        # e.g., {'input': "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", 'target': 18}
        question = sample["input"]
        prompt = MATH_PROMPT.format(question=question)
        
        # build reference (i.e., testcases)
        answer = sample["target"]
        reference = f"\nassert solution() == {answer}\n"

        return {
            "id": idx,
            "text": prompt,
            "reference": reference,
            # avoid overwriting the id and text fields
            **{k: v for k, v in sample.items() if k not in {"text", "id", "reference"}}
        }

    def save_to_jsonl(data, path):
        with open(path, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")

    if __name__ == "__main__":
        # load jsonl file
        ds = load_dataset("json", data_files=args.input_file)["train"]

        # Process test
        print(f"Processing {len(ds)} test examples")
        test_prompts = []
        for idx, sample in tqdm(enumerate(ds)):
            test_prompts.append(
                generate_prompt(sample, idx)
            )
        save_to_jsonl(test_prompts, os.path.join(args.output_dir, "test_prompts.json"))

        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        # token_lens = []
        # for example in tqdm(test_prompts):
        #     token_lens.append(len(
        #         tokenizer(example["text"] + example["reference"])["input_ids"]
        #     ))
        # import pandas as pd
        # print(pd.Series(token_lens).describe())
        # import pdb; pdb.set_trace()

        # count    1319.000000
        # mean     1241.650493
        # std        21.561042
        # min      1206.000000
        # 25%      1226.000000
        # 50%      1237.000000
        # 75%      1252.000000
        # max      1367.000000
        # dtype: float64

        # 1536 tokens should be enough..

