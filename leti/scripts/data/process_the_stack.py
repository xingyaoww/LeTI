import argparse
import datasets
import tensorflow as tf
from tqdm import tqdm

def get_dataset(n_data, split="train"):
    ds = datasets.load_dataset("bigcode/the-stack", data_dir="data/python", split=split, streaming=True)

    rows = []
    for i, sample in tqdm(enumerate(iter(ds)), total=n_data):
        if i >= n_data:
            break
        rows.append(sample)

    # make datasets a huggingface dataset
    dataset = datasets.Dataset.from_list(rows)
    return dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gs_project', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

GS_STORAGE_OPTIONS = {"project": args.gs_project}
                                                 
ds_50k = get_dataset(50_000, split="train")
ds_50k.save_to_disk(args.output_dir, storage_options=GS_STORAGE_OPTIONS)
