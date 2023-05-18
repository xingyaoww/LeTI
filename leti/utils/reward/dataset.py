import random
import functools
import pandas as pd
from tqdm import tqdm
from typing import List

tqdm.pandas()

def process_dfs(
    dfs: List[pd.DataFrame],
    df_paths: List[str],
    build_data_instance_fn: callable,
    include_input=False
):
    # set random seed
    random.seed(42)

    print(f"Processing {len(dfs)} dataframes...")
    data_instances = []
    for i, df in enumerate(dfs):
        cur_df_path = df_paths[i]
        build_data_instance_partial_fn = functools.partial(
            build_data_instance_fn, include_input=include_input, cur_df_path=cur_df_path
        )
        for instance in df.progress_apply(build_data_instance_partial_fn, axis=1):
            if instance is not None:
                if isinstance(instance, list):
                    data_instances.extend(instance)
                else:
                    assert isinstance(instance, dict)
                    data_instances.append(instance)
    
    dataset = pd.DataFrame(data_instances)
    # deduplicate
    print("Dataset before deduplication:", len(dataset))
    dataset = dataset.drop_duplicates(
        subset=["text_a", "text_b"]
    ).reset_index(drop=True)
    dataset["id"] = dataset.index
    print("Dataset after deduplication:", len(dataset))
    return dataset
