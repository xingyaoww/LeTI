import os
import re
import random
import argparse
import subprocess
import tensorflow as tf
import pandas as pd
import streamlit as st

from leti.utils.eval import estimate_pass_at_k

# default wide mode
st.set_page_config(layout="wide")

parser = argparse.ArgumentParser()
parser.add_argument("--gs_root_dir", type=str, default="$GS_BUCKET_PREFIX/data/t5x-model")
args = parser.parse_args()

st.title("Intermediate Execution Result Visualizer")
st.write(f"Visualizing Execution Results in `{args.gs_root_dir}`")
# match glob pattern for google storage
glob_pattern = f"{args.gs_root_dir}/**/execution_results.json"
st.write(f"Matching glob pattern: `{glob_pattern}`")

def get_matched_files(refresh=False):
    """List all files that match the glob pattern in google storage."""
    TMP_MATCHED_FILES = "/tmp/matched_files.txt"
    if refresh or not os.path.exists(TMP_MATCHED_FILES):
        with st.spinner("Refreshing... (this may take a while)"):
            matched_files = subprocess.run(["gsutil", "ls", glob_pattern], capture_output=True)
            matched_files = matched_files.stdout.decode("utf-8").strip().split("\n")
            with open(TMP_MATCHED_FILES, "w") as f:
                f.write("\n".join(matched_files))
    else:
        with open(TMP_MATCHED_FILES, "r") as f:
            matched_files = f.read().strip().split("\n")
        st.write(f"Using cached matched files from `{TMP_MATCHED_FILES}`")
    st.markdown(f"Found **{len(matched_files)}** files.")
    return matched_files

# a button to refresh the matched files
matched_files = get_matched_files(refresh=st.button("Refresh Matched Files"))
st.markdown("---")

# select a part of the matched filepaths for display in the dropdown menu
# $GS_BUCKET_PREFIX/data/t5x-model/mbpp-ft/actor-rm-ppo/codegen-350M-mono/pheuristic-pt-350M+10x2+lr1e-5+0.99d/intermediate/epoch_8/train/execution_results.json
# only show actor-rm-ppo/codegen-350M-mono/pheuristic-pt-350M+10x2+lr1e-5+0.99d/intermediate/epoch_8/train

def get_display_name(filepath):
    """Get the display name for the file."""
    filepath = filepath.replace(args.gs_root_dir, "")
    filepath = filepath.strip("/")
    return filepath

display_files = [get_display_name(f) for f in matched_files]
filter_regex = st.text_input("Filter files (regex)", value=".*")

display_files = [f for f in display_files if re.match(filter_regex, f)]
# sort display files by epoch_number
display_files = sorted(
    display_files,
    key=lambda f: int(re.search(r"epoch_(\d+)", f).group(1)) \
        if "epoch_" in f else int(re.search(r"checkpoint_(\d+)", f).group(1))
)
st.markdown(f"Filtered to **{len(display_files)}** files.")

# dropdown menu for selecting a file
selected_name: str = st.selectbox("Select a execution result file for visualization", display_files)
st.write(f"Selected file: `{selected_name}`")

# get the full path of the selected file

def get_execution_result_df(selected_filepath):
    with st.spinner(f"Loading file from Google Storage: `{selected_filepath}`"):
        with tf.io.gfile.GFile(selected_filepath, "r") as f:
            df = pd.read_json(f, orient="records", lines=True)
    return df

selected_filepath = os.path.join(args.gs_root_dir, selected_name)
df = get_execution_result_df(selected_filepath)
st.markdown(f"Loaded **{len(df)}** records from the execution result file.")

# add a button to download the dataframe as jsonl
if st.button("Download Execution Result as JSONL"):
    st.markdown(f"Downloading **{len(df)}** records as JSONL.")
    df.to_json("/tmp/cur_execution_result.jsonl", orient="records", lines=True)
    st.download_button(
        "Download",
        data=open("/tmp/cur_execution_result.jsonl", "rb").read(),
        file_name="cur_execution_result.jsonl"
    )

# compute pass@k
task_success = df.groupby("prompt_idx")["execution_result"].agg(
    lambda gen_results_for_prompt: [
        gen_res["success"] for gen_res in gen_results_for_prompt
    ]
)
correct = task_success.apply(sum).to_numpy()
total = task_success.apply(len).to_numpy()
pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in [1, 10, 100] if (total >= k).all()}
st.markdown(f"pass@k")
st.json(pass_at_k)

# analysis of the execution results
st.markdown("---")
success_rate_for_prompt = task_success.apply(lambda x: sum(x) / len(x))
st.markdown("Success Rate for Prompt Idx")
# st.bar_chart(success_rate_for_prompt)
# sort x-axis by success rate
success_rate_for_prompt = success_rate_for_prompt.sort_values(ascending=False)
st.bar_chart(success_rate_for_prompt)
st.markdown("---")

# randomly sample a row index using a button
row_index = st.button("Sample a row index")
if row_index:
    row_index = random.randint(0, len(df) - 1)

# a input box for manually inputting a row index (populated by the sampled row index)
row_index = st.number_input("Row index", value=row_index, min_value=0, max_value=len(df) - 1, step=1)

# visualize the selected row
cur_row = df.iloc[row_index]
st.markdown(f"Visualizing row **{row_index}**")
st.write(cur_row[["prompt_idx", "generation_idx"]])

# visualize the prompt (using code block)
st.markdown("Prompt:")
st.code(cur_row["prompt"], language="python")

# visualize the generated code (using code block)
st.markdown("Generated Code:")
st.code(cur_row["generation"], language="python")
st.markdown("Test (Reference):")
st.code(cur_row["reference"], language="python")

# visualize the execution result (using json)
st.markdown("Execution Result:")
st.json(cur_row["execution_result"])

if "original_generation" in cur_row:
    st.markdown("Original Generation:")
    st.code(cur_row["original_generation"], language="python")

if "postprocessed_generation" in cur_row:
    st.markdown("Postprocessed Generation:")
    st.code(cur_row["postprocessed_generation"], language="python")

if "additional_feedback" in cur_row:
    st.markdown("Additional Feedback:")
    st.markdown(cur_row["additional_feedback"])
