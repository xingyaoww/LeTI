import os
import re
import jax
import json
import numpy as np
import datasets
import pandas as pd
import functools
import tensorflow as tf

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import HfArgumentParser, set_seed
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from tqdm import tqdm
from jax_smi import initialise_tracking
initialise_tracking()

from leti.models.jax_inferencer import Inferencer
from leti.utils.execute.executor import Executor, future_tasks_iter_fn
from leti.utils.eval import estimate_pass_at_k
from leti.utils.dataloader import generation_dataloader

set_seed(42)

@dataclass
class EvalArguments:
    """
    Configuration for running the evaluation.
    """
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Sample from the language model's output distribution."},
    )
    temperature: Optional[float] = field(
        default=0.2, metadata={"help": "Sampling temperature used for generation."}
    )
    top_k: Optional[int] = field(
        default=0, metadata={"help": "Top-k parameter used for generation."}
    )
    top_p: Optional[float] = field(
        default=1, metadata={"help": "Top-p parameter used for nucleus sampling."}
    )
    n_samples: Optional[int] = field(
        default=1,
        metadata={"help": "Number of completions to generate for each sample."},
    )
    eos: Optional[str] = field(
        default="<|endoftext|>", metadata={"help": "end of sentence token."}
    )
    postprocess_type: Optional[str] = field(
        default=None,
        metadata={"help": "Postprocess type for the output."},
        # choose from ["mbpp", "humaneval"]
    )
    conditioned_on_good: Optional[bool] = field(
        default=False,
        metadata={"help": "Condition on good rw token."},
    )
    conditioned_on_bad: Optional[bool] = field(
        default=False,
        metadata={"help": "Condition on bad rw token."},
    )
    disable_execution: Optional[bool] = field(
        default=False,
        metadata={"help": "Disable execution of the generated code."},
    )

RW_GOOD_TOKEN = "<|good|>"
RW_BAD_TOKEN = "<|bad|>"

def parse_args():
    parser = HfArgumentParser(EvalArguments)
    parser.add_argument(
        "--infer_data",
        type=str
    )
    parser.add_argument(
        "--hf_ckpt",
        type=str,
        default="data/models/flax-pretrained/actor/codegen-350M-mono"
    )
    parser.add_argument(
        "--t5x_path",
        type=str,
        default="$GS_BUCKET_PREFIX/data/t5x-model/flax-pretrained/actor/codegen-350M-mono/checkpoint_0"
    )
    parser.add_argument(
        "--gs_output_dir",
        type=str
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        default=1
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=128, # to avoid repeated compilation
    )
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()

def mbpp_postprocess(string):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    STOP_WORDS = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/"]
    return re.split("|".join(STOP_WORDS), string)[0].rstrip()

def humaneval_postprocess(string):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    STOP_WORDS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
    # Remove the last block of the code containing stop_words for HumanEval
    string_list = re.split("(%s)" % "|".join(STOP_WORDS), string)
    # last string should be ""
    return "".join(string_list[:-2])

def gsm_pal_postprocess(string):
    # split using "\nQ:", remove anything after the "\nQ:"
    return string.split("\nQ:")[0]

args = parse_args()

assert not (args.conditioned_on_good and args.conditioned_on_bad), "Cannot condition on both good and bad rw token"
if args.conditioned_on_good:
    print(f"Generation is conditioned on GOOD rw token")
    conditioned_txt = RW_GOOD_TOKEN
elif args.conditioned_on_bad:
    print(f"Generation is conditioned on BAD rw token")
    conditioned_txt = RW_BAD_TOKEN
else:
    conditioned_txt = ""


if args.postprocess_type == "mbpp":
    postprocess_fn = mbpp_postprocess
    print("Using MBPP postprocessing")
elif args.postprocess_type == "humaneval":
    postprocess_fn = humaneval_postprocess
    print("Using HumanEval postprocessing")
elif args.postprocess_type == "gsm_pal":
    postprocess_fn = gsm_pal_postprocess
    print("Using GSM postprocessing")
else:
    print(f"WARNING: No postprocessing is specified")
    postprocess_fn = None

# Ready output file
with tf.io.gfile.GFile(args.gs_output_dir + "/infer_args.json", "w") as f:
    json.dump(args.__dict__, f, indent=2)

# Initialize generator
generator = Inferencer(
    args.hf_ckpt,
    args.t5x_path,
    num_partitions=args.num_partitions
)
generator.load_model_and_params()

generation_config = {
    "do_sample": args.do_sample,
    "temperature": args.temperature,
    "top_p": args.top_p,
    "top_k": args.top_k,
    "num_beam": 1,
    "max_length": args.max_len,
}
if args.temperature == 0:
    # If temperature is 0, we set do_sample to False (greedy decoding)
    generation_config["do_sample"] = False
    generation_config["temperature"] = None
    generation_config["num_beam"] = 1

print(f"Generation config: {generation_config}")
# Load datasets
dataset = datasets.load_dataset("json", data_files=args.infer_data)["train"]

if conditioned_txt != "":
    # add conditioned_txt to the "text"
    dataset = dataset.map(lambda x: {"text": conditioned_txt + x["text"]})
    # print out 1 examples
    print("Example prompt:")
    print(dataset[0]["text"])

# remove langfb tokens for execution
LANG_FEEDBACK_BEGIN_TOKEN = "<|lang_feedback|>"
LANG_FEEDBACK_END_TOKEN = "<|/lang_feedback|>"

def remove_lang_feedback_span(string):
    """Remove lang feedback span from string.

    The span is enclosed by LANG_FEEDBACK_BEGIN_TOKEN and LANG_FEEDBACK_END_TOKEN.
    """
    # Do not interpret LANG_FEEDBACK_BEGIN_TOKEN and LANG_FEEDBACK_END_TOKEN as regex
    return re.sub(
        re.escape(LANG_FEEDBACK_BEGIN_TOKEN) + ".*" + re.escape(LANG_FEEDBACK_END_TOKEN),
        "",
        string
    )
dataset = dataset.map(lambda x: {"text": remove_lang_feedback_span(x["text"])})
print("Example prompt after removing lang feedback span:")
print(dataset[0]["text"])

# Generate solutions for each prompt in the dataset
generations = []
references = []
tasks = []

rng = jax.random.PRNGKey(42)
solution_executor = Executor(
    process_id=0,
    num_workers=os.cpu_count() // 2,
    disable_execution=args.disable_execution
)

if postprocess_fn is not None:
    postprocess_executor = Executor(process_id=0, num_workers=os.cpu_count() // 2)
    print("Also calculating postprocess metrics")

dataloader = generation_dataloader(
    dataset,
    args.n_samples,
    args.batch_size,
    generator.tokenizer
)
pbar = tqdm(total=len(dataset) * args.n_samples)

for batch in dataloader:
    # update rng for each batch
    rng, generation_rng = jax.random.split(rng)
    cur_generation_ids: jax.Array = generator.generate_fast(
        batch,
        generation_rng=generation_rng,
        generation_config=generation_config
    )

    def task_callback_fn(cur_task: dict):
        # this will be called in a multi-threaded context
        # so we need to be careful about thread safety
        # list.append() is thread-safe
        generations.append((
            cur_task["prompt_idx"],
            cur_task["generation_idx"],
            cur_task["generation"]
        ))
        references.append((
            cur_task["prompt_idx"],
            cur_task["generation_idx"],
            cur_task["reference"]
        ))
        tasks.append(cur_task)
        pbar.update(1)

    solution_executor.add_future_tasks(
        functools.partial(
            future_tasks_iter_fn,
            cur_generation_ids,
            batch,
            generator.tokenizer,
            task_callback_fn=task_callback_fn,
            postprocess_fn=None,
            prompt_processing_fn=lambda x: x.lstrip(conditioned_txt) if args.postprocess_type != "gsm_pal" else ""
        ),
        delay_execution=False # execute immediately
    )
    if postprocess_fn is not None:
        postprocess_executor.add_future_tasks(
            functools.partial(
                future_tasks_iter_fn,
                cur_generation_ids,
                batch,
                generator.tokenizer,
                task_callback_fn=None, # do not call task_callback_fn again
                postprocess_fn=postprocess_fn,
                prompt_processing_fn=lambda x: x.lstrip(conditioned_txt) if args.postprocess_type != "gsm_pal" else ""
            ),
            delay_execution=False # execute immediately
        )
pbar.close()

# write a json encoder that can handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

if not args.disable_execution:
    # Execute the solutions
    def save_executor_result(executor: Executor, suffix=""):
        execution_results: List[Dict] = executor.get_results_and_reset()
        # sort execution results by prompt_idx and generation_idx
        execution_results.sort(key=lambda x: (x["prompt_idx"], x["generation_idx"]))

        with tf.io.gfile.GFile(
            args.gs_output_dir + f"/execution_results{suffix}.json", "w"
        ) as f_execution_results:
            for result in execution_results: # jsonl format
                f_execution_results.write(json.dumps(result) + "\n")
        executor.shutdown(wait=False)

        # Evaluate the solutions and write to output file
        def get_test_results(test_execution_results):
            test_execution_results = pd.DataFrame(test_execution_results)
            test_execution_results["id"] = test_execution_results.index
            
            task_success = test_execution_results.groupby("prompt_idx")["execution_result"].agg(
                lambda gen_results_for_prompt: [
                    gen_res["success"] for gen_res in gen_results_for_prompt
                ]
            )
            correct = task_success.apply(sum).to_numpy()
            total = task_success.apply(len).to_numpy()
            ks = [1, 10, 100]
            print(f"Estimating pass@k {ks=} for {len(total)} problems...")
            pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
            print(f"{pass_at_k=}")

            # calculate problem solve rate
            assert len(total) == len(correct)
            n_problem_solved = (correct > 0).sum()
            n_total = len(total)
            solve_rate = n_problem_solved / n_total
            pass_at_k["solve_rate"] = {
                "n_total": int(n_total),
                "n_correct": int(n_problem_solved),
                "solve_rate": float(solve_rate),
            }
            print(f"{pass_at_k['solve_rate']=}")

            # add task_success info to pass_at_k
            pass_at_k["task_success"] = task_success.to_dict()
            return pass_at_k

        pass_at_k = get_test_results(execution_results)

        with tf.io.gfile.GFile(
            args.gs_output_dir + f"/pass_at_k{suffix}.json", "w"
        ) as f:
            json.dump(pass_at_k, f)
        print(f"pass@k saved to {args.gs_output_dir + f'/pass_at_k{suffix}.json'}")
        return pass_at_k

    print("Waiting for solution executor to finish...")
    save_executor_result(solution_executor)

    if postprocess_fn is not None:
        print("Waiting for postprocess executor to finish...")
        save_executor_result(postprocess_executor, suffix=".postprocess")

with tf.io.gfile.GFile(args.gs_output_dir + "/generations.json", "w") as f_generations:
    f_generations.write(json.dumps(
        generations,
        cls=NumpyEncoder
    ) + "\n")

with tf.io.gfile.GFile(args.gs_output_dir + "/references.json", "w") as f_references:
    f_references.write(json.dumps(
        references,
        cls=NumpyEncoder
    ) + "\n")

with tf.io.gfile.GFile(
    args.gs_output_dir + "/generation_exec_tasks.jsonl",
    "w"
) as f_tasks:
    for task in tasks:
        f_tasks.write(json.dumps(task, cls=NumpyEncoder) + "\n")

import psutil
# Get the current process
current_process = psutil.Process()
# Get all child processes
child_processes = current_process.children(recursive=True)
# Kill all child processes
for child in child_processes:
    child.kill()
