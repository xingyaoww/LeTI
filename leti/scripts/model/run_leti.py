#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implementation of "LeTI: Learning to Generate from Textual Interactions"
This code implements Feedback-conditioned Fine-tuning (FCFT), see paper for more details (https://arxiv.org/abs/2305.10314).

Based on code: https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_clm_flax.py
"""

import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logging.getLogger("jax").setLevel(logging.WARNING)

import re
import json
import signal
import itertools
import functools
import math
import os
import sys
import time
import wandb
import tensorflow as tf
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Optional, Mapping, Union, Dict, Tuple, List, Iterable

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import flax
import pandas as pd
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from collections import Counter

jax.distributed.initialize()

import transformers
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.testing_utils import CaptureLogger

from jax_smi import initialise_tracking
from jax.experimental.compilation_cache import compilation_cache as cc
initialise_tracking()
cc.initialize_cache("/tmp/jax_cache")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from leti.models import FlaxCodeGenRLForCausalLM, CodeGenRLConfig
from leti.utils.jax import partitioning
from leti.utils.jax import optimizers as t5x_optimizers
from leti.utils.jax import utils as t5x_utils
from leti.utils.jax import train_state as t5x_train_state_lib
from leti.utils.training import (
    accumulate_grads_microbatched,
    create_learning_rate_fn,
    pad_to_batch_size,
    loss_fn
)
from leti.utils.dataloader import dataloader
from leti.utils.execute.executor import Executor, future_tasks_iter_fn
from leti.utils.eval import estimate_pass_at_k
from leti.utils.reward.training import convert_execution_result_to_train_instance
from leti.models.jax_inferencer import Inferencer
from leti.utils.dataloader import generation_dataloader


tok_logger = transformers.utils.logging.get_logger(
    "transformers.tokenization_utils_base"
)
MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
GS_STORAGE_OPTIONS = {"project": os.environ["GCLOUD_PROJECT"]}

RW_GOOD_TOKEN = "<|good|>"
RW_BAD_TOKEN = "<|bad|>"
LANG_FEEDBACK_BEGIN_TOKEN = "<|lang_feedback|>"
LANG_FEEDBACK_END_TOKEN = "<|/lang_feedback|>"

@dataclass
class TrainingArguments:
    gs_output_dir: str = field(
        default=None,
        metadata={"help": "Google Storage Path to a T5X checkpoint to load from."},
    )
    gs_rm_ckpt_dir: str = field(
        default=None,
        metadata={"help": "Google Storage Path to a Reward Model T5X checkpoint to load from."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_test: bool = field(default=False, metadata={"help": "Whether to run test on the test set."})
    per_replica_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_replica_eval_batch_size: int = field(
        default=None, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    num_microbatches: int = field(
        default=1,
        metadata={
            "help": "Number of microbatches per batch. "
            "Each batch will be split into `num_microbatches` for training."
        },
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    learning_rate_end: float = field(default=0.0, metadata={"help": "The final learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    
    num_train_epochs: int = field(default=10, metadata={"help": "Total number of training epochs to perform per iteration."})
    num_iterations: int = field(default=10, metadata={"help": "Number of RL iterations to perform."})

    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    save_intermediate: bool = field(default=False, metadata={"help": "Save intermediate data."})
    save_steps: int = field(default=None, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})

    # def __post_init__(self):
    #     if self.output_dir is not None:
    #         self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    actor_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    num_partitions: int = field(
        default=1,
        metadata={
            "help": (
                "Number of partitions to split the model into."
            )
        },
    )
    model_parallel_submesh: str = field(
        # looks like "2,2,1,1"
        default=None,
        metadata={
            "help": (
                "Submesh to use for model parallelism."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    # For RL-style training
    n_generations_per_input: int = field(
        default=256,
        metadata={"help": "Number of generations to make for each input sequence."},
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether to use sampling or not."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling."},
    )
    top_k: int = field(
        default=0,
        metadata={"help": "Top-k for sampling."},
    )
    top_p: float = field(
        default=1,
        metadata={"help": "Top-p for sampling."},
    )
    max_gen_length: int = field(
        default=512,
        metadata={"help": "Maximum length of the generated sequence."},
    )
    pad_to_multiple_of: Optional[int] = field(
        default=128,
        metadata={"help": "Pad the generated sequence to a multiple of this value."},
    )
    filter_exec_results: bool = field(
        default=False,
        metadata={"help": "Whether to filter out execution results."},
    )
    coarse_grained_feedback_only: bool = field(
        default=False,
        metadata={"help": "Whether to only give coarse-grained feedback."},
    )
    postprocess_generation: str = field(
        default=None,
        metadata={"help": "Whether to postprocess the generated sequence using heuristic."},
    )
    postprocess_lang_feedback: bool = field(
        default=False,
        metadata={"help": "Whether to provide the post-process language feedback using heuristic."},
    )
    execute_original_code: bool = field(
        default=False,
        metadata={"help": "Whether to execute the original code."},
    )
    execution_timeout: int = field(
        default=5,
        metadata={"help": "Timeout (in seconds) for execution."},
    )
    execution_extra_headers: str = field(
        default="",
        metadata={"help": "Extra headers for execution."},
    )
    mix_pretrain_data: bool = field(
        default=False,
        metadata={"help": "Whether to mix pretraining data."},
    )
    only_pretrain_data: bool = field(
        default=False,
        metadata={"help": "Whether to only use pretraining data."},
    )
    kl_divergence: float = field(
        default=None,
        metadata={"help": "KL divergence for finetuning."},
    )
    pretrain_data_gs_path: str = field(
        default=None,
        metadata={"help": "Path to the pretraining data."},
    )
    pretrain_valid_split: float = field(
        default=0.1,
        metadata={"help": "Validation split for pretraining data."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def get_intermediate_dir(output_dir, epoch, prefix):
    return os.path.join(
        output_dir,
        "intermediate",
        "epoch_{}".format(epoch),
        prefix,
    )

def print_dataset_length_distribution(dataset, prefix=""):
    counter = Counter()
    for input_ids in dataset["input_ids"]:
        counter[len(input_ids)] += 1
    logger.info(f"[{prefix}] input_ids length distribution:")
    for k, v in sorted(counter.items()):
        logger.info("- {}: {}".format(k, v))

def mbpp_postprocess_generation(gen: str) -> str:
    STOP_WORDS = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/"]
    def code_first_block(string):
        """Split off first block of code by scanning for class, def etc. on newlines."""
        return re.split("|".join(STOP_WORDS), string)[0].rstrip()
    return code_first_block(gen)

def eae_postprocess_generation(gen: str) -> str:
    STOP_WORDS = ['"""', 'class', 'print', '#']
    gen = re.split('|'.join(STOP_WORDS), gen)[0]
    return gen

def generate_and_execute(
    dataset: Dataset,
    actor_inferencer: Inferencer,
    params: flax.core.FrozenDict,
    batch_size: int,
    n_generations: int,
    epoch: int,
    cur_step: int,
    prefix="",
    data_args: DataTrainingArguments = None,
    training_args: TrainingArguments = None,
    generation_config: dict = None,
    partitioner: partitioning.PjitPartitioner = None,
) -> List[dict]:
    intermediate_dir = get_intermediate_dir(training_args.gs_output_dir, epoch, prefix)
    if tf.io.gfile.exists(os.path.join(intermediate_dir, "execution_results.json")):
        logger.info("Intermediate directory already exists, skipping generation and execution.")
        with tf.io.gfile.GFile(os.path.join(intermediate_dir, "execution_results.json"), "r") as f:
            execution_results = []
            for line in f:
                execution_results.append(json.loads(line))
        return execution_results

    logger.info(
        "Intializing solution executor with timeout {}s and extra headers {}...".format(
        data_args.execution_timeout, data_args.execution_extra_headers)
    )
    solution_executor = Executor(
        process_id=jax.process_index(),
        timeout=data_args.execution_timeout,
        extra_headers=data_args.execution_extra_headers,
    )

    try:
        # ======================== Generating ==============================
        # Generate data_args.n_generations_per_input generations for each input

        if tf.io.gfile.exists(os.path.join(intermediate_dir, "generation_exec_tasks.jsonl")):
            logger.info("Generation + Execution tasks detected in intermediate directory, skipping generation - executing only.")
            with tf.io.gfile.GFile(os.path.join(
                intermediate_dir, "generation_exec_tasks.jsonl"), "r") as f:
                tasks = []
                # Load the generation tasks & execution them
                for line in tqdm(f, desc="Sending solutions for execution..."):
                    cur_task = json.loads(line)
                    tasks.append(cur_task)
                    solution_executor.add_task(cur_task)
            
        else:
            # Do the generation
            generation_start_time = time.time()
            generations = []
            references = []
            tasks = []
            infer_pbar = tqdm(
                total=len(dataset) * n_generations,
                desc=f"Generating solutions for [{prefix}]"
            )
            dataloader = generation_dataloader(
                dataset,
                n_samples=n_generations,
                batch_size=batch_size,
                tokenizer=actor_inferencer.tokenizer,
                return_jnp_array=True,
            )
            logger.info(f"Generation config: {generation_config}")
            n_skipped_generations_due_to_max_length = 0
            rng = jax.random.PRNGKey(42)
            
            for batch in dataloader:
                # update rng for each batch
                rng, generation_rng = jax.random.split(rng)
                cur_generation_ids: jax.Array = actor_inferencer.generate_fast(
                    batch,
                    params,
                    generation_rng=generation_rng,
                    generation_config=generation_config
                )

                if jax.process_count() > 1:
                    # We need to do an all-gather operation to get the current batch
                    # NOTE: we can't delay this in a thread because it may cause deadlock
                    # i.e., [allgather, next generate_fast, ...] vs. [next generate_fast, allgather, ...]
                    cur_generation_ids = jax.experimental.multihost_utils.process_allgather(
                        cur_generation_ids
                    )
                
                def task_callback_fn(cur_task: dict):
                    if data_args.execute_original_code:
                        cur_task["postprocessed_generation"] = cur_task["generation"]
                        cur_task["generation"] = cur_task["original_generation"]

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
                
                def mbpp_postprocess_langfeedback_fn(
                    original_generation: str,
                    postprocessed_generation: str,
                ) -> str:
                    # postprocessed_generation is a prefix of original_generation
                    # we need to add the rest of the original generation
                    assert original_generation.startswith(postprocessed_generation)
                    # find the rest of the original generation
                    rest_of_original_generation = original_generation[
                        len(postprocessed_generation):
                    ].strip()[:128]
                    if rest_of_original_generation:
                        return f"We only need the first defined function. You should not generate the rest of the code that starts with [{rest_of_original_generation}]. Try to use <|endoftext|> to end your code."
                    else:
                        return "" # no additional feedback

                if data_args.postprocess_generation == "eae":
                    postprocess_generation_fn = eae_postprocess_generation
                    postprocess_langfeedback_fn = None
                elif data_args.postprocess_generation == "mbpp":
                    postprocess_generation_fn = mbpp_postprocess_generation
                    postprocess_langfeedback_fn = mbpp_postprocess_langfeedback_fn
                else:
                    postprocess_generation_fn = None
                    postprocess_langfeedback_fn = None

                # Run the future tasks in a separate thread
                solution_executor.add_future_tasks(
                    functools.partial(
                        future_tasks_iter_fn,
                        cur_generation_ids,
                        batch,
                        actor_inferencer.tokenizer,
                        task_callback_fn=task_callback_fn,
                        postprocess_fn=postprocess_generation_fn,
                        prompt_processing_fn=lambda x: x.lstrip(RW_GOOD_TOKEN).lstrip(RW_BAD_TOKEN),

                        # This will add "additional_feedback" to the task (i.e., execution result)
                        postprocess_lang_feedback_fn=postprocess_langfeedback_fn \
                            if data_args.postprocess_lang_feedback else None,
                    )
                )
                infer_pbar.update(batch_size)
            infer_pbar.close()
            
            if jax.process_index() == 0 and training_args.save_intermediate:
                with tf.io.gfile.GFile(
                    os.path.join(intermediate_dir, "generations.json"), "w"
                ) as f_generations:
                    f_generations.write(json.dumps(generations) + "\n")
                with tf.io.gfile.GFile(
                    os.path.join(intermediate_dir, "references.json"), "w"
                ) as f_references:
                    f_references.write(json.dumps(references) + "\n")
                with tf.io.gfile.GFile(
                    os.path.join(intermediate_dir, "generation_exec_tasks.jsonl"), "w"
                ) as f_tasks:
                    for task in tasks:
                        f_tasks.write(json.dumps(task) + "\n")
                logger.info(
                    "Saved generation result to intermediate directory {}.".format(
                    intermediate_dir)
                )
            logger.info("Finished generating solutions.")
            wandb.log(
                {f"{prefix}/generation_time": time.time() - generation_start_time},
                step=cur_step
            )
            # n_skipped_generations_due_to_max_length
            wandb.log(
                {f"{prefix}/n_skipped_generations_due_to_max_length": n_skipped_generations_due_to_max_length},
                step=cur_step
            )
            
        # ======================== Execution ================================
        # Execute the generated solutions
        execution_start_time = time.time()
        logger.info("Waiting for solutions to be executed...")
        # NOTE: each process will execute its own solutions (repeatly) - need further optimization
        execution_results: List[Dict] = solution_executor.get_results_and_reset()
        if jax.process_index() == 0 and training_args.save_intermediate:
            with tf.io.gfile.GFile(
                os.path.join(intermediate_dir, "execution_results.json"), "w"
            ) as f_execution_results:
                for result in execution_results: # jsonl format
                    f_execution_results.write(json.dumps(result) + "\n")
        wandb.log({
            f"{prefix}/execution_time": time.time() - execution_start_time},
            step=cur_step
        )
        logger.info("Finished executing solutions.")
        solution_executor.shutdown(wait=False)
    except (KeyboardInterrupt, Exception):
        logger.info(f"Shutting down solution executor... (process {jax.process_index()})")
        solution_executor.shutdown(wait=False)
        raise
    return execution_results

def load_dataset_from_disk_if_available(gs_path: str):
    logger.info(f"Loading dataset from {gs_path}")
    try:
        dataset = load_from_disk(
            gs_path,
            storage_options=GS_STORAGE_OPTIONS
        )
        logger.info(f"Loaded {len(dataset)} examples from {gs_path}")
    except FileNotFoundError as e:
        logger.info(f"Dataset not found at {gs_path}: {e}")
        return None
    return dataset

def build_rw_conditioned_train_dataset(
    train_execution_results,
    train_batch_size,
    tokenizer,
    training_args,
    data_args,
    epoch: int,
    wait_to_load_dataset: bool,
):
    intermediate_dir = get_intermediate_dir(training_args.gs_output_dir, epoch, "train")
    dataset_gs_path = os.path.join(intermediate_dir, "rw_cond_train_dataset.arrow")

    if wait_to_load_dataset:
        _n_tries = 0
        train_dataset = None
        while train_dataset is None:
            if _n_tries % 5 == 0:
                logger.info(f"[n_tries={_n_tries}] Waiting for dataset to be available at {dataset_gs_path}")
            train_dataset = load_dataset_from_disk_if_available(dataset_gs_path)
            time.sleep(5)
            _n_tries += 1
        assert train_dataset is not None
    else:
        train_dataset = load_dataset_from_disk_if_available(dataset_gs_path)
    
    if train_dataset is not None:
        return train_dataset

    def train_tokenize_function(examples, batch_size=train_batch_size):

        with CaptureLogger(tok_logger) as cl:
            # text_a is the feedback
            text_a = examples["text_a"]
            # text_b is the problem + solution
            text_b = examples["text_b"]
            # execution successfulness
            success = examples["success"]

            # add feedback begin/end tokens if feedback is not None
            if data_args.coarse_grained_feedback_only:
                # Don't add feedback tokens if we're using coarse-grained feedback
                text_a = [
                    "" for txt in text_a
                ]
            else:
                text_a = [
                    f"{LANG_FEEDBACK_BEGIN_TOKEN}{txt}{LANG_FEEDBACK_END_TOKEN}"
                    if txt is not None else "" # if feedback is None, give empty string
                    for txt in text_a
                ]

            # # concatenate text_a and text_b, prepend GOOD/BAD token by checking success
            # if data_args.postprocess_generation:
            #     # add <eos>
            #     text = [
            #         f"{RW_GOOD_TOKEN if s else RW_BAD_TOKEN}{txt_a}{txt_b}{tokenizer.eos_token}"
            #         for txt_a, txt_b, s in zip(text_a, text_b, success)
            #     ]
            #     prefix_char_length = [
            #         len(f"{RW_GOOD_TOKEN if s else RW_BAD_TOKEN}{txt_a}")
            #         for txt_a, s in zip(text_a, success)
            #     ]
            # else:
            text = [
                f"{RW_GOOD_TOKEN if s else RW_BAD_TOKEN}{txt_a}{txt_b}"
                for txt_a, txt_b, s in zip(text_a, text_b, success)
            ]
            prefix_char_length = [
                len(f"{RW_GOOD_TOKEN if s else RW_BAD_TOKEN}{txt_a}")
                for txt_a, s in zip(text_a, success)
            ]

            # tokenize text
            output = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="np",
                pad_to_multiple_of=128, # larger value to avoid repetitive compilation
                return_offsets_mapping=True
            )
            # convert this BatchEncoding to dict
            output = dict(output)

            # create a non_feedback_mask to mask out the feedback & paadding part
            # of the input_ids
            output["non_feedback_mask"] = np.zeros_like(output["input_ids"])
            for i, (input_id, offset_mapping) in enumerate(zip(output["input_ids"], output["offset_mapping"])):
                # offset_mapping is a list of tuples (start, end)
                # start/end are the character indices of the token
                # we want to mask out the feedback part of the input_ids
                # i.e., 0 for padding & non-feedback tokens, 1 for feedback tokens

                cur_prefix_char_length = prefix_char_length[i]
                for j, (start, end) in enumerate(offset_mapping):
                    # if the token is not a special token (e.g., pad)
                    if start != 0 or end != 0:
                        # if the token is not part of the feedback
                        if start >= cur_prefix_char_length:
                            output["non_feedback_mask"][i, j] = 1
        # drop offset_mapping
        output.pop("offset_mapping")
        output["id"] = examples["id"]
        output["labels"] = output["input_ids"].copy()
        # set labels to -100 for pad_token
        output["labels"][
            np.where(output["attention_mask"] == 0)
        ] = -100
        num_examples = len(output["id"])

        # if last batch (len < global_batch_size), pad examples to global_batch_size
        output = pad_to_batch_size(
            output,
            num_examples=num_examples,
            batch_size=batch_size,
            tokenizer=tokenizer,
        )

        output["num_examples"] = np.array(
            [num_examples] * batch_size,
            dtype=np.int32,
        )
        output["num_tokens"] = np.array(
            np.sum(output["attention_mask"], axis=1),
            dtype=np.int32,
        )
        return output

    train_df = pd.DataFrame([
        # nested list comprehension
        # flatten the list
        item
        for row in train_execution_results
        for item in convert_execution_result_to_train_instance(row)
    ])

    train_df["id"] = train_df.index
    train_dataset = datasets.Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(
        train_tokenize_function,
        batched=True,
        batch_size=train_batch_size,
        remove_columns=train_dataset.column_names,
        num_proc=data_args.preprocessing_num_workers,
        desc="Tokenizing train dataset based on execution results",
    )
    print_dataset_length_distribution(train_dataset, "Reward-conditioned Train Dataset")

    # save reward model inference results
    if jax.process_index() == 0 and training_args.save_intermediate:
        logger.info(f"Saving reward-confitioned train dataset to {dataset_gs_path}")
        train_dataset.save_to_disk(
            dataset_gs_path,
            storage_options=GS_STORAGE_OPTIONS
        )
    return train_dataset

def log_execution_result_stats(train_execution_results, cur_step):
    # - num_results
    wandb.log({
        "train/execution/num_results": len(train_execution_results)
    }, step=cur_step)
    # - sucess rate
    success_rate = sum([
        1 for result in train_execution_results
        if result["execution_result"]["success"]]
    ) / len(train_execution_results)
    wandb.log({
        "train/execution/success_rate": success_rate
    }, step=cur_step)
    # - reason
    reason_counter = Counter([
        result["execution_result"]["reason"]
        for result in train_execution_results
    ])
    for reason, count in reason_counter.items():
        wandb.log({
            f"train/execution/reason/{reason}": count
        }, step=cur_step)
    # - exception type
    exception_type_counter = Counter([
        result["execution_result"]["traceback"]["exc_type"]
        for result in train_execution_results
        if result["execution_result"]["reason"] == "exception"
    ])
    for exception_type, count in exception_type_counter.items():
        wandb.log({
            f"train/execution/exception_type/{exception_type}": count
        }, step=cur_step)

def get_test_results(test_execution_results, training_args, epoch):
    if jax.process_index() > 0:
        return {}

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
    logger.info(f"Estimating pass@k {ks=} for {len(total)} problems...")
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
    logger.info(f"{pass_at_k=}")

    if jax.process_index() == 0 and training_args.save_intermediate:
        with tf.io.gfile.GFile(
            os.path.join(get_intermediate_dir(
            training_args.gs_output_dir, epoch, "test"),
            f"pass_at_k.json"
            ), "w") as f:
            json.dump(pass_at_k, f)
    return pass_at_k

def main():
    # See all possible arguments in leti/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Wandb
    if jax.process_index() != 0:
        # disable wandb on non-master processes
        os.environ["WANDB_MODE"] = "disabled"
    wandb.init(
        project="lang-reward",
        group=os.environ.get("WANDB_TASK"),
        resume="allow", # It will look for WANDB_RUN_ID in the environment
        config={
            "exp_id": os.environ.get("EXP_ID", None),
            "instance_name": os.environ.get("INSTANCE_NAME", None),
            **vars(model_args),
            **vars(data_args),
            **vars(training_args),
        }
    )

    # Make one log on every process with the configuration for debugging.
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)


    #  Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    train_dataset_for_inference = datasets.load_dataset("json", data_files=data_args.train_file)["train"]

    if data_args.max_train_samples is not None:
        logger.info(f"Current train dataset size: {len(train_dataset_for_inference)}")
        train_dataset_for_inference = train_dataset_for_inference.select(range(data_args.max_train_samples))
        logger.info(f"Truncate train dataset to {data_args.max_train_samples} samples (usually for debugging)")

    # for validation loss
    eval_dataset = datasets.load_dataset("json", data_files=data_args.validation_file)["train"]

    # to calculate pass@1 for mbpp
    if training_args.do_test:
        test_dataset_for_inference = datasets.load_dataset("json", data_files=data_args.test_file)["train"]

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    assert model_args.actor_model_name_or_path
    config = CodeGenRLConfig.from_pretrained(
        model_args.actor_model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.actor_model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # NOTE: set padding_side to left (since GPT2 is left-padded)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting `pad_token` to `eos_token`:{} for open-end generation.".format(tokenizer.eos_token))

    # actor model
    model: FlaxCodeGenRLForCausalLM = FlaxCodeGenRLForCausalLM(
        config,
        seed=training_args.seed,
        dtype=getattr(jnp, model_args.dtype),
        _do_init=False,
    )

    # Store some constant
    num_epochs = int(training_args.num_train_epochs) * int(training_args.num_iterations)
    n_replica = (jax.device_count() // model_args.num_partitions)
    train_batch_size = int(training_args.per_replica_batch_size) * n_replica
    if training_args.per_replica_eval_batch_size is not None:
        eval_batch_size = int(training_args.per_replica_eval_batch_size) * n_replica
    else:
        eval_batch_size = train_batch_size

    def add_rw_token(examples, add_rw_token=True):
        if add_rw_token:
            examples["text"] = RW_GOOD_TOKEN + examples["text"]
        return examples

    train_dataset_for_inference_wo_rw_token = train_dataset_for_inference.map(
        functools.partial(add_rw_token, add_rw_token=False),
        batched=False,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    train_dataset_for_inference = train_dataset_for_inference.map(
        functools.partial(add_rw_token, add_rw_token=True),
        batched=False,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if training_args.do_test:
        test_dataset_for_inference_wo_rw_token = test_dataset_for_inference.map(
            functools.partial(add_rw_token, add_rw_token=False),
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        test_dataset_for_inference = test_dataset_for_inference.map(
            functools.partial(add_rw_token, add_rw_token=True),
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    def tokenize_function(
        examples,
        batch_size,
        add_eos=True,
        add_rw_token=True,
    ):
        with CaptureLogger(tok_logger) as cl:
            text = examples["text"]
            # add bos_token to beginning of each text
            if add_eos:
                text = [f"{tokenizer.bos_token}{txt}{tokenizer.eos_token}" for txt in text]
            else:
                text = [f"{tokenizer.bos_token}{txt}" for txt in text]

            if add_rw_token:
                text = [RW_GOOD_TOKEN + txt for txt in text]

            output = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="np",
                pad_to_multiple_of=128, # larger value to avoid repetitive compilation
            )

        output["id"] = examples["id"]
        num_examples = len(output["id"])

        output["labels"] = output["input_ids"].copy()
        # set labels to -100 for pad_token
        output["labels"][
            np.where(output["attention_mask"] == 0)
        ] = -100

        # When "reward_val" is predicted for training the actor model
        if "reward_mask" in examples:
            assert "reward_val" in examples
            assert not add_eos
            # pad reward_mask and reward_val to the same as output["input_ids"] and shift by 1 (bos_token)
            reward_mask = np.zeros(output["input_ids"].shape)
            reward_val = np.ones(output["input_ids"].shape) * np.nan
            for i, (mask, val) in enumerate(zip(examples["reward_mask"], examples["reward_val"])):
                reward_mask[i, 1:1 + len(mask)] = mask
                reward_val[i, 1:1 + len(val)] = val
            output["reward_mask"] = reward_mask.astype(np.bool_)
            output["reward_val"] = reward_val.astype(np.float32)
        
        # if last batch (len < train_batch_size), pad examples to train_batch_size
        output = pad_to_batch_size(
            output,
            num_examples=num_examples,
            batch_size=batch_size,
            tokenizer=tokenizer,
        )

        output["num_examples"] = np.array(
            [num_examples] * batch_size,
            dtype=np.int32,
        )
        output["num_tokens"] = np.array(
            np.sum(output["attention_mask"], axis=1),
            dtype=np.int32,
        )
        return output

    eval_dataset_wo_rw_token = eval_dataset.map(
        functools.partial(
            tokenize_function,
            add_rw_token=False,
            batch_size=eval_batch_size
        ),
        batched=True,
        batch_size=eval_batch_size,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    eval_dataset = eval_dataset.map(
        functools.partial(
            tokenize_function,
            add_rw_token=True,
            batch_size=eval_batch_size
        ),
        batched=True,
        batch_size=eval_batch_size,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)

    # i.e., number of train steps per epoch (supposely)
    unit_steps = len(train_dataset_for_inference) * data_args.n_generations_per_input
    steps_per_epoch = unit_steps
    if data_args.mix_pretrain_data:
        steps_per_epoch += unit_steps # 1 ex paired with 1 ex of pretrain data
    steps_per_epoch = math.ceil(steps_per_epoch / train_batch_size)
    total_train_steps = steps_per_epoch * num_epochs

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        steps_per_epoch,
        train_batch_size,
        num_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
        training_args.learning_rate_end,
    )
    if training_args.learning_rate == training_args.learning_rate_end:
        logger.info(f"Using constant learning rate of {training_args.learning_rate}")

    assert not (data_args.mix_pretrain_data and data_args.only_pretrain_data), \
        "mix_pretrain_data and only_pretrain_data cannot be True at the same time"

    if data_args.only_pretrain_data:
        data_args.mix_pretrain_data = True
        logger.info(f"Only pretraining model!")

    if data_args.mix_pretrain_data:
        intermediate_dir = get_intermediate_dir(training_args.gs_output_dir, 0, "init")
        pretrain_dataset_train_gs_path = os.path.join(intermediate_dir, "pretrain_dataset_train.arrow")
        pretrain_dataset_train = load_dataset_from_disk_if_available(pretrain_dataset_train_gs_path)
        pretrain_dataset_valid_gs_path = os.path.join(intermediate_dir, "pretrain_dataset_valid.arrow")
        pretrain_dataset_valid = load_dataset_from_disk_if_available(pretrain_dataset_valid_gs_path)

        if pretrain_dataset_train is None or pretrain_dataset_valid is None:
            logger.info(f"Pretraining dataset not found, creating new one...")

            # CLM pretraining pipeline based on
            # https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_clm_flax.py

            raw_pretrain_dataset = load_from_disk(
                data_args.pretrain_data_gs_path,
                storage_options=GS_STORAGE_OPTIONS
            )

            # train/valid split
            raw_pretrain_dataset = raw_pretrain_dataset.train_test_split(
                test_size=data_args.pretrain_valid_split,
                seed=training_args.seed,
            )

            def pretrain_tokenize_function(examples):
                    with CaptureLogger(tok_logger) as cl:
                        output = tokenizer(examples["content"])
                    # clm input could be much much longer than block_size
                    if "Token indices sequence length is longer than the" in cl.out:
                        tok_logger.warning(
                            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                            " before being passed to the model."
                        )
                    return output

            tokenized_pretrain_dataset = raw_pretrain_dataset.map(
                pretrain_tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_pretrain_dataset["train"].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

            if data_args.block_size is None:
                block_size = tokenizer.model_max_length
                if block_size > config.max_position_embeddings:
                    logger.warning(
                        f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                        "Picking 1024 instead. You can change that default value by passing --block_size xxx."
                    )
                    block_size = 1024
            else:
                if data_args.block_size > tokenizer.model_max_length:
                    logger.warning(
                        f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                        f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                    )
                block_size = min(data_args.block_size, tokenizer.model_max_length)
            
            logger.info(f"Use block_size={block_size} for pretrain dataset")
            
            # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                if total_length >= block_size:
                    total_length = (total_length // block_size) * block_size
                # Split by chunks of max_len.
                result = {
                    k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
            # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
            # to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

            pretrain_dataset = tokenized_pretrain_dataset.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

            def add_metadata(examples, batch_size):
                examples["num_tokens"] = np.array(
                    np.sum(examples["attention_mask"], axis=1),
                    dtype=np.int32,
                )

                # if last batch (len < train_batch_size), pad examples to train_batch_size
                examples = pad_to_batch_size(
                    examples,
                    num_examples=len(examples["id"]),
                    batch_size=batch_size,
                    tokenizer=tokenizer,
                )
                return examples

            # select the number of examples to be a multiple of the batch size 
            pretrain_dataset_train = pretrain_dataset["train"]
            pretrain_dataset_valid = pretrain_dataset["test"]
            logger.info(f"Pretrain dataset (before truncating to multiples of 128) train size: {len(pretrain_dataset_train)}; valid size: {len(pretrain_dataset_valid)}")
            pretrain_dataset_train = pretrain_dataset_train.select(range(
                len(pretrain_dataset_train) // 128 * 128))
            pretrain_dataset_valid = pretrain_dataset_valid.select(range(
                len(pretrain_dataset_valid) // 128 * 128))
            logger.info(f"Pretrain dataset (after truncation) train size: {len(pretrain_dataset_train)}; valid size: {len(pretrain_dataset_valid)}")

            # Add relevant columns.
            pretrain_dataset_train = pretrain_dataset_train.add_column("id", list(range(len(pretrain_dataset_train))))
            pretrain_dataset_train = pretrain_dataset_train.map(
                functools.partial(add_metadata, batch_size=train_batch_size),
                batched=True,
                batch_size=train_batch_size,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )
            pretrain_dataset_valid = pretrain_dataset_valid.add_column("id", list(range(len(pretrain_dataset_valid))))
            pretrain_dataset_valid = pretrain_dataset_valid.map(
                functools.partial(add_metadata, batch_size=eval_batch_size),
                batched=True,
                batch_size=eval_batch_size,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )
            # if jax.process_index() == 0 and training_args.save_intermediate:
            #     logger.info(f"Saving pre-trained train/val dataset to {pretrain_dataset_train_gs_path} and {pretrain_dataset_valid_gs_path}")
            #     pretrain_dataset_train.save_to_disk(
            #         pretrain_dataset_train_gs_path,
            #         storage_options=GS_STORAGE_OPTIONS
            #     )
            #     pretrain_dataset_valid.save_to_disk(
            #         pretrain_dataset_valid_gs_path,
            #         storage_options=GS_STORAGE_OPTIONS
            #     )

        print_dataset_length_distribution(pretrain_dataset_train, "Pretrain dataset (train)")
        print_dataset_length_distribution(pretrain_dataset_valid, "Pretrain dataset (valid)")

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = flax.traverse_util.flatten_dict(params)
        # find out all LayerNorm parameters
        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
        layer_norm_named_params = set(
            [
                layer[-2:]
                for layer_norm_name in layer_norm_candidates
                for layer in flat_params.keys()
                if layer_norm_name in "".join(layer).lower()
            ]
        )
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
        return flax.traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    optimizer = t5x_optimizers.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        mask=decay_mask_fn,
    )

    # Initialize partitioner and train state
    # https://github.com/google-research/t5x/blob/main/docs/usage/partitioning.md#data-parallel-with-parameter-gather
    # 2D parameter and activation partitioning
    logical_axis_rules_full = [
        ('batch', 'data'),
        ('mlp', 'model'),
        ('heads', 'model'),
        ('vocab', 'model'),
        # shard both activations and weight matrices on the remaining available axis
        ('embed', 'model'),
        ('embed', 'data'),
        ('kv', None),
        ('joined_kv', None),
        ('relpos_buckets', None),
        ('abspos_buckets', None),
        ('length', None),
        ('layers', None),
        ('stack', None),
        ('mlp_activations', None),
    ]
    if model_args.model_parallel_submesh is not None:
        # model_parallel_submesh looks like "2,2,1,1"
        model_args.model_parallel_submesh = tuple(map(
            int, model_args.model_parallel_submesh.split(",")
        ))
        logger.info(f"model_parallel_submesh: {model_args.model_parallel_submesh}")
        model_args.model_parallel_submesh = tuple(model_args.model_parallel_submesh)
        assert np.prod(model_args.model_parallel_submesh) == model_args.num_partitions, \
            f"model_parallel_submesh {model_args.model_parallel_submesh} does not match num_partitions {model_args.num_partitions}"

        # unset num_partitions since we are using model_parallel_submesh
        model_args.num_partitions = None

    partitioner = partitioning.PjitPartitioner(
        num_partitions=model_args.num_partitions,
        model_parallel_submesh=model_args.model_parallel_submesh,
        logical_axis_rules=logical_axis_rules_full,
    )
    
    def _init_variables(rng: jax.random.KeyArray, *unused_args, **unused_kwargs) -> flax.core.scope.FrozenVariableDict:
        # Note we ignore input_shapes and input_types here (they are only used for t5x models)
        initial_vars = model.init_weights(rng, input_shape=(1, 1))
        return initial_vars

    train_state_initializer = t5x_utils.TrainStateInitializer(
        optimizer_def=optimizer,
        partitioner=partitioner,
        # Ignore input shapes and types (t5x specific)
        init_fn=_init_variables,
        input_shapes=None,
        input_types=None,
    )

    # Load train state from checkpoint
    checkpoint_cfg = t5x_utils.CheckpointConfig(
        save=t5x_utils.SaveCheckpointConfig(
            dtype="bfloat16",
            period=training_args.save_steps,
        ),
        restore=t5x_utils.RestoreCheckpointConfig(
            path=training_args.gs_output_dir,
            mode="latest",
            dtype="bfloat16"
        ),
    )
    valid_restore_cfg, restore_paths = t5x_utils.get_first_valid_restore_config_and_paths(
        [checkpoint_cfg.restore]
    )

    checkpoint_manager = t5x_utils.LegacyCheckpointManager(
        save_cfg=checkpoint_cfg.save,
        restore_cfg=valid_restore_cfg,
        train_state_shape=train_state_initializer.global_train_state_shape,
        partitioner=partitioner,
        ds_iter=None, # do not save ds_iter
        model_dir=training_args.gs_output_dir,
    )

    rng, fallback_init_rng = jax.random.split(rng)
    train_state: t5x_train_state_lib.TrainState = checkpoint_manager.restore(
        restore_paths,
        restore_cfg=valid_restore_cfg,
        fallback_state=train_state_initializer.from_scratch(fallback_init_rng).state_dict()
    )

    if data_args.kl_divergence is not None:
        initial_actor_inferencer = Inferencer(
            hf_ckpt=model_args.actor_model_name_or_path,
            t5x_path=os.path.join(training_args.gs_output_dir, "checkpoint_0"),
            config=config,
            tokenizer=tokenizer,
            partitioner=partitioner,
            state_axes=train_state_initializer.train_state_axes
        )
        initial_actor_inferencer.load_model_and_params()
    assert train_state is not None, "Failed to restore train state"

    # Define gradient update step fn
    rng, dropout_rng = jax.random.split(rng)

    if data_args.kl_divergence is not None:
        loss_fn_w_model = functools.partial(
            loss_fn,
            model=model,
            kl_divergence_coef=data_args.kl_divergence,
        )
    else:
        loss_fn_w_model = functools.partial(loss_fn, model=model)
    
    def train_step(
        state: t5x_train_state_lib.TrainState,
        batch,
        dropout_rng=dropout_rng
    ):
        loss, grad, metrics = accumulate_grads_microbatched(
            loss_fn_w_model,
            train_state=state,
            batch=batch,
            data_partition_spec=partitioner.data_partition_spec,
            dropout_rng=dropout_rng,
            num_microbatches=training_args.num_microbatches,
        )

        # normalize loss, gradient, and metrics by number of tokens
        num_tokens = batch["num_tokens"].sum()
        # only normalize if num_tokens > 0
        def _normalize(loss, grad, metrics, num_tokens):
            loss = loss / num_tokens
            grad = jax.tree_map(lambda x: x / num_tokens, grad)
            metrics = jax.tree_map(lambda x: x / num_tokens, metrics)
            return loss, grad, metrics

        loss, grad, metrics = jax.lax.cond(
            num_tokens > 0,
            lambda x: _normalize(*x),
            lambda x: x[:-1], # noop, return loss, grad, metrics
            (loss, grad, metrics, num_tokens)
        )

        new_state = state.apply_gradient(
            grads=grad,
            # learning rate is already set when initializing the optimizer
            learning_rate=None
        )
        cur_lr = linear_decay_lr_schedule_fn(state.step)
        metrics = {
            "loss": loss,
            "learning_rate": cur_lr,
            "num_tokens": num_tokens,
            **metrics
        }
        return new_state, metrics

    def eval_step(state: t5x_train_state_lib.TrainState, batch):
        loss, _, metrics = accumulate_grads_microbatched(
            functools.partial(loss_fn_w_model, train=False),
            train_state=state,
            batch=batch,
            data_partition_spec=partitioner.data_partition_spec,
            dropout_rng=dropout_rng,
            num_microbatches=training_args.num_microbatches,
            loss_only=True
        )

        # summarize metrics
        metrics = {
            "lm_loss": loss,
            "num_tokens": batch["num_tokens"].sum(),
            **metrics
        }
        return metrics

    # Create pjit version of the train and eval step
    p_train_step = partitioner.partition(
        train_step,
        in_axis_resources=(
            train_state_initializer.train_state_axes,
            partitioner.data_partition_spec
        ),
        out_axis_resources=(train_state_initializer.train_state_axes, None),
        donate_argnums=(0,)
    )
    p_eval_step = partitioner.partition(
        eval_step,
        in_axis_resources=(
            train_state_initializer.train_state_axes,
            partitioner.data_partition_spec
        ),
        out_axis_resources=None,
    )

    def evaluate_dataset(
        p_eval_step,
        train_state,
        dataset,
        eval_batch_size, 
        name=""
    ):
        eval_loader = dataloader(
            dataset, eval_batch_size,
            return_jnp_array=True
        )
        eval_steps = math.ceil(len(dataset) / eval_batch_size)

        eval_metrics = None

        with jax.spmd_mode("allow_all"):
            for _ in tqdm(range(eval_steps), desc=f"Evaluating {name}...", position=2, leave=False):
                # Model forward
                batch = next(eval_loader)
                
                metrics = p_eval_step(train_state, batch)
                # NOTE: the lm_loss is the sum of the loss of each token over the batch
                if eval_metrics is None:
                    eval_metrics = metrics
                else:
                    eval_metrics = jax.tree_map(lambda x, y: x + y, eval_metrics, metrics)

            # normalize eval metrics
            eval_metrics["lm_loss"] = eval_metrics["lm_loss"] / eval_metrics["num_tokens"]
            eval_metrics["lm_loss"] = eval_metrics["lm_loss"].item()

            try:
                eval_metrics["perplexity"] = math.exp(eval_metrics["lm_loss"])
            except OverflowError:
                eval_metrics["perplexity"] = float("inf")
        return eval_metrics
      
    def eval_epoch(
        p_eval_step,
        train_state,
        eval_dataset,
        eval_dataset_wo_rw_token,
        eval_batch_size
    ):

        # ======================== Evaluating ==============================
        eval_metrics = evaluate_dataset(
            p_eval_step, train_state,
            eval_dataset, eval_batch_size, name="w_rw_token"
        )
        eval_metrics_wo_rw_token = evaluate_dataset(
            p_eval_step, train_state,
            eval_dataset_wo_rw_token, eval_batch_size, name="wo_rw_token"
        )

        return {
            "w_rw_token/lm_loss": eval_metrics["lm_loss"],
            "w_rw_token/perplexity": eval_metrics["perplexity"],
            "wo_rw_token/lm_loss": eval_metrics_wo_rw_token["lm_loss"],
            "wo_rw_token/perplexity": eval_metrics_wo_rw_token["perplexity"],
        }

    actor_inferencer = Inferencer(
        config=config,
        tokenizer=tokenizer,
        model=model,
        partitioner=partitioner,
        state_axes=train_state_initializer.train_state_axes
    )

    train_generation_config = {
        "do_sample": data_args.do_sample,
        "temperature": data_args.temperature,
        "top_p": data_args.top_p,
        "top_k": data_args.top_k,
        "num_beam": 1,
        "max_length": data_args.max_gen_length
    }
    test_generation_config = {
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 0,
        "num_beam": 1,
        "max_length": data_args.max_gen_length
    }

    # no need to load model and params, since we already passed them to the generator

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset_for_inference)}")
    logger.info(f"  Num generations per training example = {data_args.n_generations_per_input}")
    logger.info(f"  Num examples per epoch = {steps_per_epoch * train_batch_size}")
    logger.info(f"  Num steps per epoch = {steps_per_epoch}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Num Epochs per iteration = {training_args.num_train_epochs}")
    logger.info(f"  Num iterations = {training_args.num_iterations}")
    logger.info(f"  Num of devices = {jax.device_count()}")
    logger.info(f"  Num of replicas = {n_replica}")
    logger.info(f"  Num of partitions = {model_args.num_partitions}")
    logger.info(f"  Num of microbatches = {training_args.num_microbatches}")
    logger.info(f"  Instantaneous batch size per replica = {training_args.per_replica_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    # check cur epoch number from ckpt
    cur_step_from_ckpt = train_state._optimizer.state.step.item()
    cur_epoch_from_ckpt = cur_step_from_ckpt // steps_per_epoch
    assert cur_epoch_from_ckpt * steps_per_epoch == cur_step_from_ckpt, \
        f"cur_step_from_ckpt {cur_step_from_ckpt} is not a multiple of steps_per_epoch {steps_per_epoch}, and does not match cur_epoch_from_ckpt {cur_epoch_from_ckpt}"
    logger.info(f"Start training from epoch {cur_epoch_from_ckpt}")
    
    pbar = tqdm(range(num_epochs), desc="Epoch ...  ")
    cur_epoch = cur_epoch_from_ckpt
    # set pbar to the current epoch
    for _ in range(cur_epoch):
        pbar.update(1)
    logger.info(f"Start training from epoch {cur_epoch_from_ckpt}")

    if training_args.eval_steps is None:
        # default to evaluate every epoch
        training_args.eval_steps = steps_per_epoch
    if training_args.save_steps is None:
        # default to save every RL iteration
        training_args.save_steps = steps_per_epoch * training_args.num_train_epochs


    # ======================== Evaluation before Training Start ==============================
    if cur_epoch == 0:
        eval_metrics = eval_epoch(
            p_eval_step,
            train_state,
            eval_dataset,
            eval_dataset_wo_rw_token,
            eval_batch_size
        )

        if data_args.mix_pretrain_data:
            pretrain_eval_metrics = evaluate_dataset(
                p_eval_step,
                train_state,
                dataset=pretrain_dataset_valid,
                eval_batch_size=eval_batch_size,
                name="pretrain"
            )

            eval_metrics = {
                **eval_metrics,
                **{
                    f"pretrain/{k}": v
                    for k, v in pretrain_eval_metrics.items()
                }
            }
        # Print metrics and update progress bar
        desc = f"Step... ({0} | {eval_metrics})"
        pbar.write(desc)
        pbar.desc = desc

        # test before training
        if training_args.do_test:
            test_execution_results = generate_and_execute(
                test_dataset_for_inference_wo_rw_token, # This is evaluation before training starts (no rw token)
                actor_inferencer,
                train_state.params,
                batch_size=eval_batch_size,
                n_generations=16, # hard-coded for mbpp test
                epoch=0,
                cur_step=0,
                prefix="test",
                data_args=data_args,
                training_args=training_args,
                generation_config=test_generation_config,
                partitioner=partitioner,
            )
            pass_at_k = get_test_results(test_execution_results, training_args, epoch=0)
        else:
            pass_at_k = {}

        # Save metrics
        wandb.log(
            {
                **{"eval/" + k: v for k, v in eval_metrics.items()},
                **{"test/" + k: v for k, v in pass_at_k.items()}
            },
            step=0
        )

    while cur_epoch < num_epochs:
        # e.g., when num_iterations = 2, num_train_epochs = 3, 
        # cur_epoch = 5 -> iteration_start_epoch = 3, inner_train_epoch = 2
        inner_train_epoch = cur_epoch % int(training_args.num_train_epochs)
        iteration_start_epoch = cur_epoch - inner_train_epoch
        assert iteration_start_epoch % int(training_args.num_train_epochs) == 0
        # both will be initialized to 0 when start from scratch
    
        train_start = time.time()
        if not data_args.only_pretrain_data:
            # ======================== Generate & Execution for Training ================================
            train_execution_results = generate_and_execute(
                # use train_dataset_for_inference_wo_rw_token for the first epoch
                # since the actor has not been trained to understand the rw token yet
                train_dataset_for_inference if iteration_start_epoch != 0 else train_dataset_for_inference_wo_rw_token,
                actor_inferencer,
                train_state.params,
                batch_size=eval_batch_size,
                n_generations=data_args.n_generations_per_input,
                epoch=iteration_start_epoch,
                cur_step=iteration_start_epoch * steps_per_epoch,
                prefix="train",
                data_args=data_args,
                training_args=training_args,
                generation_config=train_generation_config,
                partitioner=partitioner,
            )

            # Filter train_execution_results to remove result with undesired reasons
            if data_args.filter_exec_results:
                logger.info(f"Filtering train execution results with undesired reasons: Before filtering: {len(train_execution_results)}")
                wandb.log({
                    "train/num_train_execution_results_unfiltered": len(train_execution_results)
                }, step=iteration_start_epoch * steps_per_epoch)
                train_execution_results = [
                    result for result in train_execution_results
                    if result["execution_result"]["reason"] not in {
                        "execution driver exception",
                        "process timeout"
                    }
                ]
                logger.info(f"After filtering: {len(train_execution_results)}")

            if jax.process_index() == 0:
                log_execution_result_stats(train_execution_results, iteration_start_epoch * steps_per_epoch)

            # ======================== Reward(and Language)-Conditioned Dataset ================================
            train_dataset = build_rw_conditioned_train_dataset(
                train_execution_results,
                train_batch_size=train_batch_size,
                tokenizer=tokenizer,
                training_args=training_args,
                data_args=data_args,
                epoch=iteration_start_epoch,
                # the process 0 will save the dataset to disk
                # and the other processes will load the dataset from disk
                wait_to_load_dataset=jax.process_index() > 0,
            )

        # ======================== Training ================================


        while inner_train_epoch < int(training_args.num_train_epochs):

            # Generate an epoch by shuffling sampling indices from the train dataset
            if not data_args.only_pretrain_data:
                train_loader = dataloader(
                    train_dataset, train_batch_size,
                    return_jnp_array=True,
                )
            
            if data_args.mix_pretrain_data: # mix pretrain data
                pretrain_loader = dataloader(
                    pretrain_dataset_train, train_batch_size,
                    return_jnp_array=True,
                )
            
            # train
            training_pbar = tqdm(
                total=steps_per_epoch,
                desc=f"Epoch {iteration_start_epoch + inner_train_epoch}",
            )
            step = 0
            while step < steps_per_epoch:
                if not data_args.only_pretrain_data:
                    batch = next(train_loader)
                    if data_args.kl_divergence is not None:
                        # add logits to batch for later kl-divergence calculation
                        # break the batch into smaller batches to avoid OOM
                        initial_model_logits_lst = []
                        mini_batch_size = train_batch_size // training_args.num_microbatches
                        for i in range(0, train_batch_size, mini_batch_size):
                            cur_infer = initial_actor_inferencer.infer({
                                    "input_ids": batch["input_ids"][i:i+mini_batch_size],
                                    "attention_mask": batch["attention_mask"][i:i+mini_batch_size],
                            }).logits
                            initial_model_logits_lst.append(
                                np.array(cur_infer)
                            )
                        initial_model_logits = np.concatenate(initial_model_logits_lst, axis=0)
                        batch["initial_model_logits"] = initial_model_logits
                    train_state, train_metric = p_train_step(train_state, batch)

                    step += 1
                    training_pbar.update(1)
                else:
                    # if we only want to pretrain the actor (for ablation study)
                    train_metric = {}

                # one more optimization step for the actor
                if data_args.mix_pretrain_data: # mix pretrain data
                    try:
                        batch = next(pretrain_loader)
                    except StopIteration:
                        pretrain_loader = dataloader(
                            pretrain_dataset_train, train_batch_size,
                            return_jnp_array=True,
                        )
                        batch = next(pretrain_loader)
                    train_state, pretrain_metric = p_train_step(train_state, batch)

                    step += 1
                    training_pbar.update(1)
                else:
                    pretrain_metric = {}

                cur_step = (iteration_start_epoch + inner_train_epoch) * steps_per_epoch + step

                if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                    # Save metrics
                    wandb.log(
                        {
                            "train/time": (time.time() - train_start) // 60,
                            **{
                                f"train/{k}": v
                                for k, v in train_metric.items()
                            },
                            **{
                                f"train/pretrain/{k}": v
                                for k, v in pretrain_metric.items()
                            }
                        },
                        step=cur_step
                    )

                    pbar.write(
                        f"Step... ({cur_step} | metrics: {train_metric} | pretrain metrics: {pretrain_metric}"
                    )

                if cur_step % training_args.eval_steps == 0 and cur_step > 0:
                    eval_metrics = eval_epoch(
                        p_eval_step,
                        train_state,
                        eval_dataset,
                        eval_dataset_wo_rw_token,
                        eval_batch_size
                    )

                    if data_args.mix_pretrain_data:
                        pretrain_eval_metrics = evaluate_dataset(
                            p_eval_step,
                            train_state,
                            dataset=pretrain_dataset_valid,
                            eval_batch_size=eval_batch_size,
                        )

                        eval_metrics = {
                            **eval_metrics,
                            **{
                                f"pretrain/{k}": v
                                for k, v in pretrain_eval_metrics.items()
                            }
                        }

                    # Print metrics and update progress bar
                    desc = f"Step... ({cur_step} | {eval_metrics})"
                    pbar.write(desc)
                    pbar.desc = desc

                    wandb.log(
                        {"eval/" + k: v for k, v in eval_metrics.items()},
                        step=cur_step
                    )
            
            # increment inner epoch
            training_pbar.close()
            inner_train_epoch += 1
            pbar.update(1)

        assert inner_train_epoch == int(training_args.num_train_epochs)
        assert (iteration_start_epoch + inner_train_epoch) % int(training_args.num_train_epochs) == 0
        checkpoint_manager.save(train_state)
        
        # Update epoch
        cur_epoch = iteration_start_epoch + inner_train_epoch

        # ======================== Test set Evaluation per epoch ================================
        if training_args.do_test:
            test_execution_results = generate_and_execute(
                # since the actor has already been trained to understand the rw token
                # if only_pretrain_data is True, we don't need to add the rw token to the test dataset
                test_dataset_for_inference if not data_args.only_pretrain_data else test_dataset_for_inference_wo_rw_token,
                actor_inferencer,
                train_state.params,
                batch_size=eval_batch_size,
                n_generations=16, # hard-coded for mbpp test
                epoch=cur_epoch,
                cur_step=cur_step,
                prefix="test",
                data_args=data_args,
                training_args=training_args,
                generation_config=test_generation_config,
                partitioner=partitioner,
            )
            pass_at_k = get_test_results(test_execution_results, training_args, cur_epoch)
        else:
            pass_at_k = {}
        
        wandb.log(
            {"test/" + k: v for k, v in pass_at_k.items()},
            step=cur_step
        )

if __name__ == "__main__":
    main()
    import psutil
    # Get the current process
    current_process = psutil.Process()
    # Get all child processes
    child_processes = current_process.children(recursive=True)
    # Kill all child processes
    for child in child_processes:
        child.kill()
