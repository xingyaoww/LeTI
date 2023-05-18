"""
Initializes a Flax model from a HuggingFace model, then save its weights in the T5x format.
"""
import os
import jax
import jax.numpy as jnp
import argparse

from typing import Optional, Tuple, Union
from transformers import CodeGenConfig, CodeGenTokenizer
from leti.models import FlaxCodeGenRLForCausalLM, CodeGenRLConfig
from leti.utils.jax.convert_hf import convert_hf_weight_to_t5x_ckpt, hf_from_pretrained

jax.config.update('jax_platform_name', 'cpu') # avoid OOM for large models

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Salesforce/codegen-350M-mono")
parser.add_argument("--model-type", type=str, default="reward")
parser.add_argument("--output-dir")
parser.add_argument("--gs-output-dir")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-partitions", type=int, default=4)
args = parser.parse_args()

assert args.model_type in ["reward", "actor", "rw_conditioned",]
args.output_dir = os.path.join(args.output_dir, args.model.split("/")[-1])
print(f"Saving model to {args.output_dir}")


def custom_process_state_fn(state):
    """Custom process state function for loading pretrained checkpoints."""
    # NOTE: this is different compared to .from_pretrained() in huggingface/transformers
    # add (params, transformer) prefix to all keys
    # state = {("params", "transformer", *k): v for k, v in state.items()}
    new_state = {}
    for k, v in state.items():
        new_k = ("params", "transformer", *k)
        
        # check if "ln_f", "kernel" or "ln_1", "kernel" is in the key
        if ("ln_f" in k or "ln_1" in k) and "kernel" in k:
            # replace "kernel" with "scale"
            assert new_k[-1] == "kernel"
            new_k = new_k[:-1] + ("scale",)
        elif "wte" in k and "kernel" in k:
            assert new_k[-1] == "kernel"
            new_k = new_k[:-1] + ("embedding",)
            v = v.T # special case for wte kernel
        elif "lm_head" in k:
            assert new_k[1] == "transformer", f"unexpected key: {k}"
            # remove the "transformer" prefix for lm_head
            new_k = ("params", *new_k[2:])
            import pdb
        
        new_state[new_k] = v
    return new_state

tokenizer: CodeGenTokenizer = CodeGenTokenizer.from_pretrained(args.model)

if args.model_type == "reward":
    # Modify the pre-trained config into CodeGenRLConfig
    pretrained_config = CodeGenConfig.from_pretrained(args.model)
    pretrained_config.bos_token_id = tokenizer.bos_token_id # correct the bug in the original configs
    assert pretrained_config.bos_token_id == pretrained_config.eos_token_id
    config = CodeGenRLConfig(
        is_reward_model=True, **pretrained_config.to_dict()
    )

    # Load pretrained CodeGen model into a different class
    model = hf_from_pretrained(
        FlaxCodeGenRLForCausalLM,
        args.model,
        custom_process_state_fn=custom_process_state_fn,
        from_pt=True,
        dtype=jnp.float32,
        seed=args.seed
    )

elif args.model_type == "rw_conditioned":
    pretrained_config = CodeGenConfig.from_pretrained(args.model)
    pretrained_config.bos_token_id = tokenizer.bos_token_id # correct the bug in the original configs

    # Add custom special tokens for reward-conditioned and/or language-feedback conditioned generation
    additional_special_tokens = [
        # coarser-grained reward tokens
        "<|bad|>", "<|good|>",
        # begin/end of language feedback
        "<|lang_feedback|>", "<|/lang_feedback|>",
    ]

    # Add the special tokens to the tokenizer
    print(f"Tokenizer (vocab_size {len(tokenizer)}): Adding {len(additional_special_tokens)} special tokens")
    tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
    # pretrained_config.vocab_size = len(tokenizer) # NOTE: this is not needed
    assert pretrained_config.vocab_size >= len(tokenizer), f"vocab_size {pretrained_config.vocab_size} < {len(tokenizer)}"
    print(f"Tokenizer vocab_size (after added special tokens): {len(tokenizer)}")
    
    config = CodeGenRLConfig(
        is_reward_model=False,
        **pretrained_config.to_dict()
    )
    # Load pretrained CodeGen model into a different class
    model = hf_from_pretrained(
        FlaxCodeGenRLForCausalLM,
        args.model,
        custom_process_state_fn=custom_process_state_fn,
        from_pt=True,
        dtype=jnp.float32,
        seed=args.seed
    )

elif args.model_type == "actor":
    # Modify the pre-trained config into CodeGenRLConfig
    pretrained_config = CodeGenConfig.from_pretrained(args.model)
    pretrained_config.bos_token_id = tokenizer.bos_token_id # correct the bug in the original configs
    assert pretrained_config.bos_token_id == pretrained_config.eos_token_id
    config = CodeGenRLConfig(
        is_reward_model=False, **pretrained_config.to_dict()
    )

    # Load pretrained CodeGen model into a different class
    model = hf_from_pretrained(
        FlaxCodeGenRLForCausalLM,
        args.model,
        custom_process_state_fn=custom_process_state_fn,
        from_pt=True,
        dtype=jnp.float32,
        seed=args.seed
    )

tokenizer.save_pretrained(args.output_dir)
config.save_pretrained(args.output_dir)
print("Tokenizer and Config are saved to", args.output_dir)

convert_hf_weight_to_t5x_ckpt(
    model,
    model.params["params"],
    gs_output_dir=args.gs_output_dir,
    num_partitions=args.num_partitions,
)
