import warnings

import jax
import jax.numpy as jnp
import flax
import numpy as np
from jax.experimental import PartitionSpec as P
from jax.experimental.compilation_cache import compilation_cache as cc

from transformers import (
    AutoTokenizer,
    GenerationConfig
)
from . import FlaxCodeGenRLForCausalLM, CodeGenRLConfig
from leti.utils.jax.checkpoints import Checkpointer
from leti.utils.jax.train_state import InferenceState
from leti.utils.jax.partitioning import PjitPartitioner

cc.initialize_cache("/tmp/jax_cache")

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ResourceWarning)

if jax.process_index() == 0:
    warnings.filterwarnings("default")


# print but only on the first node
def head_print(*args, **kwargs):
    if jax.process_index() == 0:
        print(*args, **kwargs)


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

class Inferencer:
    def __init__(
        self,
        hf_ckpt=None,
        t5x_path=None,
        num_partitions=4,
        generation_kwargs: dict = {},
        # When running training
        config: None = None,
        tokenizer: None = None,
        model: None = None,
        partitioner: None = None,
        state_axes: None = None,
    ):

        # Only required for loading from checkpoint
        self.hf_ckpt = hf_ckpt
        self.path = t5x_path

        # Config
        if config is None:
            config = CodeGenRLConfig.from_pretrained(self.hf_ckpt)
        else:
            config = config

        # Tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_ckpt)
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        assert self.tokenizer.pad_token is not None
        assert self.tokenizer.padding_side == "left"

        # Partitioner
        if partitioner is None:
            self.partitioner = PjitPartitioner(
                num_partitions=num_partitions,
                logical_axis_rules=logical_axis_rules_full
            )
        else:
            self.partitioner = partitioner

        # State axes
        if state_axes is not None:
            self.params_spec = state_axes.params

        # Model
        if model is None:
            self.model = FlaxCodeGenRLForCausalLM(config, _do_init=False, dtype=jnp.bfloat16)
            # Only consider init state for partitioning when model is not provided
            def init_state():
                rng = jax.random.PRNGKey(42)
                initial_vars = self.model.init_weights(rng, input_shape=(1, 1))
                return InferenceState.create(initial_vars)

            state_shapes = jax.eval_shape(init_state)
            self.params_spec = self.partitioner.get_mesh_axes(state_shapes).params

            # Instantiate checkpointer
            self.checkpointer = Checkpointer(
                state_shapes,
                self.partitioner,
                self.path,
                use_gda=True,
                restore_dtype=jnp.bfloat16,
                save_dtype=jnp.bfloat16
            )
        else:
            self.model = model
            assert partitioner is not None, "Partitioner must be provided when model is provided"
            assert state_axes is not None, "State axes must be provided when model is provided"

        # Generation config
        self.extra_generation_kwargs = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        self.init_fn()

    def init_fn(self):
        def infer(params, input_ids, attention_mask):
            # generate
            output = self.model(
                input_ids,
                attention_mask=attention_mask,
                params=params
            )
            return output

        self.p_infer = self.partitioner.partition(
            infer,
            in_axis_resources=(
                self.params_spec,
                self.partitioner.data_partition_spec,
                self.partitioner.data_partition_spec,
            ),
            out_axis_resources=self.partitioner.data_partition_spec
        )

        def generate(
            params,
            input_ids,
            attention_mask,
            prng_key,
            generation_config: dict
        ):
            generation_config = GenerationConfig(**generation_config)
            output_ids = self.model.generate(
                input_ids,
                generation_config=generation_config,
                attention_mask=attention_mask,
                params=params,
                prng_key=prng_key
            ).sequences
            return output_ids

        self.p_generate = self.partitioner.partition(
            generate,
            in_axis_resources=(
                self.params_spec,
                self.partitioner.data_partition_spec,
                self.partitioner.data_partition_spec,
                None,
                # ignore generation_config since it is a compile-time constant
            ),
            static_argnums=(4,),
            out_axis_resources=self.partitioner.data_partition_spec
        )

    def load_model_and_params(self):
        # load state
        assert self.path is not None, "Path must be provided when loading from checkpoint"
        self.loaded_state = self.checkpointer.restore(path=self.path)

    def generate(
        self,
        inputs,
        params=None,
        generation_rng=None,
        generation_config={},
        only_decode_generation=False
    ):
        generation_config = flax.core.freeze({
            **generation_config,
            **self.extra_generation_kwargs
        }) # make generation config hashable

        if isinstance(inputs, list):
            inputs = self.tokenizer(
                inputs,
                return_tensors="jax",
                padding=True,
                pad_to_multiple_of=8,
            )
        
        if params is None:
            params = self.loaded_state.params
        assert params is not None, "No params provided"

        if inputs["input_ids"].shape[1] > generation_config["max_length"]:
            gen_ids = inputs["input_ids"]
        else:
            # This will auto-magically run in mesh context
            gen_ids = self.p_generate(
                params,
                inputs["input_ids"],
                inputs["attention_mask"],
                generation_rng,
                generation_config
            )

        # convert jax.Array to numpy.ndarray
        # This will block jax's async dispatch! use with caution
        gen_ids = np.array(gen_ids)

        if only_decode_generation:
            input_seq_len = inputs["input_ids"].shape[1]
            gen_ids = gen_ids[:, input_seq_len:]

        generated_text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return generated_text

    def generate_fast(
        self,
        inputs,
        params=None,
        generation_rng=None,
        generation_config={}
    ):
        generation_config = flax.core.freeze({
            **generation_config,
            **self.extra_generation_kwargs
        }) # make generation config hashable

        if params is None:
            params = self.loaded_state.params
        assert params is not None, "No params provided"

        if inputs["input_ids"].shape[1] >= generation_config["max_length"]:
            gen_ids = inputs["input_ids"]
        else:
            # This will auto-magically run in mesh context
            gen_ids = self.p_generate(
                params,
                inputs["input_ids"],
                inputs["attention_mask"],
                generation_rng,
                generation_config
            )
        return gen_ids

    def infer(self, inputs, params=None):
        if isinstance(inputs, list):
            inputs = self.tokenizer(
                inputs,
                return_tensors="jax",
                padding=True,
                pad_to_multiple_of=8,
            )

        if params is None:
            params = self.loaded_state.params
        assert params is not None, "No params provided"

        # This will auto-magically run in mesh context
        outputs = self.p_infer(
            params,
            inputs["input_ids"],
            inputs["attention_mask"]
        )
        return outputs

