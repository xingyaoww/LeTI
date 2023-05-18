import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from transformers.modeling_flax_utils import ACT2FN
from transformers.modeling_flax_outputs import ModelOutput, FlaxCausalLMOutput
from typing import Optional, Tuple, Union, Dict, List

from .modeling_flax_codegen import (
    FlaxCodeGenForCausalLM,
    FlaxCodeGenForCausalLMModule,
    FlaxCodeGenModule,
    FlaxCodeGenMLP
)
from .configuration_codegen_rl import CodeGenRLConfig
from ..utils import partition_nn_utils as pnn
from leti.utils.algo import (
    bce_loss_fn,
    bce_loss_weighted_fn,
    mse_loss_fn,
    reward_weighted_finetuning,
    ppo_objective,
)

class FlaxValueHead(nn.Module):
    config: CodeGenRLConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)

        self.fc_in = pnn.Dense(
            self.config.hidden_size * 2,
            dtype=self.dtype,
            kernel_init=kernel_init,
            shard_axes={
                "kernel": ("embed", "mlp"),
                "bias": ("mlp",),
            }
        )
        self.fc_out = pnn.Dense(
            1,
            dtype=self.dtype,
            kernel_init=kernel_init,
            shard_axes={
                "kernel": ("mlp", None),
                "bias": (None,),
            }
        )

        self.act = ACT2FN[self.config.activation_function]

    def __call__(self, hidden_states, deterministic: bool = True):
        # self.config.hidden_size -> 2 * self.config.hidden_size
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        # 2 * self.config.hidden_size -> 1
        hidden_states = self.fc_out(hidden_states)
        return hidden_states

@flax.struct.dataclass
class CodeGenRLOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    loss: Optional[jnp.ndarray] = None
    values: Optional[jnp.ndarray] = None
    metrics: Optional[Dict[str, jnp.ndarray]] = None

class FlaxCodeGenRLForCausalLMModule(FlaxCodeGenForCausalLMModule):
    config: CodeGenRLConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.transformer = FlaxCodeGenModule(self.config, dtype=self.dtype)
        self.lm_head = pnn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            shard_axes={
                "kernel": ("embed", "vocab"),
                "bias": ("vocab",),
            }
        )
        if self.config.is_reward_model:
            self.reward_head = pnn.Dense(
                1,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
                shard_axes={
                    "kernel": ("embed", None),
                    "bias": (None,),
                }
            )


        self.use_value_head = self.config.has_value_head and not self.config.is_reward_model
        if self.use_value_head:
            self.value_head = FlaxValueHead(
                self.config,
                dtype=self.dtype,
            )

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if not self.config.is_reward_model:
            if self.config.tie_word_embeddings:
                shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
                lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
            else:
                lm_logits = self.lm_head(hidden_states)
        else:
            lm_logits = self.reward_head(hidden_states)

        if self.use_value_head:
            values = self.value_head(hidden_states)

        if not return_dict:
            if self.use_value_head:
                return (lm_logits, values) + outputs[1:]
            else:
                return (lm_logits,) + outputs[1:]
        else:
            if self.use_value_head:
                return CodeGenRLOutput(
                    logits=lm_logits,
                    values=values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            else:
                return FlaxCausalLMOutput(
                    logits=lm_logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions
                )
            
class FlaxCodeGenRLForCausalLM(FlaxCodeGenForCausalLM):
    config_class = CodeGenRLConfig
    module_class = FlaxCodeGenRLForCausalLMModule

    def __call__(
        self,
        input_ids,
        *args,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[jnp.ndarray] = None,
        reward_mask: Optional[jnp.ndarray] = None,
        reward_val: Optional[jnp.ndarray] = None,
        advantages: Optional[jnp.ndarray] = None,
        old_log_probs_act_taken: Optional[jnp.ndarray] = None,
        returns: Optional[jnp.ndarray] = None,
        **kwargs
    ):
        outputs = super().__call__(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            params=params,
            past_key_values=past_key_values,
            dropout_rng=dropout_rng,
            train=train,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if not train:
            # evaluation - only compute language modeling loss when labels are provided
            if labels is not None:
                loss = bce_loss_fn(outputs.logits, labels)
                outputs = CodeGenRLOutput(loss=loss, **outputs)
                return outputs

        if self.config.is_reward_model:
            # Regression (train reward model)
            if reward_mask is not None and reward_val is not None:
                reward_pred = outputs.logits.squeeze(-1) # shape (batch_size, seq_len)
                loss = mse_loss_fn(reward_pred, reward_val, reward_mask)
                outputs = CodeGenRLOutput(loss=loss, **outputs)
        else:
            if self.config.rl_algorithm is None:
                outputs = reward_weighted_finetuning(
                    outputs, labels, reward_mask, reward_val,
                    output_cls=CodeGenRLOutput
                )
            elif self.config.rl_algorithm == "ppo":
                outputs = ppo_objective(
                    outputs,
                    labels,
                    reward_mask,
                    advantages,
                    old_log_probs_act_taken,
                    returns,
                    output_cls=CodeGenRLOutput,
                    config=self.config,
                )
            else:
                raise ValueError(f"Unknown RL algorithm: {self.config.rl_algorithm}")
        return outputs
