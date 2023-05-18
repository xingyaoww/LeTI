import os
import math
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from typing import Callable, Optional, Mapping, Union, Dict, Tuple, List

from leti.utils.jax import partitioning
from leti.utils.jax import train_state as t5x_train_state_lib

def create_learning_rate_fn(
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
    learning_rate: float,
    learning_rate_end: float = 0.0,
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = math.ceil(train_ds_size / train_batch_size)
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=learning_rate_end, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn

def accumulate_grads_microbatched(
    loss_fn,
    train_state: t5x_train_state_lib.TrainState,
    batch: Dict[str, np.ndarray],
    data_partition_spec,
    dropout_rng,
    loss_only: bool = False,
    num_microbatches: Optional[int] = 1,
):
    """Implements optional microbatched gradient accumulation.

    Args:
      loss_fn: The loss function that takes in (train_state.params, batch, dropout_rng).
      train_state: A train state with model parameters and optimizer state.
      batch: A batch of data.
      dropout_rng: jax PRNGKey for dropout.
      num_microbatches: the number of microbatches to use, or None for direct
        training.
      data_partition_spec: the PartitionSpec to use for partitioning annotations
        on the batch.

    Returns:
     Accumulated gradients and incremental metrics.
    """
    batch_size = batch["id"].shape[0]
    if loss_only:
        # fake grad_fn that returns None for the gradient
        grad_fn = lambda *args, **kwargs: (
            loss_fn(*args, **kwargs),
            None
        )
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    assert train_state.flax_mutables == t5x_train_state_lib.EMPTY_DICT, "Flax mutables are not supported."

    if num_microbatches is None or num_microbatches <= 1:
        (loss_accum, metrics_accum), grad_accum = grad_fn(train_state.params, batch, dropout_rng)
    else:
        assert batch_size % num_microbatches == 0, (
            f"Batch size {batch_size} isn't divided evenly by num_microbatches {num_microbatches}.")
        microbatch_size = batch_size // num_microbatches

        def get_microbatch(batch: dict, idx: int) -> Mapping[str, jnp.ndarray]:
            """Fetch microbatch slice from possibly-packed input data."""
            offset = idx * microbatch_size
            length = microbatch_size
            starts = {k: [offset] + [0] * (b.ndim - 1)
                      for k, b in batch.items()}
            limits = {k: [length] + list(b.shape[1:])
                      for k, b in batch.items()}
            return {
                k: jax.lax.dynamic_slice(b, starts[k], limits[k])
                for k, b in batch.items()
            }

        def calculate_grad(loop_cnt, dropout_rng):
            dropout_rng, sub_dropout_rng = jax.random.split(dropout_rng)
            mbatch = get_microbatch(batch, loop_cnt)
            # We need to annotate the microbatch sharding as we would a batch.
            mbatch = jax.tree_util.tree_map(
                lambda x: partitioning.with_sharding_constraint(  # pylint: disable=g-long-lambda
                    x, data_partition_spec),
                mbatch
            )
            (loss, metrics), grad = grad_fn(train_state.params, mbatch, sub_dropout_rng)
            return loss, grad, metrics

        def per_microbatch_train_step(
            loop_cnt: int, state: Tuple[jnp.ndarray, jnp.ndarray,
                                        Mapping[str, jnp.ndarray],
                                        Optional[flax.core.FrozenDict]]
        ) -> Tuple[jnp.ndarray, jnp.ndarray, Mapping[str, jnp.ndarray],
                   Optional[flax.core.FrozenDict]]:
            (dropout_rng, loss_accum, grad_accum, metrics_accum) = state
            loss, grad, metrics = calculate_grad(loop_cnt, dropout_rng)
            loss_accum = loss_accum + loss
            metrics_accum = jax.tree_util.tree_map(
                jnp.add, metrics_accum, metrics
            )
            if not loss_only: # only accumulate gradients if we are training
                grad_accum = jax.tree_util.tree_map(jnp.add, grad_accum, grad)
            return dropout_rng, loss_accum, grad_accum, metrics_accum

        # Initialize gradient accumulation loop state.
        accum_dtype = jnp.float32
        loss_accum_init = jnp.zeros((), accum_dtype)
        grad_accum_init = jax.tree_util.tree_map(
            lambda x: jnp.zeros(x.shape, accum_dtype),
            train_state.params
        )

        _, _, initial_metrics_shape = jax.eval_shape(
            calculate_grad, loop_cnt=0,
            dropout_rng=dropout_rng
        )
        metrics_accum_init = {
            k: jnp.zeros((), accum_dtype)
            for k in initial_metrics_shape
        }
        loop_init = (dropout_rng, loss_accum_init, grad_accum_init, metrics_accum_init)
        new_dropout_rng, loss_accum, grad_accum, metrics_accum = jax.lax.fori_loop(
            0, num_microbatches, per_microbatch_train_step, loop_init
        )
        del new_dropout_rng
    return loss_accum, grad_accum, metrics_accum


def pad_to_batch_size(output, num_examples, batch_size, tokenizer, ignore_keys=[]):
    if num_examples < batch_size:
        pad_len = batch_size - num_examples
        for k, v in output.items():
            if k == "id":
                assert isinstance(output[k], list)
                output[k] = output[k] + [-1] * pad_len
                continue
            
            if k in ignore_keys:
                output[k] = np.array(list(output[k]) + [None] * pad_len)
                continue

            if k == "attention_mask":
                padding_value = 0
            elif k == "non_feedback_mask":
                padding_value = 0
            # GT
            elif k == "labels":
                padding_value = -100
            elif k == "reward_mask":
                padding_value = False
            elif k == "reward_val":
                padding_value = np.nan
            # other tokens
            elif k == "input_ids":
                padding_value = tokenizer.pad_token_id
            elif k == "num_tokens":
                padding_value = 0
            else:
                raise ValueError(f"Unknown key {k} in output dict.")

            if not isinstance(v, np.ndarray):
                v = np.array(v)
            output[k] = np.concatenate(
                [
                    v,
                    np.ones(
                        (pad_len,) + v.shape[1:],
                        dtype=v.dtype,
                    ) * padding_value,
                ],
                axis=0,
            )
    return output


def loss_fn(
    params,
    batch,
    dropout_rng,
    model=None,
    train=True, 
    kl_divergence_coef=None
):
    output = model.__call__(
        **batch,
        params=params,
        dropout_rng=dropout_rng,
        train=train
    )
    if hasattr(output, "metrics") and output.metrics is not None:
        metrics = output.metrics
    else:
        metrics = {}

    loss = output.loss

    if kl_divergence_coef is not None \
        and "initial_model_logits" in batch \
        and "non_feedback_mask" in batch:
        # Compute KL divergence on non-feedback tokens
        non_feedback_mask = batch["non_feedback_mask"]
        p = jax.nn.softmax(batch["initial_model_logits"], axis=-1)
        q = jax.nn.softmax(output.logits, axis=-1)

        eps = 1e-8
        kl_divergence = p * (jnp.log(p + eps) - jnp.log(q + eps))
        kl_divergence *= non_feedback_mask[:, :, None]
        kl_divergence = jnp.sum(kl_divergence)
        
        # Average over non-padding tokens
        n_non_padding_tokens = jnp.sum(non_feedback_mask)
        kl_divergence = jax.lax.cond(
            n_non_padding_tokens > 0,
            lambda x: x / n_non_padding_tokens,
            lambda x: x,
            kl_divergence
        )

        # Log KL divergence
        metrics["kl_divergence"] = kl_divergence
        # Update loss
        loss += kl_divergence_coef * kl_divergence

    return loss, metrics
