import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn
from flax.training.common_utils import onehot


def bce_loss_fn(logits, labels):
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    loss = optax.softmax_cross_entropy(
        shift_logits, onehot(shift_labels, shift_logits.shape[-1])
    )
    # we ignore the loss on the <pad> tokens (marked with -100)
    mask = jnp.where(shift_labels >= 0, 1, 0)#.astype(jnp.bool_)
    return jnp.sum(loss, where=mask) # scalar

def bce_loss_weighted_fn(logits, labels, weights, mask=None):
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    assert weights.shape == labels.shape, f"{weights.shape} != {labels.shape}"
    # e.g., sequence: [1, 2, 3, 4]
    #       weights:  [0.1, 0.2, 0.3, 0.4[]]
    #       labels:   [2, 3, 4]
    #       logits -  try to predict labels
    #       weights for (loss of) logits prediction should be: [0.2, 0.3, 0.4] (i.e., weights[..., 1:])
    weights = weights[..., 1:] # same as shift_labels
    assert weights.shape == shift_labels.shape, f"{weights.shape} != {shift_labels.shape}"
    loss = optax.softmax_cross_entropy(
        shift_logits, onehot(shift_labels, shift_logits.shape[-1])
    )
    
    if mask is None:
        # ignore the loss with weight == 0
        mask = jnp.where(weights == 0, 0, 1).astype(jnp.bool_)
    loss = loss * weights
    
    return jnp.sum(loss, where=mask)

def mse_loss_fn(predicted_val, target_val, mask):
    # mask is a boolean mask of shape (batch_size, sequence_length)
    assert predicted_val.shape == target_val.shape
    assert predicted_val.shape == mask.shape
    # replace nan with 0 so that we don't take the loss of masked tokens into account
    target_val = jnp.nan_to_num(target_val, nan=0.0)
    
    # ignore the loss on the masked tokens
    loss = jnp.square(predicted_val - target_val)
    mask = mask.astype(jnp.bool_)
    
    return jnp.sum(loss, where=mask)
