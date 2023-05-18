import jax.numpy as jnp
from . import bce_loss_fn, bce_loss_weighted_fn

def reward_weighted_finetuning(
    outputs, labels, reward_mask, reward_val, output_cls
):
    if reward_mask is not None and reward_val is not None:
        # Weighted finetuning
        reward_val = jnp.nan_to_num(reward_val, nan=0.0)
        # loss_weight == 0 means the token is masked
        loss_weight = reward_mask * reward_val
        # clip the loss weight between -1 and 1
        loss_weight = jnp.clip(loss_weight, -1, 1)
        loss = bce_loss_weighted_fn(
            outputs.logits, labels, loss_weight
        )
        outputs = output_cls(loss=loss, **outputs)
    elif labels is not None:
        # Finetuning (equally weighted)
        loss = bce_loss_fn(outputs.logits, labels)
        outputs = output_cls(loss=loss, **outputs)
    else:
        pass # inference only
    return outputs
