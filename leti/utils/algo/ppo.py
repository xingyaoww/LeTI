import jax
import jax.numpy as jnp
import numpy as np
import flax

# Reference:
# https://github.com/google/flax/blob/main/examples/ppo/ppo_lib.py


def gae_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    discount: float,
    gae_param: float
):
    """Use Generalized Advantage Estimation (GAE) to compute advantages.
    As defined by eqs. (11-12) in PPO paper arXiv: 1707.06347. Implementation uses
    key observation that A_{t} = delta_t + gamma*lambda*A_{t+1}.
    Args:
      rewards: array shaped (actor_steps, num_agents), rewards from the game
      terminal_masks: array shaped (actor_steps, num_agents), zeros for terminal
                      and ones for non-terminal states
      values: array shaped (actor_steps, num_agents), values estimated by critic
      discount: RL discount usually denoted with gamma
      gae_param: GAE parameter usually denoted with lambda
    Returns:
      advantages: calculated advantages shaped (actor_steps, num_agents)
    """
    assert rewards.shape[0] + 1 == values.shape[0], ('One more value needed; Eq. '
                                                     '(12) in PPO paper requires '
                                                     'V(s_{t+1}) for delta_t')
    advantages = []
    gae = 0.
    assert values[-1] == 0, "Last state value S_{t+1} should be 0"

    for t in reversed(range(len(rewards))):
        # Masks used to set next state value to 0 for terminal states.
        # value_diff = discount * values[t + 1] * terminal_masks[t] - values[t]

        # we don't need to multiply by terminal_masks[t] because we set values[t+1] to 0
        value_diff = discount * values[t + 1] - values[t]
        delta = rewards[t] + value_diff

        # # Masks[t] used to ensure that values before and after a terminal state
        # # are independent of each other.
        # gae = delta + discount * gae_param * terminal_masks[t] * gae

        # we don't need to multiply by terminal_masks[t] because gae is initialized to 0
        # AND we expect to only have one terminal state at the end of the episode (not in the middle)
        gae = delta + discount * gae_param * gae

        advantages.append(gae)

    advantages = advantages[::-1]
    return np.array(advantages)

def gae_advantages_jax(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    discount: float,
    gae_param: float
):
    """Use Generalized Advantage Estimation (GAE) to compute advantages.
    As defined by eqs. (11-12) in PPO paper arXiv: 1707.06347. Implementation uses
    key observation that A_{t} = delta_t + gamma*lambda*A_{t+1}.
    Args:
      rewards: array shaped (actor_steps, num_agents), rewards from the game
      terminal_masks: array shaped (actor_steps, num_agents), zeros for terminal
                      and ones for non-terminal states
      values: array shaped (actor_steps, num_agents), values estimated by critic
      discount: RL discount usually denoted with gamma
      gae_param: GAE parameter usually denoted with lambda
    Returns:
      advantages: calculated advantages shaped (actor_steps, num_agents)
    """
    assert rewards.shape[0] + 1 == values.shape[0], ('One more value needed; Eq. '
                                                     '(12) in PPO paper requires '
                                                     'V(s_{t+1}) for delta_t')

    def loop_body_fn(inp):
        t, gae, advantages, values = inp

        value_diff = discount * values[t + 1] - values[t]
        delta = rewards[t] + value_diff
        gae = delta + discount * gae_param * gae
        advantages = advantages.at[t].set(gae)

        return (t - 1, gae, advantages, values)

    # gae = 0.
    # advantages = jnp.zeros_like(rewards)
    # carry = (gae, advantages)
    # _, advantages = jax.lax.fori_loop(0, len(rewards), loop_body_fn, carry)
    # advantages = advantages[::-1]

    # do it with jax while loop
    t = len(rewards) - 1
    gae = 0.
    advantages = jnp.zeros_like(rewards)
    init_val = (t, gae, advantages, values)
    _, _, advantages, _ = jax.lax.while_loop(
        lambda carry: carry[0] >= 0, # iterate from t to 0
        loop_body_fn,
        init_val
    )
    return advantages
    

def ppo_objective(
    outputs,
    labels: np.ndarray,
    reward_mask: np.ndarray,
    advantages: np.ndarray,
    old_log_probs_act_taken: np.ndarray,
    returns: np.ndarray,
    output_cls,
    config
):
    """Computes PPO objective.

    Args:
      outputs: Output instance from model forward pass.
      labels: array shaped (batch_size, seq_len), labels for non-masked tokens
      reward_mask: array shaped (batch_size, seq_len), zeros for masked and ones for non-masked tokens
      advantages: array shaped (batch_size, seq_len), advantages for non-masked tokens
      old_log_probs_act_taken: array shaped (batch_size, seq_len), logits from previous iteration (old policy)
      returns: array shaped (batch_size, seq_len), returns for non-masked tokens
      output_cls: Output class.

    Returns:
      Output class with loss and other metrics.
    """
    # NOTE: reward_val is not needed, since it is already incorporated into the advantages
    if labels is None or reward_mask is None or \
        advantages is None or old_log_probs_act_taken is None or \
        returns is None:
        # inference - no loss is computed
        return outputs

    # configs
    clip_param = config.ppo_clip # PPO clipping parameter
    vf_coeff = config.ppo_vf_coef # Value function coefficient
    entropy_coeff = config.ppo_entropy_coef # Entropy coefficient

    # ==== left-shift labels everything (autoregressive) ====
    # since the last logits does not have action sampled from it
    actions = labels[..., 1:]  # shape (batch_size, seq_len - 1)
    reward_mask = reward_mask[..., 1:].astype(
        jnp.bool_)  # shape (batch_size, seq_len - 1)
    advantages = advantages[..., 1:]  # shape (batch_size, seq_len - 1)
    # old_log_probs = old_logits[:, :-1, :] # shape (batch_size, seq_len - 1, vocab_size), \pi_{old}
    returns = returns[..., 1:]  # shape (batch_size, seq_len - 1)

    # get inferred values and log_probs from current policy \pi
    # shape (batch_size, seq_len - 1)
    values = outputs.values.squeeze(-1)[:, :-1]
    # shape (batch_size, seq_len - 1, vocab_size)
    log_probs = outputs.logits[:, :-1, :]
    probs = jnp.exp(log_probs)  # shape (batch_size, seq_len - 1, vocab_size)

    # ==== compute losses ====
    # NOTE: we use reward_mask to mask out the losses for padded tokens (and non-rewarded tokens)
    value_loss = jnp.sum(
        jnp.square(returns - values),  # * reward_mask,
        where=reward_mask
    )  # scalar
    entropy = jnp.sum(
        jnp.sum(-probs * log_probs, axis=2),  # * reward_mask,
        where=reward_mask
    )  # scalar

    # log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)
    log_probs_act_taken = jnp.take_along_axis(log_probs, actions[..., None], axis=2).squeeze(-1) \
        * reward_mask  # shape (batch_size, seq_len - 1)
    old_log_probs_act_taken = old_log_probs_act_taken * \
        reward_mask  # shape (batch_size, seq_len - 1)

    # # old_log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(old_log_probs, actions)
    # old_log_probs_act_taken = jnp.take_along_axis(old_log_probs, actions[..., None], axis=2).squeeze(-1) \
    #   * reward_mask # shape (batch_size, seq_len - 1)
    assert old_log_probs_act_taken.shape == log_probs_act_taken.shape
    # shape (batch_size, seq_len - 1)
    ratios = jnp.exp(log_probs_act_taken - old_log_probs_act_taken)

    _pg_loss = ratios * advantages  # shape (batch_size, seq_len - 1)
    _clipped_loss = advantages * jax.lax.clamp(
        1. - clip_param, ratios, 1. + clip_param
    )  # shape (batch_size, seq_len - 1)
    ppo_loss = - jnp.sum(
        jnp.minimum(_pg_loss, _clipped_loss),
        where=reward_mask
    )  # scalar

    # ==== Calculate total loss and metrics ====
    total_loss = ppo_loss + vf_coeff * value_loss - entropy_coeff * entropy

    metrics = {
        # arrays of shape (,)
        "ppo_loss": ppo_loss,
        "value_loss": value_loss,
        "entropy": entropy,
    }
    outputs = output_cls(
        loss=total_loss,
        metrics=metrics,
        **outputs
    )
    return outputs
