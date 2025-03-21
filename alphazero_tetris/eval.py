from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from flax.nnx import TrainState
from tetris_gymnasium.functional.core import EnvConfig
from tetris_gymnasium.functional.tetrominoes import Tetrominoes

from alphazero_tetris.base import PolicyFn, RootFn, RolloutData, EvalFn, RootFnOutput
from alphazero_tetris.config import Config
from alphazero_tetris.mcts import MctsTree
from alphazero_tetris.visualize import save_graph_in_memory, create_rgb_frames_array

def log_metrics_and_video(rollouts: RolloutData, step):
    # Log metrics first (unchanged)
    first_termination = jnp.argmax(rollouts.terminated)
    episode_length = jnp.where(first_termination == 0, rollouts.terminated.shape[0], first_termination + 1)
    episode_score = jnp.sum(rollouts.reward)

    episode_lines_cleared = rollouts.lines_cleared.sum()
    metrics = {
        "score/mean": float(episode_score),
        "score/min": float(episode_score),
        "score/max": float(episode_score),
        "steps/mean": float(episode_length),
        "steps/min": float(episode_length),
        "steps/max": float(episode_length),
        "lines_cleared/mean": float(episode_lines_cleared),
        "lines_cleared/min": float(episode_lines_cleared),
        "lines_cleared/max": float(episode_lines_cleared),
    }
    metrics = {f"eval/{k}": v for k, v in metrics.items()}
    wandb.log(metrics, step=step)

    # Create video
    sample_all_obs = rollouts.observation[:episode_length]  # [T, H, W]
    frames_array = create_rgb_frames_array(sample_all_obs, cell_size=30)
    frames_array = np.array(frames_array)

    # Log video
    wandb.log({
        "episode_video": wandb.Video(
            frames_array,
            fps=1,
            format="mp4"
        )
    }, step=step)


@partial(jax.jit, static_argnames=('step_fn', 'reset_fn', 'recurrent_fn', 'policy_fn', 'tree_policy_fn', 'eval_fn', 'config'))
def collect_eval_play_data(
        rng_key,
        config,
        step_fn,
        reset_fn,
        recurrent_fn,
        policy_fn: PolicyFn,
        tree_policy_fn: Callable,
        eval_fn: EvalFn,
        train_state,
        tetrominoes
):
    def play_step(carry):
        rng_key, step, step_count, rollouts, _ = carry
        state, obs, reward, terminated, info = step
        rng_key, rng_key_root, rng_key_policy = jax.random.split(rng_key, 3)

        # Set root
        state_batched = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)
        obs_batched = jnp.expand_dims(obs, axis=0)
        eval_root = eval_fn(rng_key_root, train_state, state_batched, obs_batched)
        root = RootFnOutput(state=state, value=eval_root.value, variance=eval_root.variance, observation=obs, prior_logits=eval_root.p)
        # Get policy output
        policy_output = policy_fn(train_state, rng_key_policy, root, recurrent_fn, tree_policy_fn, config.num_simulations, config, 0.0)

        # Step environment
        new_step = step_fn(tetrominoes, state, policy_output.action)

        # Update rollout information
        new_rollouts = RolloutData(
            # Step statistics
            reward=rollouts.reward.at[step_count].set(reward),
            observation=rollouts.observation.at[step_count].set(obs),
            terminated=rollouts.terminated.at[step_count].set(terminated),
            lines_cleared=rollouts.lines_cleared.at[step_count].set(info['lines_cleared']),
            # Policy statistics
            value=rollouts.value.at[step_count].set(policy_output.tree.node_values[MctsTree.ROOT_INDEX]),
            variance=rollouts.variance.at[step_count].set(policy_output.tree.node_variances[MctsTree.ROOT_INDEX]),
            children_values=rollouts.children_values.at[step_count].set(policy_output.tree.children_values[MctsTree.ROOT_INDEX]),
            children_variances=rollouts.children_variances.at[step_count].set(policy_output.tree.children_variances[MctsTree.ROOT_INDEX]),
            children_rewards=rollouts.children_rewards.at[step_count].set(policy_output.tree.children_rewards[MctsTree.ROOT_INDEX]),
            children_discounts=rollouts.children_discounts.at[step_count].set(policy_output.tree.children_discounts[MctsTree.ROOT_INDEX]),
            children_visits=rollouts.children_visits.at[step_count].set(policy_output.tree.children_visits[MctsTree.ROOT_INDEX]),
            # Todo: info output for score
            score=rollouts.score.at[step_count].set(policy_output.tree.node_state.score[MctsTree.ROOT_INDEX]),
        )

        return rng_key, new_step, step_count + 1, new_rollouts, ~terminated

    def cond_fn(carry):
        _, _, step_count, _, continuing = carry
        return (step_count < config.wandb_eval_max_steps) & continuing

    # Initial setup
    rng_key, rng_key_reset = jax.random.split(rng_key)
    _, state, obs = reset_fn(tetrominoes, rng_key_reset)

    # Initialize ExamineFrame
    num_timesteps = config.wandb_eval_max_steps + 1  # root inclusive
    initial_rollouts = RolloutData(
        observation=jnp.zeros((num_timesteps,) + obs.shape, dtype=obs.dtype),
        value=jnp.zeros((num_timesteps,), dtype=jnp.float32),
        variance=jnp.zeros((num_timesteps,), dtype=jnp.float32),
        children_values=jnp.zeros((num_timesteps, config.num_actions), dtype=jnp.float32),
        children_variances=jnp.zeros((num_timesteps, config.num_actions), dtype=jnp.float32),
        children_rewards=jnp.zeros((num_timesteps, config.num_actions), dtype=jnp.float32),
        children_discounts=jnp.zeros((num_timesteps, config.num_actions), dtype=jnp.float32),
        children_visits=jnp.zeros((num_timesteps, config.num_actions), dtype=jnp.float32),
        reward=jnp.zeros((num_timesteps,), dtype=jnp.float32),
        score=jnp.zeros((num_timesteps,), dtype=jnp.float32),
        lines_cleared=jnp.zeros((num_timesteps,), dtype=jnp.float32),
        terminated=jnp.zeros((num_timesteps,), dtype=jnp.int32)
    )

    # Initial step
    init_step = (state, obs, jnp.zeros((), dtype=jnp.float32), jnp.zeros((), dtype=jnp.bool), {'lines_cleared': jnp.zeros((), dtype=jnp.int32)})
    initial_carry = (rng_key, init_step, 0, initial_rollouts, True)
    _, _, _, final_frame, _ = jax.lax.while_loop(
        cond_fn,
        play_step,
        initial_carry
    )

    return final_frame

def log_policy_loss_metrics(total_losses, value_losses, var_losses, step):
    """Log average policy losses across steps and batches."""

    # Calculate means across non-zero values (zero values are padding of early terminations)
    def compute_mean(x):
        mask = x != 0
        return jnp.where(jnp.any(mask), jnp.mean(x[mask]), 0.0)

    metrics = {
        "eval/loss/avg": float(compute_mean(total_losses)),
        "eval/loss/value/avg": float(compute_mean(value_losses)),
        "eval/loss/variance/avg": float(compute_mean(var_losses))
    }

    wandb.log(metrics, step=step)

def evaluate(
        rng_key: chex.PRNGKey,
        train_state: TrainState,
        step_fn: Callable,
        reset_fn: Callable,
        recurrent_fn: Callable,
        policy_fn: PolicyFn,
        tree_policy_fn: Callable,
        eval_fn: EvalFn,
        tetrominoes: Tetrominoes,
        step: int,
        config: Config,
):
    rollouts = collect_eval_play_data(rng_key, config, step_fn, reset_fn, recurrent_fn, policy_fn, tree_policy_fn, eval_fn, train_state,
                                         tetrominoes)

    jax.debug.print("Logging metrics and video...")
    log_metrics_and_video(
        rollouts=rollouts,
        step=step,
    )

    jax.debug.print("Evaluation complete.")
