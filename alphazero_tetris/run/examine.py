import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import tyro
from orbax.checkpoint import PyTreeCheckpointer
from tetris_gymnasium.envs.tetris_fn import step, reset
from tetris_gymnasium.functional.core import EnvConfig
from tetris_gymnasium.functional.queue import bag_queue_get_next_element, create_bag_queue
from tetris_gymnasium.functional.tetrominoes import TETROMINOES

from alphazero_tetris.base import make_recurrent_fn, AlgorithmConfig, RolloutData, make_nn_eval_fn, make_rr_eval_fn, \
    make_dummy_eval_fn
from alphazero_tetris.config import Config
from alphazero_tetris.eval import collect_eval_play_data
from alphazero_tetris.network import make_p_nn
from alphazero_tetris.policies import policy, tree_policy_clt, norm_quantile, tree_policy_random


def make_env():
    env_config = EnvConfig(width=10, height=20, padding=10, queue_size=7)

    # Create batched versions of step and reset
    step_partial = partial(
        step,
        config=env_config,
        queue_fn=bag_queue_get_next_element
    )

    reset_partial = partial(
        reset,
        config=env_config,
        create_queue_fn=create_bag_queue,
        queue_fn=bag_queue_get_next_element,
    )

    return jax.jit(step_partial), jax.jit(reset_partial), env_config


def serialize_rollout(rollout_data: RolloutData) -> dict:
    # Get dimensions
    T = rollout_data.observation.shape[0]
    H = rollout_data.observation.shape[1]
    W = rollout_data.observation.shape[2]
    C = rollout_data.children_values.shape[1]

    first_termination = jnp.argmax(rollout_data.terminated, axis=0)
    episode_length = jnp.where(first_termination == 0, T, first_termination + 1)

    discount = rollout_data.children_discounts[:episode_length]
    value = rollout_data.children_rewards[:episode_length] + rollout_data.children_values[:episode_length] * discount
    variance = rollout_data.children_variances[:episode_length] * discount * discount
    children_visits = rollout_data.children_visits[:episode_length]
    parent_visits = jnp.expand_dims(jnp.sum(children_visits, axis=1), -1)
    exploit_component = value
    explore_component = norm_quantile(parent_visits) * jnp.sqrt(variance / children_visits)
    policy_scores = exploit_component + explore_component

    return {
        "metadata": {
            "timesteps": int(episode_length),
            "height": H,
            "width": W,
            "num_children": C
        },
        "root": {
            "observations": rollout_data.observation[:episode_length].tolist(),
            "values": rollout_data.value[:episode_length].tolist(),
            "variances": rollout_data.variance[:episode_length].tolist()
        },
        "tree": {
            "children_values": rollout_data.children_values[:episode_length].tolist(),
            "children_variances": rollout_data.children_variances[:episode_length].tolist(),
            "children_rewards": rollout_data.children_rewards[:episode_length].tolist(),
            "children_discounts": rollout_data.children_discounts[:episode_length].tolist(),
            "children_visits": rollout_data.children_visits[:episode_length].tolist(),
            "children_policy_scores": policy_scores.tolist()
        },
        "game": {
            "rewards": rollout_data.reward[:episode_length].tolist(),
            "scores": rollout_data.score[:episode_length].tolist(),
            "lines_cleared": rollout_data.lines_cleared[:episode_length].tolist(),
            "terminated": rollout_data.terminated[:episode_length].tolist()
        }
    }


def save_rollout(rollout_data: RolloutData, filepath: str):
    serialized = serialize_rollout(rollout_data)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(serialized, f)
    return serialized


def calculate_and_save_statistics(scores, steps, output_dir):
    """
    Calculate statistics and generate visualizations for the collected metrics.

    Args:
        scores: List of final scores from each iteration
        steps: List of episode lengths (steps) from each iteration
        output_dir: Directory to save the statistics and visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy arrays
    scores_array = np.array(scores)
    steps_array = np.array(steps)

    # Calculate statistics
    stats = {
        "score": {
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "mean": float(np.mean(scores_array)),
            "median": float(np.median(scores_array))
        },
        "steps": {
            "min": int(np.min(steps_array)),
            "max": int(np.max(steps_array)),
            "mean": float(np.mean(steps_array)),
            "median": float(np.median(steps_array))
        },
        "sample_size": len(scores)
    }

    # Print statistics
    print("\nSummary Statistics:")
    print(f"Sample Size (games): {stats['sample_size']}")
    print("\nScore Statistics:")
    print(f"  Min: {stats['score']['min']}")
    print(f"  Max: {stats['score']['max']}")
    print(f"  Mean: {stats['score']['mean']:.2f}")
    print(f"  Median: {stats['score']['median']}")
    print("\nSteps Statistics:")
    print(f"  Min: {stats['steps']['min']}")
    print(f"  Max: {stats['steps']['max']}")
    print(f"  Mean: {stats['steps']['mean']:.2f}")
    print(f"  Median: {stats['steps']['median']}")

    # Save statistics to file
    stats_file = os.path.join(output_dir, "summary_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Sample Size (games): {stats['sample_size']}\n\n")
        f.write("Score Statistics:\n")
        f.write(f"  Min: {stats['score']['min']}\n")
        f.write(f"  Max: {stats['score']['max']}\n")
        f.write(f"  Mean: {stats['score']['mean']:.2f}\n")
        f.write(f"  Median: {stats['score']['median']}\n\n")
        f.write("Steps Statistics:\n")
        f.write(f"  Min: {stats['steps']['min']}\n")
        f.write(f"  Max: {stats['steps']['max']}\n")
        f.write(f"  Mean: {stats['steps']['mean']:.2f}\n")
        f.write(f"  Median: {stats['steps']['median']}\n")

    # Create histograms
    plt.figure(figsize=(10, 6))
    plt.hist(scores_array, bins=min(10, len(scores) // 2 + 1), alpha=0.7, color='blue')
    plt.title('Distribution of Final Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "score_histogram.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(steps_array, bins=min(10, len(steps) // 2 + 1), alpha=0.7, color='green')
    plt.title('Distribution of Episode Lengths (Steps)')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "steps_histogram.png"))
    plt.close()

    return stats


if __name__ == "__main__":
    # Optional flags for debugging jax
    # jax.config.update('jax_disable_jit', True)
    # jax.config.update("jax_log_compiles", True)
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
    script_dir = Path(__file__).resolve().parent
    viewer_path = script_dir / "viewer.html"
    viewer_abs_path = viewer_path.resolve()
    checkpoints_path = script_dir / "checkpoints"
    checkpoints_abs_path = str(checkpoints_path.resolve())

    jax.debug.print("Starting examine script")
    jax.debug.print("Available devices: {}", jax.devices())
    jax.debug.print("JAX backend: {}", jax.default_backend())
    print(f"Open viewer: file://{viewer_abs_path}")

    # Parse command line arguments
    config = tyro.cli(Config)
    config = replace(config,
         project_name=config.project_name + "-test",
         wandb_eval_max_steps=10000,
         temperature=False,
         checkpoint_dir=checkpoints_abs_path,
         # seed=234
     )

    # Initialize environment with batch size
    step_fn, reset_fn, env_config = make_env()
    # initialize network (model, optimizer, etc.)
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, rng_key_net = jax.random.split(rng_key)
    checkpointer = PyTreeCheckpointer()
    train_state, init_epoch = make_p_nn(rng_key_net, checkpointer, config.checkpoint_dir)

    # pick 1:
    eval_rr_fn = make_rr_eval_fn(step_fn, TETROMINOES, config)
    eval_dd_fn = make_dummy_eval_fn()
    eval_nn_fn = make_nn_eval_fn()
    # configure algorithm
    algorithm = AlgorithmConfig(
        num_simulations=300,
        policy_fn=policy,
        tree_policy_fn=tree_policy_clt,
        eval_fn=eval_nn_fn,
        recurrent_fn=make_recurrent_fn(step_fn, eval_nn_fn, TETROMINOES, config),
        loss_fn=lambda x: 0.0
    )
    config = replace(config, num_simulations=algorithm.num_simulations)

    # Number of games to record
    n_games = 25

    # Create directories for examine data
    timestamp = time.strftime("%Y_%m_%d_%H%M%S")
    rollouts_dir = f"examine_out/{timestamp}/rollouts"
    stats_dir = f"examine_out/{timestamp}/stats"

    print(f"Starting examination with {n_games} games...")

    all_scores, all_steps = [], []
    for i in range(n_games):
        print(f"Game {i + 1}/{n_games}")

        # Split RNG key for this game
        rng_key, rng_key_iter = jax.random.split(rng_key)

        print(f"  Collecting rollout data...")
        rollout = collect_eval_play_data(
            rng_key=rng_key_iter,
            config=config,
            step_fn=step_fn,
            reset_fn=reset_fn,
            recurrent_fn=algorithm.recurrent_fn,
            policy_fn=algorithm.policy_fn,
            tree_policy_fn=algorithm.tree_policy_fn,
            eval_fn=algorithm.eval_fn,
            train_state=train_state,
            tetrominoes=TETROMINOES
        )

        # Save rollout to file
        rollout_filepath = os.path.join(rollouts_dir, f"rollout_{i + 1}.json")
        print(f"  Saving rollout data to {rollout_filepath}...")
        serialized = save_rollout(rollout, rollout_filepath)

        # Extract metrics from this rollout
        final_score = max(serialized["game"]["scores"])
        episode_length = serialized["metadata"]["timesteps"]

        # Store metrics
        all_scores.append(final_score)
        all_steps.append(episode_length)

        print(f"  Game {i + 1} complete - Score: {final_score}, Steps: {episode_length}")

    # Calculate and save summary statistics
    print("\nGenerating summary statistics and visualizations...")
    calculate_and_save_statistics(all_scores, all_steps, stats_dir)

    print(f"\nExamination complete. Results saved to:")
    print(f"  Rollouts: {os.path.abspath(rollouts_dir)}")
    print(f"  Statistics: {os.path.abspath(stats_dir)}")

    print(f"Open viewer: file://{viewer_abs_path}")
