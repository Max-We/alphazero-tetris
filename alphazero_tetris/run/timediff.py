import time
from dataclasses import replace
from functools import partial

import jax
import tyro
from orbax.checkpoint import PyTreeCheckpointer
from tetris_gymnasium.envs.tetris_fn import reset, step
from tetris_gymnasium.functional.core import EnvConfig
from tetris_gymnasium.functional.queue import bag_queue_get_next_element, create_bag_queue
from tetris_gymnasium.functional.tetrominoes import TETROMINOES

import wandb
from alphazero_tetris.base import make_recurrent_fn, AlgorithmConfig, make_dummy_eval_fn
from alphazero_tetris.config import Config
from alphazero_tetris.eval import evaluate
from alphazero_tetris.network import make_p_nn
from alphazero_tetris.policies import policy, \
    tree_policy_dfs, tree_policy_bfs


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


def timed_evaluate(*args, **kwargs):
    evaluate(*args, **kwargs)

    # Now measure actual runtime
    start_time = time.time()
    evaluate(*args, **kwargs)
    end_time = time.time()

    elapsed_seconds = end_time - start_time
    return elapsed_seconds


if __name__ == "__main__":
    # Optional flags for debugging jax
    # jax.config.update('jax_disable_jit', True)
    # jax.config.update("jax_log_compiles", True)
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
    jax.debug.print("Starting training script")
    jax.debug.print("Available devices: {}", jax.devices())
    jax.debug.print("JAX backend: {}", jax.default_backend())

    # Parse command line arguments
    config = tyro.cli(Config)

    # Logging
    wandb.init(
        project=config.project_name,
        config=vars(config),
        mode="online",
    )

    # Initialize environment with batch size
    step_fn, reset_fn, env_config = make_env()
    # initialize network (model, optimizer, etc.)
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, rng_key_net = jax.random.split(rng_key)
    checkpointer = PyTreeCheckpointer()
    train_state, init_epoch = make_p_nn(rng_key_net, checkpointer, config.checkpoint_dir)

    # vgl. clt mit bestem exp
    num_sims = 300
    # eval_rr_fn = make_rr_eval_fn(step_fn, TETROMINOES, config)
    eval_dd_fn = make_dummy_eval_fn()
    # eval_nn_fn = make_nn_eval_fn()

    algo_configs = {
        "mcts 1e2 dfs": AlgorithmConfig(
            num_simulations=100,
            policy_fn=policy,
            tree_policy_fn=tree_policy_dfs,
            eval_fn=eval_dd_fn,
            recurrent_fn = make_recurrent_fn(step_fn, eval_dd_fn, TETROMINOES, config),
            loss_fn=lambda x: 0.0
        ),
        "mcts 5e2 dfs": AlgorithmConfig(
            num_simulations=500,
            policy_fn=policy,
            tree_policy_fn=tree_policy_dfs,
            eval_fn=eval_dd_fn,
            recurrent_fn=make_recurrent_fn(step_fn, eval_dd_fn, TETROMINOES, config),
            loss_fn=lambda x: 0.0
        ),
        "mcts 1e3 dfs": AlgorithmConfig(
            num_simulations=1000,
            policy_fn=policy,
            tree_policy_fn=tree_policy_dfs,
            eval_fn=eval_dd_fn,
            recurrent_fn=make_recurrent_fn(step_fn, eval_dd_fn, TETROMINOES, config),
            loss_fn=lambda x: 0.0
        ),
        "mcts 1e2 bfs": AlgorithmConfig(
            num_simulations=100,
            policy_fn=policy,
            tree_policy_fn=tree_policy_bfs,
            eval_fn=eval_dd_fn,
            recurrent_fn = make_recurrent_fn(step_fn, eval_dd_fn, TETROMINOES, config),
            loss_fn=lambda x: 0.0
        ),
        "mcts 5e2 bfs": AlgorithmConfig(
            num_simulations=500,
            policy_fn=policy,
            tree_policy_fn=tree_policy_bfs,
            eval_fn=eval_dd_fn,
            recurrent_fn=make_recurrent_fn(step_fn, eval_dd_fn, TETROMINOES, config),
            loss_fn=lambda x: 0.0
        ),
        "mcts 1e3 bfs": AlgorithmConfig(
            num_simulations=1000,
            policy_fn=policy,
            tree_policy_fn=tree_policy_bfs,
            eval_fn=eval_dd_fn,
            recurrent_fn=make_recurrent_fn(step_fn, eval_dd_fn, TETROMINOES, config),
            loss_fn=lambda x: 0.0
        ),
    }

    timestep = 1
    for k, algorithm in algo_configs.items():
        jax.debug.print("Running configuration {}: <{}>", timestep, k)

        if config.num_simulations != algorithm.num_simulations:
            config = replace(config, num_simulations=algorithm.num_simulations)

        rng_key = rng_key  # always use same rng for all algorithms

        # Time the evaluation
        runtime = timed_evaluate(
            rng_key=rng_key,
            train_state=train_state,
            step_fn=step_fn,
            reset_fn=reset_fn,
            recurrent_fn=algorithm.recurrent_fn,
            policy_fn=algorithm.policy_fn,
            tree_policy_fn=algorithm.tree_policy_fn,
            eval_fn=algorithm.eval_fn,
            tetrominoes=TETROMINOES,
            step=timestep,
            config=config
        )

        # Log the runtime
        jax.debug.print("Configuration {} runtime: {:.2f} seconds", k, runtime)
        wandb.log({f"runtime_{k}": runtime}, step=timestep)

        timestep += 1

    wandb.finish()
