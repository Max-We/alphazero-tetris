import os
import os
import shutil
from dataclasses import replace
from functools import partial

import jax
import tyro
import wandb
from orbax.checkpoint import PyTreeCheckpointer
from tetris_gymnasium.envs.tetris_fn import step, reset
from tetris_gymnasium.functional.core import EnvConfig
from tetris_gymnasium.functional.queue import bag_queue_get_next_element, create_bag_queue
from tetris_gymnasium.functional.tetrominoes import TETROMINOES

from alphazero_tetris.base import make_recurrent_fn, AlgorithmConfig, make_nn_eval_fn
from alphazero_tetris.config import Config, print_config
from alphazero_tetris.device import training_device, collect_device
from alphazero_tetris.eval import evaluate
from alphazero_tetris.network import save_checkpoint, archive_final_checkpoint, make_loss_fn, \
    puct_loss, make_p_nn, kl_loss
from alphazero_tetris.policies import policy, tree_policy_puct, tree_policy_clt
from alphazero_tetris.replay_buffer import init_replay_buffer, augment_replay_buffer
from alphazero_tetris.collect import train_epoch, collect_epoch


def make_env():
    env_config = EnvConfig(width=10, height=20, padding=10, queue_size=7)

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

def main(config):
    # for convenience the run-name is used as indicator of a new "production" run
    is_production_run = config.wandb_run_name is not None

    # Logging
    with wandb.init(
        name=config.wandb_run_name,
        project=config.project_name,
        config=vars(config),
        mode="online" if is_production_run else "offline",
    ):
        # Update config with W&B config (could be different, when using sweeps)
        config = replace(Config(**wandb.config), obs_size=tuple(config.obs_size))

        # Optional: clear previous run (local use only)
        if not is_production_run:
            if os.path.exists(config.checkpoint_dir) and "alphazero-tetris" in config.checkpoint_dir:
                shutil.rmtree(config.checkpoint_dir)

        # Initialize environment
        step_fn, reset_fn, env_config = make_env()

        # Initialize policy / algorithm config
        eval_nn_fn = make_nn_eval_fn()
        algorithm = AlgorithmConfig(
            num_simulations=config.num_simulations,
            policy_fn=policy,
            tree_policy_fn=tree_policy_puct,
            eval_fn=eval_nn_fn,
            loss_fn=make_loss_fn(puct_loss),
            recurrent_fn = make_recurrent_fn(step_fn, eval_nn_fn, TETROMINOES, config),
        )

        # Alternative: custom CLT-based policy without priors
        # algorithm = AlgorithmConfig(
        #     num_simulations=config.num_simulations,
        #     policy_fn=policy,
        #     tree_policy_fn=tree_policy_clt,
        #     eval_fn=eval_nn_fn,
        #     loss_fn=make_loss_fn(kl_loss),
        #     recurrent_fn = make_recurrent_fn(step_fn, eval_nn_fn, TETROMINOES, config),
        # )

        # Initialize nn
        rng_key = jax.random.PRNGKey(config.seed)
        rng_key, rng_key_net = jax.random.split(rng_key)
        checkpointer = PyTreeCheckpointer()
        train_state, init_epoch = make_p_nn(rng_key_net, checkpointer, config.checkpoint_dir)

        # Run training loop
        eval_interval = int(config.train_epochs * config.wandb_eval_fraction)
        checkpoint_interval = int(config.train_epochs * config.checkpoint_fraction)
        for e in range(init_epoch, (config.train_epochs + 1)):
            # Progress indicator
            progress = e / config.train_epochs
            progress_percent = int(progress * 100)
            jax.debug.print("Epoch: {}/{} Progress: {}%", e, config.train_epochs, progress_percent)

            # RNG
            rng_key, rng_key_collect, rng_key_train, rng_key_eval = jax.random.split(rng_key, 4)

            # Temperature schedule
            temperature = (1.0 if progress <= 0.5 else (0.5 if progress <= 0.75 else 0.25)) if config.temperature else 0.0

            # 1. Data collection
            # Initialize rb (has to be fresh every epoch to use augmentation in this implementation)
            rb_state, rb_info = init_replay_buffer(
                config.collect_samples_per_epoch,
                config.obs_size,
                config.num_actions
            )

            rb_state, rb_info = collect_epoch(
                rng_key=rng_key_collect,
                step_fn=step_fn,
                reset_fn=reset_fn,
                recurrent_fn=algorithm.recurrent_fn,
                policy_fn=algorithm.policy_fn,
                tree_policy_fn=algorithm.tree_policy_fn,
                eval_fn=algorithm.eval_fn,
                tetrominoes=TETROMINOES,
                rb_state=rb_state,
                rb_info=rb_info,
                train_state=train_state,
                temperature=temperature,
                config=config,
                num_samples=config.collect_samples_per_epoch
            )
            rb_state = augment_replay_buffer(rb_state)

            # 2. Training
            rng_key_train = jax.device_put(rng_key_train, training_device)
            train_state = jax.device_put(train_state, training_device)
            rb_state = jax.device_put(rb_state, training_device)
            rb_info = jax.device_put(rb_info, training_device)
            train_state, train_metrics = train_epoch(
                rng_key=rng_key_train,
                train_state=train_state,
                max_train_steps=config.train_max_steps,
                loss_fn=algorithm.loss_fn,
                rb_state=rb_state,
                rb_info=rb_info,
                config=config,
            )
            train_state = jax.device_put(train_state, collect_device)

            # Logging train stats in W&B
            train_metrics["train/temperature"] = temperature
            jax.debug.print("Metrics: {}", train_metrics)
            wandb.log(train_metrics, step=e)

            # 3. Evaluation (optional)
            if e % eval_interval == 0:
                jax.debug.print("Evaluating...")
                evaluate(
                    rng_key=rng_key_eval,
                    train_state=train_state,
                    step_fn=step_fn,
                    reset_fn=reset_fn,
                    recurrent_fn=algorithm.recurrent_fn,
                    policy_fn=algorithm.policy_fn,
                    tree_policy_fn=algorithm.tree_policy_fn,
                    eval_fn=algorithm.eval_fn,
                    tetrominoes=TETROMINOES,
                    step=e,
                    config=config
                )

            # 4. Checkpointing (optional)
            if e % checkpoint_interval == 0:
                jax.debug.print("Checkpointing...")
                save_checkpoint(checkpointer, train_state, e, config.checkpoint_dir)

        # Archive final checkpoint and upload to Weights & Biases
        final_checkpoint = archive_final_checkpoint(config)
        wandb.save(final_checkpoint)

if __name__ == "__main__":
    # Optional flags for debugging jax
    # jax.config.update('jax_disable_jit', True)
    # jax.config.update("jax_log_compiles", True)
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
    jax.debug.print("Starting training script")
    jax.debug.print("Available devices: {}", jax.devices())
    jax.debug.print("JAX backend: {}", jax.default_backend())

    # Run training
    config = tyro.cli(Config)
    print_config(config)
    main(config)
