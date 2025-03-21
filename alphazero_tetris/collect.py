from functools import partial
from typing import Callable, Any

import chex
import jax
import jax.numpy as jnp
from flax.nnx import TrainState
from tetris_gymnasium.functional.tetrominoes import Tetrominoes

from alphazero_tetris.base import PolicyFn, EvalFn, RootFnOutput
from alphazero_tetris.config import Config
from alphazero_tetris.network import train
from alphazero_tetris.replay_buffer import ReplayBufferState, ReplayBufferInfo, add_to_replay_buffer

def collect_play_data(
        rng_key: chex.PRNGKey,
        step_fn: Callable,
        reset_fn: Callable,
        recurrent_fn: Callable,
        policy_fn: PolicyFn,
        tree_policy_fn: Callable,
        eval_fn: EvalFn,
        tetrominoes: Tetrominoes,
        rb_state: ReplayBufferState,
        rb_info: ReplayBufferInfo,
        train_state: TrainState,
        temperature: float,
        config: Config,
        num_samples: int
):
    def collect_experiences(i, carry):
        rng_key, state, obs, rb_state, rb_info = carry

        # Split RNG keys for batch
        rng_keys = jax.random.split(rng_key, num=4)
        next_rng_key = rng_keys[0]
        rng_key_reset = rng_keys[1]
        rng_key_root = rng_keys[2]
        rng_key_policy = rng_keys[3]

        # Create root outputs for batch
        # eval expects batched inputs
        state_batched = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)
        obs_batched = jnp.expand_dims(obs, axis=0)
        eval_root = eval_fn(rng_key_root, train_state, state_batched, obs_batched)
        root = RootFnOutput(state=state, value=eval_root.value, variance=eval_root.variance, observation=obs, prior_logits=eval_root.p)
        policy_output = policy_fn(train_state, rng_key_policy, root, recurrent_fn, tree_policy_fn, config.num_simulations, config, temperature)
        action = policy_output.action

        # Add experiences to replay buffer
        rb_state, rb_info = add_to_replay_buffer(
            rb_state,
            rb_info,
            policy_output.value,
            policy_output.variance,
            policy_output.p,
            obs
        )

        # Step environments
        step_state, step_obs, reward, terminated, info = step_fn(
            tetrominoes,
            state,
            action
        )

        _, new_state, new_obs = jax.lax.cond(
            terminated,
            lambda rng: reset_fn(tetrominoes, rng_key),
            lambda rng: (rng, step_state, step_obs),
            rng_key_reset
        )

        return (next_rng_key, new_state, new_obs, rb_state, rb_info)

    # Split key for initialization and main loop
    rng_key, rng_key_init = jax.random.split(rng_key)

    _, init_state, init_obs = reset_fn(tetrominoes, rng_key_init)
    rng_key, state, obs, rb_state, rb_info = jax.lax.fori_loop(
        0, num_samples, collect_experiences,
        (rng_key, init_state, init_obs, rb_state, rb_info)
    )

    return rb_state, rb_info


@partial(jax.jit, static_argnames=('step_fn', 'reset_fn', 'recurrent_fn', 'policy_fn', 'tree_policy_fn', 'eval_fn', 'config'))
def collect_epoch(
        rng_key: chex.PRNGKey,
        step_fn: Callable,
        reset_fn: Callable,
        recurrent_fn: Callable,
        policy_fn: PolicyFn,
        tree_policy_fn: Callable,
        eval_fn: EvalFn,
        config: Config,
        tetrominoes: Tetrominoes,
        rb_state: ReplayBufferState,
        rb_info: ReplayBufferInfo,
        train_state: TrainState,
        temperature: float,
        num_samples: int
):
    jax.debug.print("Collecting {} samples of play data....", config.collect_samples_per_epoch)

    rb_state, rb_info = collect_play_data(
        rng_key=rng_key,
        step_fn=step_fn,
        reset_fn=reset_fn,
        recurrent_fn=recurrent_fn,
        policy_fn=policy_fn,
        tree_policy_fn=tree_policy_fn,
        eval_fn=eval_fn,
        tetrominoes=tetrominoes,
        rb_state=rb_state,
        rb_info=rb_info,
        train_state=train_state,
        temperature=temperature,
        config=config,
        num_samples=num_samples
    )

    return rb_state, rb_info


def train_epoch(rng_key: chex.PRNGKey, train_state: TrainState, max_train_steps: int, loss_fn: Callable,
                rb_state: ReplayBufferState, rb_info: ReplayBufferInfo, config: Config):
    jax.debug.print("Training for a maximum of {} steps with batch size {}...",
                    max_train_steps,
                    config.train_batch_size)

    train_state, metrics = train(
        rng_key=rng_key,
        state=train_state,
        loss_fn=loss_fn,
        rb_state=rb_state,
        rb_info=rb_info,
        max_train_steps=max_train_steps,
        batch_size=config.train_batch_size,
        config=config,
    )

    return train_state, metrics