from dataclasses import replace
from functools import partial
from typing import Tuple

import chex
import jax.numpy as jnp
import jax

from alphazero_tetris.config import Config



@chex.dataclass(frozen=True)
class ReplayBufferState:
    value: jnp.ndarray  # (augmented_dim, buffer_length)
    variance: jnp.ndarray  # (augmented_dim, buffer_length)
    p: jnp.ndarray  # (augmented_dim, buffer_length)
    observation: jnp.ndarray  # (augmented_dim, buffer_length, *obs_dims)

@chex.dataclass(frozen=True)
class ReplayBufferInfo:
    write_index: chex.Array  # Scalar index
    is_full: chex.Array  # Scalar flag

@jax.jit
def add_to_replay_buffer(
        state: ReplayBufferState,
        info: ReplayBufferInfo,
        value: jnp.ndarray,  # Scalar
        variance: jnp.ndarray,  # Scalar
        p: jnp.ndarray,  # (num_actions,)
        observation: jnp.ndarray,  # (*obs_dims)
) -> Tuple[ReplayBufferState, ReplayBufferInfo]:
    buffer_length = state.value.shape[1]

    # Single write index
    write_index = info.write_index
    new_write_index = (write_index + 1) % buffer_length

    # Update is_full flag
    is_full = jnp.logical_or(info.is_full, new_write_index < write_index)

    # Update buffer states using shared index
    new_state = ReplayBufferState(
        value=state.value.at[0,write_index].set(value),
        variance=state.variance.at[0,write_index].set(variance),
        p=state.p.at[0,write_index].set(p),
        observation=state.observation.at[0,write_index].set(observation)
    )
    # chex.assert_shape(new_state.value, (1, buffer_length))
    # chex.assert_shape(new_state.variance, (1, buffer_length))
    # chex.assert_shape(new_state.p, (1, buffer_length, 7))
    # chex.assert_shape(new_state.observation, (1, buffer_length, 20, 10))

    new_info = replace(
        info,
        write_index=new_write_index,
        is_full=is_full
    )

    return new_state, new_info

@partial(jax.jit, static_argnames=("sample_batch_size", "config",))
def sample_rb_train(
        rng_key: chex.PRNGKey,
        sample_batch_size: int,
        state: ReplayBufferState,
        info: ReplayBufferInfo,
        config: Config,
):
    return get_batch_from_replay_buffer(
        state,
        info,
        sample_batch_size,
        rng_key,
        0,
        config.collect_samples_per_epoch - int(config.train_eval_fraction * config.collect_samples_per_epoch)
    )

@partial(jax.jit, static_argnames=("sample_batch_size", "config",))
def sample_rb_val(
        rng_key: chex.PRNGKey,
        sample_batch_size: int,
        state: ReplayBufferState,
        info: ReplayBufferInfo,
        config: Config,
):
    return get_batch_from_replay_buffer(
        state,
        info,
        sample_batch_size,
        rng_key,
        config.collect_samples_per_epoch - int(config.train_eval_fraction * config.collect_samples_per_epoch),
        config.collect_samples_per_epoch
    )

def get_batch_from_replay_buffer(
        state: ReplayBufferState,
        info: ReplayBufferInfo,
        sample_batch_size: int,
        rng_key: chex.PRNGKey,
        batch_lower: int, # inclusive
        batch_upper: int # exclusive
) -> ReplayBufferState:
    augmented_dim, buffer_length = state.value.shape

    # Generate random batch and position indices
    rng_key_aug, rng_key_pos = jax.random.split(rng_key)
    aug_indices = jax.random.randint(
        rng_key_aug,
        shape=(sample_batch_size,),
        minval=0,
        maxval=augmented_dim
    )
    timestep_indices = jax.random.randint(
        rng_key_pos,
        shape=(sample_batch_size,),
        minval=batch_lower,
        maxval=batch_upper
    )

    return ReplayBufferState(
        value=state.value[aug_indices, timestep_indices],
        variance=state.variance[aug_indices, timestep_indices],
        p=state.p[aug_indices, timestep_indices],
        observation=state.observation[aug_indices, timestep_indices]
    )



@jax.jit
def augment_replay_buffer(
        state: ReplayBufferState,
) -> ReplayBufferState:
    """Augment replay buffer with flipped observations. Can be expanded with more augmentations."""
    *obs_prefix, height, width = state.observation.shape

    # Create flipped observations
    obs_reshaped = state.observation.reshape(-1, height, width)
    flipped_obs = jnp.flip(obs_reshaped, axis=2)  # Flip horizontally
    flipped_obs = flipped_obs.reshape(*obs_prefix, height, width)

    return ReplayBufferState(
        value=jnp.concatenate([state.value, state.value], axis=0),
        variance=jnp.concatenate([state.variance, state.variance], axis=0),
        p=jnp.concatenate([state.p, state.p], axis=0),
        observation=jnp.concatenate([state.observation, flipped_obs], axis=0)
    )

def init_replay_buffer(
        buffer_size: int,
        obs_size: Tuple[int, ...],
        num_actions: int
) -> Tuple[ReplayBufferState, ReplayBufferInfo]:
    memory = ReplayBufferState(
        value=jnp.zeros((1, buffer_size)),
        variance=jnp.zeros((1, buffer_size)),
        p=jnp.zeros((1, buffer_size, num_actions)),
        observation=jnp.zeros((1, buffer_size,) + obs_size)
    )

    info = ReplayBufferInfo(
        write_index=jnp.array(0, dtype=jnp.int32),
        is_full=jnp.array(False)
    )

    return memory, info