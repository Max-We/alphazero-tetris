import os
import shutil
from functools import partial
from typing import Tuple, Callable

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import orbax_utils
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer

from alphazero_tetris.config import Config
from alphazero_tetris.replay_buffer import ReplayBufferState, ReplayBufferInfo, \
    sample_rb_train, sample_rb_val


class FeatureNet(nn.Module):
    """
    A network that takes a vector of stack heights as input and outputs value and variance
    """

    @nn.compact
    def __call__(self, x):
        x = jnp.squeeze(x, axis=-1)

        # Hidden layers
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=2)(x)

        # Output value and variance
        val, var = x[:, 0] + 1e-1, nn.softplus(x[:, 1])
        return val, var, jnp.zeros((x.shape[0], 7))


class Net(nn.Module):
    input_shape: Tuple[int, int] = (20, 10)
    eps: float = 1e-1

    @nn.compact
    def __call__(self, x):
        kernel_size = 3
        stride = 1
        filters = 32
        bias = True

        x = nn.Conv(features=filters, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), use_bias=bias, padding="VALID")(
            x)
        x = nn.relu(x)
        x = nn.Conv(features=filters, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), use_bias=bias, padding="VALID")(
            x)
        x = nn.relu(x)
        x = nn.Conv(features=filters, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), use_bias=bias, padding="VALID")(
            x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=2)(x)

        val, var = x[:, 0]+self.eps, nn.softplus(x[:, 1])
        return val, var, jnp.zeros((x.shape[0], 7))

class PNet(nn.Module):
    input_shape: Tuple[int, int] = (20, 10)
    eps: float = 1e-1

    @nn.compact
    def __call__(self, x):
        kernel_size = 3
        stride = 1
        filters = 32
        bias = True

        x = nn.Conv(features=filters, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), use_bias=bias, padding="VALID")(
            x)
        x = nn.relu(x)
        x = nn.Conv(features=filters, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), use_bias=bias, padding="VALID")(
            x)
        x = nn.relu(x)
        x = nn.Conv(features=filters, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), use_bias=bias, padding="VALID")(
            x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1+1+7)(x)

        val, var, p = x[:, 0]+self.eps, nn.softplus(x[:, 1]), nn.softmax(x[:, 2:])
        return val, var, p

def mse_loss(means, variances, ps, target_means, target_variances, target_ps):
    mse_mean = jnp.square(target_means - means)
    mse_var = jnp.square(target_variances - variances)
    return mse_mean + mse_var, mse_mean, mse_var, 0

def kl_loss(means, variances, ps, target_means, target_variances, target_ps):
    std, target_std = jnp.sqrt(variances), jnp.sqrt(target_variances)
    mean_div = 0.5 * (jnp.square(target_means-means) / target_variances)
    var_div = jnp.square((std/target_std)) + jnp.log(target_variances / variances) - 1
    kl = 0.5 * (mean_div + var_div)

    return kl, 0.5 * mean_div, 0.5 * var_div, 0

def puct_loss(values, variances, ps, target_values, target_variances, target_ps):
    value_loss = jnp.square(target_values - values)
    policy_loss = -jnp.mean(jnp.sum(target_ps * jnp.log(ps), axis=-1))
    loss = 0.5 * value_loss + policy_loss
    return loss, value_loss, 0, policy_loss

def make_loss_fn(ll_fn: Callable):
    ll_fn_jitted = jax.jit(ll_fn)

    @jax.jit
    def calc_loss(state, params, batch: ReplayBufferState):
        observation = batch.observation
        target_values, target_variance, target_ps = batch.value, batch.variance, batch.p

        variables = {'params': params}
        net_input = jnp.expand_dims(observation, axis=-1)

        values, variances, ps = state.apply_fn(variables, net_input)

        ll, value_loss, var_loss, p_loss = ll_fn_jitted(values, variances, ps, target_values, target_variance, target_ps)

        return jnp.mean(ll), {
            'loss': jnp.mean(ll),
            'std': jnp.std(ll, ddof=1),
            'value_loss': jnp.mean(value_loss),
            'var_loss': jnp.mean(var_loss),
            'p_loss': jnp.mean(p_loss)
        }

    return calc_loss


@partial(jax.jit, static_argnames=('loss_fn'))
def train_step(state: TrainState, loss_fn: Callable, batch: ReplayBufferState):
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    (loss, aux_metrics), grads = grad_fn(state, state.params, batch)

    # Compute gradient norm
    grad_norm = optax.global_norm(grads)

    # Update state
    state = state.apply_gradients(grads=grads)

    metrics = {
        'loss': aux_metrics['loss'],
        'std': aux_metrics['std'],
        'grad_norm': grad_norm,
        'loss/value': aux_metrics['value_loss'],
        'loss/variance': aux_metrics['var_loss'],
        'loss/p': aux_metrics['p_loss'],
    }

    return state, metrics


def init_metrics():
    """Initialize metrics dictionary with new components"""
    return {
        'steps': jnp.zeros(()),
        'grad_norm/avg': jnp.zeros(()),
        'grad_norm/max': jnp.full((), -jnp.inf),
        'train_loss/avg': jnp.zeros(()),
        'train_loss/value/avg': jnp.zeros(()),
        'train_loss/variance/avg': jnp.zeros(()),
        'train_loss/p/avg': jnp.zeros(()),
        'train_std/avg': jnp.zeros(()),
        'val_loss/avg': jnp.zeros(()),
        'val_loss/value/avg': jnp.zeros(()),
        'val_loss/variance/avg': jnp.zeros(()),
        'val_loss/p/avg': jnp.zeros(()),
        'val_std/avg': jnp.zeros(()),
    }


def train(rng_key: chex.PRNGKey,
          state: TrainState,
          loss_fn: Callable,
          rb_state: ReplayBufferState,
          rb_info: ReplayBufferInfo,
          max_train_steps: int,
          batch_size: int,
          config: Config
          ):
    metrics_dict = init_metrics()
    best_state = state
    best_step = 0

    rb_state = rb_state.replace(
        value=jnp.maximum(rb_state.value, 1e-1),
        variance=jnp.maximum(rb_state.variance, 0.7),
    )

    patience = 10
    best_val_loss = float('inf')
    fails = 0

    for i in range(max_train_steps):
        rng_key, train_key, val_key = jax.random.split(rng_key, 3)

        batch = sample_rb_train(train_key, batch_size, rb_state, rb_info, config)
        state, train_metrics = train_step(state, loss_fn, batch)

        if i % config.training_patience_interval == 0:
            val_batch = sample_rb_val(val_key, batch_size, rb_state, rb_info, config)
            _, val_metrics = loss_fn(state, state.params, val_batch)

            val_loss_mean = val_metrics['loss']
            val_loss_std = val_metrics['std'] / (batch_size ** 0.5)

            if val_loss_mean - best_val_loss < val_loss_std:
                fails = 0
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                    best_state = state
                    best_step = i
            else:
                fails += 1
                if fails >= patience:
                    print(f"Early stopping training at step {i}")
                    break

        metrics_dict = {
            'steps': i,
            'grad_norm/avg': (metrics_dict['grad_norm/avg'] * i + train_metrics['grad_norm']) / (i + 1),
            'grad_norm/max': jnp.maximum(metrics_dict['grad_norm/max'], train_metrics['grad_norm']),
            'train_loss/avg': (metrics_dict['train_loss/avg'] * i + train_metrics['loss']) / (i + 1),
            'train_loss/value/avg': (metrics_dict['train_loss/value/avg'] * i + train_metrics['loss/value']) / (i + 1),
            'train_loss/variance/avg': (metrics_dict['train_loss/variance/avg'] * i + train_metrics['loss/variance']) / (i + 1),
            'train_loss/p/avg': (metrics_dict['train_loss/p/avg'] * i + train_metrics['loss/p']) / (i + 1),
            'train_std/avg': (metrics_dict['train_std/avg'] * i + train_metrics['std']) / (i + 1),
            'val_loss/avg': (metrics_dict['val_loss/avg'] * i + val_metrics['loss']) / (i + 1),
            'val_loss/value/avg': (metrics_dict['val_loss/value/avg'] * i + val_metrics['value_loss']) / (i + 1),
            'val_loss/variance/avg': (metrics_dict['val_loss/variance/avg'] * i + val_metrics['var_loss']) / (i + 1),
            'val_loss/p/avg': (metrics_dict['val_loss/p/avg'] * i + val_metrics['p_loss']) / (i + 1),
            'val_std/avg': (metrics_dict['val_std/avg'] * i + val_metrics['std']) / (i + 1),
        }

    final_metrics = {f"train/{k}": v for k, v in metrics_dict.items()}
    final_metrics['train/best_step'] = best_step

    return best_state, final_metrics

def inference(train_state: TrainState, observation: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
    variables = {'params': train_state.params}
    net_input = jnp.expand_dims(observation, axis=-1)

    value, variance, p = train_state.apply_fn(variables, net_input)

    return value, variance, p

def make_feature_nn(rng_key: chex.PRNGKey, checkpointer: PyTreeCheckpointer, ckpt_dir: str = None):
    model = FeatureNet()
    variables = model.init(rng_key, jnp.ones((1, 23, 1)))

    # Separate params and bounds
    params = variables['params']

    optimizer = optax.chain(
        optax.add_decayed_weights(weight_decay=1e-3),
        optax.yogi(
            learning_rate=1e-3,
            b1=0.9,
            b2=0.999,
            eps=1e-3,
        ),
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    state, epoch = handle_checkpoint(ckpt_dir, checkpointer, state)
    return state, epoch

def make_p_nn(rng_key: chex.PRNGKey, checkpointer: PyTreeCheckpointer, ckpt_dir: str = None):
    model = PNet()
    variables = model.init(rng_key, jnp.ones((1, 20, 10, 1)))

    # Separate params and bounds
    params = variables['params']

    optimizer = optax.chain(
        optax.add_decayed_weights(weight_decay=1e-3),
        optax.yogi(
            learning_rate=1e-3,
            b1=0.9,
            b2=0.999,
            eps=1e-3,
        ),
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    state, epoch = handle_checkpoint(ckpt_dir, checkpointer, state)
    return state, epoch

def handle_checkpoint(ckpt_dir, checkpointer, state: TrainState):
    if ckpt_dir is not None:
        try:
            checkpoint_files = os.listdir(ckpt_dir)
            steps = []
            for filename in checkpoint_files:
                if filename.startswith('checkpoint_'):
                    try:
                        step = int(filename.split('_')[1])
                        steps.append(step)
                    except ValueError:
                        continue

            if steps:
                latest_epoch = max(steps)
                state = checkpointer.restore(f"{ckpt_dir}/checkpoint_{latest_epoch}", item=state)
                print(f"Restored checkpoint from step {latest_epoch}")
                return state, latest_epoch + 1 # start from next epoch

        except (FileNotFoundError, ValueError, Exception) as e:
            print(f"No checkpoint found: {e}")
            print("Using freshly initialized state")

    return state, 1


def load_checkpoint(checkpointer, ckpt_dir):
    # get latest checkpoint from dir
    options = checkpointer.get_directory_iterator_options(recursive=True)
    directory_iterator = checkpointer.directory_iterator(ckpt_dir, options)
    latest_step = max(directory_iterator)

    return checkpointer.restore(f"{ckpt_dir}/checkpoint_{latest_step}")

def save_checkpoint(checkpointer, state: TrainState, step, ckpt_dir):
    save_args = orbax_utils.save_args_from_target(state)
    checkpointer.save(f"{ckpt_dir}/checkpoint_{step}", state, save_args=save_args)

def archive_final_checkpoint(config):
    final_checkpoint = f"{config.checkpoint_dir}/checkpoint_{config.train_epochs}"
    shutil.make_archive(final_checkpoint, 'zip', config.checkpoint_dir)
    return final_checkpoint + ".zip"
