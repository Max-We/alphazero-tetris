from functools import partial
from typing import Callable, Tuple, Any, ClassVar

import chex
import jax
import jax.numpy as jnp
from gymnasium.experimental.functional import Params
from tetris_gymnasium.functional.core import State

from alphazero_tetris.config import Config
from alphazero_tetris.network import inference

Action = chex.Array
RecurrentState = State

@chex.dataclass(frozen=True)
class SearchSummary:
  """Stats from MCTS search."""
  value: chex.Array
  variance: chex.Array
  p: chex.Array

@chex.dataclass(frozen=True)
class MctsTree:
    # node base info
    node_state: Any
    node_values: chex.Array
    node_scores: chex.Array
    node_visits: chex.Array
    node_variances: chex.Array
    node_prior_logits: chex.Array
    # node additional info
    node_terminals: chex.Array
    node_observations: chex.Array
    # child nodes
    children_indices: chex.Array # [B, N, num_actions]
    children_values: chex.Array
    children_visits: chex.Array
    children_variances: chex.Array
    children_rewards: chex.Array
    children_scores: chex.Array
    children_discounts: chex.Array
    # parent (for backpropagation)
    parents: chex.Array
    action_from_parent: chex.Array

    ROOT_INDEX: ClassVar[int] = 0
    NO_PARENT: ClassVar[int] = -1
    UNVISITED: ClassVar[int] = -1

    @property
    def num_actions(self):
        return self.children_indices.shape[-1]

    def qvalues(self, indices):
        """Compute q-values for any node indices in the tree."""
        if jnp.asarray(indices).shape:
            return jax.vmap(_unbatched_qvalues)(self, indices)
        else:
            return _unbatched_qvalues(self, indices)

    def summary(self) -> SearchSummary:
        """Extract summary statistics for the root node."""
        chex.assert_rank(self.node_values, 1)
        chex.assert_rank(self.node_variances, 1)

        value = self.node_values[MctsTree.ROOT_INDEX]
        variance = self.node_variances[MctsTree.ROOT_INDEX]
        visit_counts = self.children_visits[MctsTree.ROOT_INDEX].astype(value.dtype)
        total_counts = jnp.sum(visit_counts, axis=-1, keepdims=True)
        visit_probs = visit_counts / jnp.maximum(total_counts, 1)
        p = jnp.where(total_counts > 0, visit_probs, 1 / self.num_actions)

        # Return relevant stats.
        return SearchSummary(
            value=value,
            variance=variance,
            p=p
        )

@chex.dataclass(frozen=True)
class RecurrentFnOutput:
  reward: chex.Array
  score: chex.Array
  observation: chex.Array
  discount: chex.Array
  variance: chex.Array
  value: chex.Array
  p: chex.Array

@chex.dataclass(frozen=True)
class PolicyFnOutput:
  action: chex.Array
  value: chex.Array
  variance: chex.Array
  p: chex.Array
  tree: MctsTree

@chex.dataclass(frozen=True)
class RootFnOutput:
  variance: chex.Array
  value: chex.Array
  prior_logits: chex.Array
  state: RecurrentState
  observation: chex.Array

@chex.dataclass(frozen=True)
class EvaluationFnOutput:
  variance: chex.Array
  value: chex.Array
  p: chex.Array

RecurrentFn = Callable[
  [Params, Action, RecurrentState],
  Tuple[RecurrentFnOutput, RecurrentState]
]
RootFn = Callable[[Params, chex.Array, Any], RootFnOutput]
EvalFn = Callable[[chex.PRNGKey, Params, Any, chex.Array], EvaluationFnOutput]

PolicyFn = Callable[
  [Params, chex.PRNGKey, RootFnOutput, RecurrentFn, Callable, int, Config, float],
  PolicyFnOutput
]

@chex.dataclass(frozen=True)
class AlgorithmConfig:
    policy_fn: PolicyFn
    tree_policy_fn: Callable
    recurrent_fn: RecurrentFn
    eval_fn: EvalFn
    num_simulations: int
    loss_fn: Callable

@chex.dataclass(frozen=True)
class RolloutData:
    # root
    observation: chex.Array # [T, H, W]
    value: chex.Array # [T]
    variance: chex.Array # [T]
    # tree
    children_values: chex.Array # [T, C]
    children_variances: chex.Array # [T, C]
    children_rewards: chex.Array # [T, C]
    children_discounts: chex.Array # [T, C]
    children_visits: chex.Array # [T, C]
    # game
    reward: chex.Array # [T]
    score: chex.Array # [T]
    lines_cleared: chex.Array # [T]
    terminated: chex.Array # [T]

# Utility functions
def _unbatched_qvalues(tree: MctsTree, index: int) -> int:
  return (  # pytype: disable=bad-return-type  # numpy-scalars
      tree.children_rewards[index] + tree.children_values[index] * tree.children_discounts[index]
  )

def update(x, vals, *indices):
  return x.at[indices].set(vals)
batch_update = jax.vmap(update)

def get_random_index(key: chex.PRNGKey, mask: chex.Array):
  # Create uniform dist
  rdm_arr = jax.random.uniform(key, shape=mask.shape)
  # Select random element form uniform dist, masking out the zeros
  return jnp.argmax(rdm_arr * mask)

def random_argmax(key: chex.PRNGKey, arr: chex.Array):
  # Find the maximum value
  max_value = arr.max()
  mask = (arr == max_value)
  # Pick a random index from the mask
  return get_random_index(key, mask)


# Recurrent fn
def make_recurrent_fn(step_fn, eval_fn: EvalFn, tetrominoes, config: Config) -> RecurrentFn:
    batched_step_fn = jax.vmap(step_fn, in_axes=(None, 0,0))

    def recurrent_fn(params, action, state) -> Tuple[RecurrentFnOutput, RecurrentState]:
        batch_size = action.shape[0]

        # Step the provided action
        new_state, new_obs, reward, done, info = batched_step_fn(tetrominoes, state, action)
        # Evaluate new state
        new_eval: EvaluationFnOutput = eval_fn(new_state.rng_key[0], params, new_state, new_obs)

        output = RecurrentFnOutput(
            reward=reward,
            score=new_state.score,
            observation=new_obs,
            discount=jnp.full(batch_size, config.discount)*(1-done),
            variance=new_eval.variance,
            value=new_eval.value,
            p=new_eval.p
        )

        return output, new_state
    return recurrent_fn


def make_nn_eval_fn() -> EvalFn:
    """Evaluation by neural network."""
    def nn_eval_fn(rng_key, params, state, observation) -> EvaluationFnOutput:
        value, variance, p = inference(params, observation)
        return EvaluationFnOutput(
            value=value,
            variance=variance,
            p=p
        )


    return nn_eval_fn

def make_rr_eval_fn(step_fn, tetrominoes, config: Config) -> EvalFn:
    """Evaluation by random rollout."""
    vmap_step = jax.vmap(step_fn, in_axes=(None, 0, 0))

    def rr_eval_fn(rng_key, params, state, observation) -> EvaluationFnOutput:
        """Evaluate by random-rollout"""
        batch_size = observation.shape[0]

        def body_fn(_, carry):
            rng, state, done, reward_sum = carry

            rng_new, rng_action = jax.random.split(rng)
            action = jax.random.randint(rng_action, shape=(batch_size,), minval=0, maxval=config.num_actions)
            new_state, new_obs, reward, new_done, info = vmap_step(tetrominoes, state, action)

            return rng_new, new_state, new_done, reward_sum + (reward * (1 - done))

        # Run for max 1000 steps
        final_carry = jax.lax.fori_loop(
            0, 100,
            body_fn,
            (rng_key, state, state.game_over, jnp.zeros(batch_size))
        )
        _, _, _, reward_random_rollout = final_carry

        return EvaluationFnOutput(
            value=reward_random_rollout,
            variance=jnp.square(reward_random_rollout),
            p=jnp.ones((batch_size, config.num_actions))/config.num_actions
        )

    return rr_eval_fn

def make_dummy_eval_fn() -> EvalFn:
    """Dummy evaluation function with fixed outputs."""
    def dummy_eval_fn(rng_key, params, state, observation) -> EvaluationFnOutput:
        batch_size = observation.shape[0]
        ones = jnp.ones(batch_size)
        return EvaluationFnOutput(
            value=ones*50,
            variance=ones*50*50,
            p=jnp.ones((batch_size, 7)) / 7
        )

    return dummy_eval_fn
