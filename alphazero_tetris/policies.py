from typing import Callable, Optional

import chex
import jax
import jax.numpy as jnp
from flax.nnx import TrainState

from alphazero_tetris.base import RootFnOutput, RecurrentFn, SearchSummary, random_argmax, PolicyFnOutput
from alphazero_tetris.config import Config
from alphazero_tetris.mcts import MctsTree, search


def norm_quantile(t):
    """http://m-hikari.com/ams/ams-2014/ams-85-88-2014/epureAMS85-88-2014.pdf"""
    alpha = 1 - 1 / t
    q = 10 * jnp.log(1 - jnp.log(-jnp.log(alpha) / jnp.log(2)) / jnp.log(22)) / jnp.log(41)
    return q

def tree_policy_random(rng_key: chex.PRNGKey, tree: MctsTree, node_index: int, temperature: float):
    return jax.random.randint(rng_key, shape=(), minval=0, maxval=tree.children_visits.shape[-1])

def tree_policy_bfs(rng_key: chex.PRNGKey, tree: MctsTree, node_index: int, temperature: float):
    return jnp.argmin(tree.children_visits[node_index])

def tree_policy_dfs(rng_key: chex.PRNGKey, tree: MctsTree, node_index: int, temperature: float):
    return jnp.argmax(tree.children_visits[node_index])

def tree_policy_value(rng_key: chex.PRNGKey, tree: MctsTree, node_index: int, temperature: float):
    return random_argmax(rng_key, tree.children_values[node_index])

def tree_policy_uct(rng_key: chex.PRNGKey, tree: MctsTree, node_index: int, temperature: float):
    c = jnp.sqrt(2)
    parent_visits = tree.node_visits[node_index]
    children_visits = tree.children_visits[node_index]
    children_values = tree.children_values[node_index]

    return random_argmax(rng_key, children_values + c * jnp.sqrt(jnp.log(parent_visits) / children_visits))

def tree_policy_uct_sp(rng_key: chex.PRNGKey, tree: MctsTree, node_index: int, temperature: float):
    """https://dke.maastrichtuniversity.nl/m.winands/documents/CGSameGame.pdf"""
    parent_visits = tree.node_visits[node_index]
    children_visits = tree.children_visits[node_index]
    children_discounts = tree.children_discounts[node_index]
    children_values = tree.children_values[node_index] * children_discounts
    children_variances = tree.children_variances[node_index] * children_discounts * children_discounts

    c = 1
    d = 5000
    uct = children_values + c * jnp.sqrt(jnp.log(parent_visits) / children_visits)
    uct_sp = uct + jnp.sqrt(children_variances + (d / children_visits))

    return random_argmax(rng_key, uct_sp)

def tree_policy_clt(rng_key: chex.PRNGKey, tree: MctsTree, node_index: int, temperature: float):
    parent_visits = tree.node_visits[node_index]
    children_visits = tree.children_visits[node_index]
    discount = tree.children_discounts[node_index]
    value = tree.children_rewards[node_index] + tree.children_values[node_index] * discount
    variance = tree.children_variances[node_index] * discount * discount

    exploit_component = value
    explore_component = norm_quantile(parent_visits) * jnp.sqrt(variance / children_visits)
    clt_score = exploit_component + explore_component

    return random_argmax(rng_key, clt_score)

def tree_policy_thompson(rng_key: chex.PRNGKey, tree: MctsTree, node_index: int, temperature: float):
    discount = tree.children_discounts[node_index]
    # performs poorly with rewards and rr
    value = tree.children_rewards[node_index] + tree.children_values[node_index] * discount
    # value = tree.children_values[node_index] * discount
    variance = tree.children_variances[node_index] * discount * discount

    samples = jax.random.normal(rng_key, shape=value.shape) * jnp.sqrt(variance) + value

    return random_argmax(rng_key, samples)

def tree_policy_puct(rng_key: chex.PRNGKey, tree: MctsTree, node_index: int, temperature: float):
    pb_c_init = 1.25
    pb_c_base = 19652.0

    visit_counts = tree.children_visits[node_index]
    node_visit = tree.node_visits[node_index]
    pb_c = pb_c_init + jnp.log((node_visit + pb_c_base + 1.) / pb_c_base)

    prior_logits = tree.node_prior_logits[node_index]
    prior_probs = jax.nn.softmax(prior_logits)
    policy_score = jnp.sqrt(node_visit) * pb_c * prior_probs / (visit_counts + 1)
    chex.assert_shape([node_index, node_visit], ())
    chex.assert_equal_shape([prior_probs, visit_counts, policy_score])

    value_score = qtransform_by_parent_and_siblings(tree, node_index)

    puct_score = value_score + policy_score

    return random_argmax(rng_key, puct_score)

def qtransform_by_parent_and_siblings(
    tree: MctsTree,
    node_index: chex.Numeric,
    *,
    epsilon: chex.Numeric = 1e-8,
) -> chex.Array:
  """Returns qvalues normalized by min, max over V(node) and qvalues.

  Copied from mctx

  Args:
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the parent node.
    epsilon: the minimum denominator for the normalization.

  Returns:
    Q-values normalized to be from the [0, 1] interval. The unvisited actions
    will have zero Q-value. Shape `[num_actions]`.
  """
  chex.assert_shape(node_index, ())
  qvalues = tree.qvalues(node_index)
  visit_counts = tree.children_visits[node_index]
  node_value = tree.node_values[node_index]
  # safe_qvalues = jnp.where(visit_counts > 0, qvalues, node_value)
  chex.assert_equal_shape([qvalues, qvalues])
  min_value = jnp.minimum(node_value, jnp.min(qvalues, axis=-1))
  max_value = jnp.maximum(node_value, jnp.max(qvalues, axis=-1))

  # completed_by_min = jnp.where(visit_counts > 0, qvalues, min_value)
  completed_by_min = qvalues
  normalized = (completed_by_min - min_value) / (
      jnp.maximum(max_value - min_value, epsilon))
  chex.assert_equal_shape([normalized, qvalues])
  return normalized

def visit_action_selection(rng_key, search_tree, temperature, config):
    def zero_temp_policy(visits):
        # For temperature = 0, select the most visited action deterministically
        return jnp.argmax(visits)

    def nonzero_temp_policy(visits, temp, rng):
        temp = temp + 1e-8  # Ensure non-zero temperature
        # Compute the AlphaGo Zero-style policy based on visit counts
        # π(a|s₀) = N(s₀,a)^(1/τ) / Σᵦ N(s₀,b)^(1/τ)
        visits = jnp.maximum(visits, 0)  # ensure non-negative
        powered_visits = jnp.power(visits, 1.0 / temp)
        visit_policy = powered_visits / (jnp.sum(powered_visits) + 1e-8)

        # Safety check
        chex.assert_shape(visits, (config.num_actions,))
        chex.assert_shape(visit_policy, (config.num_actions,))

        return jax.random.categorical(rng, logits=jnp.log(visit_policy + 1e-8))

    # Use lax.cond to handle zero vs non-zero temperature cases
    root_visits = search_tree.children_visits[search_tree.ROOT_INDEX]
    action = jax.lax.cond(
        temperature == 0.0,
        lambda: zero_temp_policy(root_visits),
        lambda: nonzero_temp_policy(root_visits, temperature, rng_key)
    )

    return action

def _get_logits_from_probs(probs):
  tiny = jnp.finfo(probs.dtype).tiny
  return jnp.log(jnp.maximum(probs, tiny))

def _add_dirichlet_noise(rng_key, probs, *, dirichlet_alpha,
                         dirichlet_fraction):
  """Mixes the probs with Dirichlet noise."""
  chex.assert_rank(probs, 1)
  chex.assert_type([dirichlet_alpha, dirichlet_fraction], float)

  noise = jax.random.dirichlet(
      rng_key,
      alpha=jnp.full(probs.shape, fill_value=dirichlet_alpha),
      shape=())
  noisy_probs = (1 - dirichlet_fraction) * probs + dirichlet_fraction * noise
  return noisy_probs


def policy(
        params: TrainState,
        rng_key: chex.PRNGKey,
        root: RootFnOutput,
        recurrent_fn: RecurrentFn,
        tree_policy_fn: Callable,
        num_simulations: int,
        config: Config,
        temperature: float
) -> PolicyFnOutput:
    """Policy function for MCTS with priors and noise."""
    rng_key, dirichlet_rng_key, search_rng_key, action_rng_key = jax.random.split(rng_key, 4)

    dirichlet_fraction = 0.25
    dirichlet_alpha = 10 / config.num_actions

    # Adding Dirichlet noise.
    noisy_logits = _get_logits_from_probs(
      _add_dirichlet_noise(
          dirichlet_rng_key,
          jax.nn.softmax(jnp.squeeze(root.prior_logits, axis=0)),
          dirichlet_fraction=dirichlet_fraction,
          dirichlet_alpha=dirichlet_alpha))
    root = root.replace(prior_logits=noisy_logits)

    search_tree = search(
      params=params,
      rng_key=search_rng_key,
      root=root,
      recurrent_fn=recurrent_fn,
      tree_policy_fn=tree_policy_fn,
      num_simulations=num_simulations,
      config=config,
      temperature=temperature
    )

    # Sampling the proposed action proportionally to the visit counts.
    summary = search_tree.summary()
    action = visit_action_selection(action_rng_key, search_tree, temperature, config)

    return PolicyFnOutput(
        action=action,
        value=summary.value,
        variance=summary.variance,
        p=summary.p,
        tree=search_tree,
    )
