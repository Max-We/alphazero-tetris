import functools
from typing import ClassVar, Any

import chex
import jax
import jax.numpy as jnp
from flax.nnx import TrainState

from alphazero_tetris.base import RecurrentFn, batch_update, update, RootFnOutput, SearchSummary, Config, get_random_index, \
    MctsTree

from tetris_gymnasium.functional.core import State

from alphazero_tetris.debug import save_tree_image


@chex.dataclass(frozen=True)
class MctsState:
    rng_key: chex.PRNGKey
    node_index: chex.Numeric
    node_action: chex.Numeric
    next_node_index: chex.Numeric
    is_continuing: chex.Numeric
    root_action: chex.Numeric

def init_tree(config: Config, root: RootFnOutput):
    num_nodes = (config.num_simulations + 1) * config.num_actions

    def _zeros(x):
        return jnp.zeros((num_nodes,) + x.shape[1:], dtype=x.dtype)

    tree = MctsTree(
        node_state=jax.tree_map(lambda x: jnp.broadcast_to(x, (num_nodes,) + x.shape), root.state),
        node_values=jnp.zeros(num_nodes),
        node_visits=jnp.zeros(num_nodes),
        node_variances=jnp.zeros(num_nodes),
        node_prior_logits=jnp.zeros((num_nodes, config.num_actions)),
        node_scores=jnp.zeros(num_nodes),
        node_terminals=jnp.zeros(num_nodes),
        node_observations=jnp.zeros((num_nodes,) + root.observation.shape),

        children_indices=jnp.full((num_nodes, config.num_actions), MctsTree.UNVISITED),
        children_values=jnp.zeros((num_nodes, config.num_actions)),
        children_visits=jnp.zeros((num_nodes, config.num_actions)),
        children_variances=jnp.zeros((num_nodes, config.num_actions)),
        children_rewards=jnp.zeros((num_nodes, config.num_actions)),
        children_scores=jnp.zeros((num_nodes, config.num_actions)),
        children_discounts=jnp.zeros((num_nodes, config.num_actions)),

        parents=jnp.full(num_nodes, MctsTree.NO_PARENT),
        action_from_parent=jnp.full(num_nodes, MctsTree.NO_PARENT),
    )

    return tree

def instantiate_tree_from_root(config: Config, root: RootFnOutput):
    tree = init_tree(config, root)
    root_index = MctsTree.ROOT_INDEX
    tree = update_node(tree, root_index, root.state, jnp.squeeze(root.value), jnp.squeeze(root.variance), jnp.squeeze(root.prior_logits), jnp.squeeze(root.state.score), root.observation, -1)
    return tree

# def expand_and_backward(tree_init, sim_i, parent_index, params, recurrent_fn, config, root_action):
#     # sim_i + 1
#     children_indices = jnp.arange(config.num_actions) + 1 + sim_i * config.num_actions
#     action_indices = jnp.arange(config.num_actions)
#
#     def body_fn(i, tree):
#         child_index = children_indices[i]
#         tree = backward(tree, child_index)
#         return tree
#
#     tree_expanded = expand(tree_init, parent_index, children_indices, recurrent_fn, params, action_indices, root_action)
#     tree_backwarded = jax.lax.fori_loop(0, config.num_actions, body_fn, tree_expanded)
#     return tree_backwarded

def expand_and_backward(tree_init, sim_i, parent_index, params, recurrent_fn, config, root_action):
    def average_children(tree, parent_index):
        values = tree.children_values[parent_index]
        variances = tree.children_variances[parent_index]
        discounts = tree.children_discounts[parent_index]
        rewards = tree.children_rewards[parent_index]

        parent_value = tree.node_values[parent_index]
        parent_variance = tree.node_variances[parent_index]
        count = tree.node_visits[parent_index]

        avg_value = jnp.mean(rewards + values * discounts)
        val_new, var_new = welfords_update(avg_value, parent_value, parent_variance, count)

        return tree.replace(
            node_values=tree.node_values.at[parent_index].set(val_new),
            node_variances=tree.node_variances.at[parent_index].set(var_new),
            node_visits=tree.node_visits.at[parent_index].set(count + 1),
        )

    children_indices = jnp.arange(config.num_actions) + 1 + sim_i * config.num_actions
    action_indices = jnp.arange(config.num_actions)

    tree_expanded = expand(tree_init, parent_index, children_indices, recurrent_fn, params, action_indices, root_action)
    tree_averaged = average_children(tree_expanded, parent_index)
    tree_backward = backward(tree_averaged, parent_index)

    return tree_backward

def search(
        params: TrainState,
        rng_key: chex.PRNGKey,
        root: RootFnOutput,
        recurrent_fn: RecurrentFn,
        tree_policy_fn,
        num_simulations: int,
        config: Config,
        temperature: float
) -> MctsTree:

    def body_fn(sim_i, loop_state):
        iter_rng_key, tree = loop_state

        iter_rng_key, select_rng_key, expand_rng_key = jax.random.split(iter_rng_key, 3)

        root_action, parent_index, action_index = select(select_rng_key, tree, config, tree_policy_fn, temperature)
        child_index = tree.children_indices[parent_index, action_index]

        tree = jax.lax.cond(
            tree.node_terminals[child_index] == 1,
            lambda t, s, p, c: backward(t, c),
            lambda t, s, p, c: expand_and_backward(t, s, c, params, recurrent_fn, config, root_action),
            tree, sim_i, parent_index, child_index
        )

        return iter_rng_key, tree

    tree = instantiate_tree_from_root(config, root)
    tree = expand_and_backward(tree, 0, MctsTree.ROOT_INDEX, params, recurrent_fn, config, 0)

    # TODO: could stop early if root is visited > num_simulations//2 (but only for inference)
    _, tree = jax.lax.fori_loop(1, num_simulations, body_fn, (rng_key, tree))

    return tree

def select(rng_key: chex.PRNGKey, tree: MctsTree, config: Config, tree_policy, temperature: float) -> chex.Numeric:
    def cond_fn(state: MctsState) -> bool:
        return state.is_continuing

    def body_fn(state: MctsState) -> MctsState:
        current_node_index = state.next_node_index
        current_node_is_root = current_node_index == MctsTree.ROOT_INDEX

        rng_key, rng_key_select = jax.random.split(state.rng_key)
        action_index = tree_policy(rng_key_select, tree, current_node_index, temperature)

        new_root_action = jnp.where(current_node_is_root, action_index, state.root_action)
        next_node_index = tree.children_indices[current_node_index, action_index]
        next_is_terminal = tree.node_terminals[next_node_index] == 1
        next_needs_child_expand = tree.node_visits[next_node_index] <= 1

        return MctsState(
            rng_key=rng_key,
            root_action=new_root_action,
            node_index=current_node_index,
            node_action=action_index,
            next_node_index=next_node_index,
            is_continuing=jnp.logical_and(~next_needs_child_expand, ~next_is_terminal)
        )

    node_index = jnp.array(MctsTree.ROOT_INDEX, dtype=jnp.int32)
    initial_state = MctsState(
        rng_key=rng_key,
        root_action=-1,
        node_index=MctsTree.NO_PARENT,
        node_action=MctsTree.NO_PARENT,
        next_node_index=node_index,
        is_continuing=jnp.array(True),
    )

    end_state: MctsState = jax.lax.while_loop(cond_fn, body_fn, initial_state)

    return end_state.root_action, end_state.node_index, end_state.node_action

def expand(
        tree: MctsTree,
        node_index: chex.Array,
        children_indices: chex.Array,
        recurrent_fn: RecurrentFn,
        params: TrainState,
        actions: chex.Array,
        root_action: chex.Array
):
    num_actions = actions.shape[-1]

    states = jax.tree.map(lambda x: jnp.squeeze(jnp.repeat(x[None, node_index], repeats=num_actions, axis=0)), tree.node_state)
    steps, states = recurrent_fn(params, actions, states)

    children_visits = jnp.ones(num_actions, dtype=jnp.int32)
    def update_child(i, iter_tree):
        c_state = jax.tree.map(lambda x: x[i], states)

        iter_tree = update_node(iter_tree, children_indices[i], c_state, steps.value[i], steps.variance[i], steps.p[i], steps.discount[i], steps.observation[i], root_action)
        return iter_tree.replace(
            children_indices=iter_tree.children_indices.at[node_index, i].set(children_indices[i]),
            children_values=iter_tree.children_values.at[node_index, i].set(steps.value[i]),
            children_variances=iter_tree.children_variances.at[node_index, i].set(steps.variance[i]),
            children_visits=iter_tree.children_visits.at[node_index, i].set(children_visits[i]),
            children_rewards=iter_tree.children_rewards.at[node_index, i].set(steps.reward[i]),
            children_scores=iter_tree.children_scores.at[node_index, i].set(steps.score[i]),
            children_discounts=iter_tree.children_discounts.at[node_index, i].set(steps.discount[i]),
            parents=iter_tree.parents.at[children_indices[i]].set(node_index),
            action_from_parent=iter_tree.action_from_parent.at[children_indices[i]].set(i)
        )

    tree = jax.lax.fori_loop(0, num_actions, update_child, tree)
    return tree

def welfords_update(val, val_old, var_old, n):
    # Welford algorithm for calculating mean & variance (for the parent)

    # 1. mean update & delta calculation
    delta_old = val - val_old
    val_new = val_old + delta_old / (n + 1.0)
    delta_new = val - val_new

    # 2. variance update
    # s_old = var_leaf * count
    # s_new = s_old + delta_old * delta_new
    # var_leaf = s_new / (count + 1.0)
    # var_leaf = ((var_leaf * count) + delta_old * delta_new) / (count + 1.0) # 1-line version
    var_new = var_old + (delta_old * delta_new - var_old) / (n + 1.0)  # 1-line version (alternative)

    return val_new, var_new

def backward(tree: MctsTree, leaf_index: chex.Numeric):
    def cond_fun(loop_state):
        _, _, index = loop_state
        return index != MctsTree.ROOT_INDEX

    def body_fun(loop_state):
        tree, val_leaf, index = loop_state
        parent = tree.parents[index]
        count = tree.node_visits[parent]
        action = tree.action_from_parent[index]
        discount = tree.children_discounts[parent, action]
        reward = tree.children_rewards[parent, action]

        val_leaf = discount * val_leaf + reward

        val_new, var_new = welfords_update(val_leaf, tree.node_values[parent], tree.node_variances[parent], count)

        child_value = tree.node_values[index]
        child_variance = tree.node_variances[index]
        child_visits = tree.children_visits[parent, action] + 1

        tree = tree.replace(
            node_values=tree.node_values.at[parent].set(val_new),
            node_variances=tree.node_variances.at[parent].set(var_new),
            node_visits=tree.node_visits.at[parent].set(count + 1),
            children_values=tree.children_values.at[parent, action].set(child_value),
            children_variances=tree.children_variances.at[parent, action].set(child_variance),
            children_visits=tree.children_visits.at[parent, action].set(child_visits)
        )

        return tree, val_leaf, parent

    leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
    val_init = tree.node_values[leaf_index]

    loop_state = (tree, val_init, leaf_index)
    tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state)

    return tree

def update_node(tree: MctsTree, node_index: chex.Array, state: State, value: chex.Array, variance: chex.Array, prior_logits: chex.Array, score: chex.Array, observation: chex.Array, root_action: chex.Numeric):
    new_visit = tree.node_visits[node_index] + 1
    updates = dict(
        node_variances=tree.node_variances.at[node_index].set(variance),
        node_scores=tree.node_scores.at[node_index].set(score),
        node_values=tree.node_values.at[node_index].set(value),
        node_prior_logits=tree.node_prior_logits.at[node_index].set(prior_logits),
        node_terminals=tree.node_terminals.at[node_index].set(state.game_over),
        node_visits=tree.node_visits.at[node_index].set(new_visit),
        node_state=jax.tree.map(lambda t, s: t.at[node_index].set(s), tree.node_state, state),
        node_observations = tree.node_observations.at[node_index].set(observation),  # Update node_observations
    )

    return tree.replace(**updates)
