import io
import os
from functools import partial
from typing import Dict
from typing import Iterator, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pygraphviz
import seaborn as sns
from PIL import Image
from tetris_gymnasium.envs.tetris_fn import ACTION_ID_TO_NAME, get_observation
from tetris_gymnasium.functional.core import EnvConfig
from tetris_gymnasium.functional.tetrominoes import TETROMINOES


@partial(jax.jit, static_argnums=(1,2,))
def create_rgb_frames_array(
        boards: chex.Array,
        cell_size: int,
        batch_size: int = 64
) -> chex.Array:
    """
    Convert boards to RGB and resize in batches to reduce memory usage (avoids large spikes).

    Args:
        boards: Array of shape (steps, height, width) with values -1, 0, or 1
        cell_size: Size of each cell in pixels
        batch_size: Number of frames to process at once

    Returns:
        Array of shape (steps, 3, height*cell_size, width*cell_size) with RGB values
    """

    def create_batch_indices(total_size: int, batch_size: int) -> Iterator[Tuple[int, int]]:
        """Generate start and end indices for each batch."""
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            yield start, end

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def process_batch(
            batch: chex.Array,
            cell_size: int,
            target_height: int,
            target_width: int
    ) -> chex.Array:
        """Process a single batch of frames."""
        # Define colors as uint8 to save memory
        color_map = jnp.array([
            [255, 255, 255],  # -1 -> white
            [0, 0, 0],  # 0 -> black
            [128, 128, 128]  # 1 -> grey
        ], dtype=jnp.uint8)

        # Convert to indices and create RGB representation
        board_indices = (batch + 1).astype(jnp.int32)
        rgb_boards = jnp.take(color_map, board_indices, axis=0)

        # Calculate source coordinates once for the batch
        y_scale = batch.shape[-2] / target_height
        x_scale = batch.shape[-1] / target_width

        y_coords = jnp.floor(jnp.arange(target_height) * y_scale).astype(jnp.int32)
        x_coords = jnp.floor(jnp.arange(target_width) * x_scale).astype(jnp.int32)

        y_indices, x_indices = jnp.meshgrid(y_coords, x_coords, indexing='ij')

        # Resize using pre-computed indices
        resized = rgb_boards[..., y_indices, x_indices, :]

        # Return in the format expected by wandb
        return resized.transpose(0, 3, 1, 2)

    # Calculate target dimensions
    target_height = boards.shape[1] * cell_size  # Use actual height instead of fixed 20
    target_width = boards.shape[2] * cell_size   # Use actual width instead of fixed 10

    # Initialize output array
    total_frames = boards.shape[0]
    output_shape = (total_frames, 3, target_height, target_width)
    output_frames = jnp.zeros(output_shape, dtype=jnp.uint8)

    # Process in batches
    for start, end in create_batch_indices(total_frames, batch_size):
        # Select batch without assuming batch dimension
        batch_frames = boards[start:end]
        # Process batch and update output
        output_frames = output_frames.at[start:end].set(
            process_batch(batch_frames, cell_size, target_height, target_width)
        )

    return output_frames

def board_to_png(board: chex.Array, cell_size: int = 15) -> Image:
    """
    Convert a Tetris board array to a PNG image using JAX operations.

    Args:
        board: A 20x10 array representing the Tetris board where:
              0 = empty (black)
              1 = filled (grey)
              -1 = falling piece (white)
        cell_size: Size of each cell in pixels (default: 15)

    Returns:
        PIL Image object of the rendered Tetris board
    """
    # Create RGB color mapping array (board_value -> RGB)
    # Convert indices to color values
    color_map = jnp.stack([jnp.array([255, 255, 255]), jnp.array([0, 0, 0]), jnp.array([128, 128, 128])])
    board_indices = (board + 1).astype(jnp.int32)  # Shift -1,0,1 to 0,1,2
    rgb_board = color_map[board_indices]

    # Convert to int8 and numpy for PIL
    rgb_board = np.array(rgb_board).astype(jnp.uint8)

    # Create and resize image
    image = Image.fromarray(rgb_board)
    if cell_size > 1:
        image = image.resize(
            (board.shape[1] * cell_size, board.shape[0] * cell_size),
            Image.Resampling.NEAREST
        )

    return image

def save_graph_in_memory(graph):
    # Create an in-memory buffer
    buffer = io.BytesIO()

    # Method 1: Using pygraphviz to write directly to buffer
    graph.draw(buffer, format='png', prog='dot')
    buffer.seek(0)

    # Create PIL Image from buffer
    image = Image.open(buffer)

    # Clean up
    # buffer.close()
    return image, buffer

def convert_mcts_tree_to_graph(
        tree,
        env_config: EnvConfig,
        action_mapping: Dict[int, str] = ACTION_ID_TO_NAME,
        batch_index: int = 0
) -> pygraphviz.AGraph:
    """Converts a MCTS tree into a Graphviz graph with compact Tetris board visualizations."""
    num_actions = tree.children_indices.shape[-1]

    if len(action_mapping) != num_actions:
        raise ValueError(
            f"action_mapping has {len(action_mapping)} actions, but tree expects {num_actions} actions. "
            "Please provide mapping for all actions.")

    def node_to_str(node_i: int, obs: chex.Array) -> str:
        is_terminal = tree.node_terminals[batch_index, node_i]
        terminal_str = " (T)" if is_terminal else ""

        base = "/tmp/my-checkpoints/graphs/"
        os.makedirs(base, exist_ok=True)
        board_png_path = f'/tmp/my-checkpoints/graphs/board_{node_i}.png'
        # del prev existing image
        if os.path.exists(board_png_path):
            os.remove(board_png_path)
        board_to_png(obs[node_i]).save(board_png_path)

        # Create HTML-style label
        label = (
            '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="1">'
            f'<TR><TD ALIGN="left">Node {node_i}{terminal_str}</TD></TR>'
            f'<TR><TD><IMG SRC="{board_png_path}"/></TD></TR>'
            f'<TR><TD ALIGN="left">V:{tree.node_values[batch_index, node_i]:.2f} '
            f'Var:{tree.node_variances[batch_index, node_i]:.1f}</TD></TR>'
            f'<TR><TD ALIGN="left">N:{tree.node_visits[batch_index, node_i]}</TD></TR>'
            f'<TR><TD ALIGN="left">S:{tree.node_scores[batch_index, node_i]}</TD></TR>'
            '</TABLE>>'
        )
        return label

    def edge_to_str(node_i: int, action_i: int, child_i: int) -> str:
        return action_mapping[action_i]


    graph = pygraphviz.AGraph(directed=True, strict=True)

    get_batched_observation = jax.vmap(
        get_observation,
        in_axes=(0, 0, 0, 0, 0, 0, None, None),  # batch everything except tetrominoes and config
        out_axes=(0)
    )

    state = tree.node_state
    observations = get_batched_observation(
        state.board[batch_index], state.x[batch_index], state.y[batch_index], state.active_tetromino[batch_index],
        state.rotation[batch_index], state.game_over[batch_index], TETROMINOES, env_config
    )

    # Configure graph attributes for compact layout
    graph.graph_attr.update({
        'rankdir': 'TB',
        'nodesep': '0.3',
        'ranksep': '0.4',
        'splines': 'ortho',
        'concentrate': 'true'
    })

    # Add root node
    graph.add_node(tree.ROOT_INDEX,
                   label=node_to_str(tree.ROOT_INDEX, observations),
                   shape='box',
                   style='rounded',
                   color="green",
                   fontname="Courier")

    # Add all other nodes and connect them
    for node_i in range(len(tree.node_values[batch_index])):
        for action_i in range(num_actions):
            child_i = tree.children_indices[batch_index, node_i, action_i]

            if child_i == tree.UNVISITED:
                continue

            node_color = "red" if tree.node_terminals[batch_index, child_i] else "black"
            graph.add_node(child_i,
                           label=node_to_str(child_i, observations),
                           shape='box',
                           style='rounded',
                           color=node_color,
                           fontname="Courier")

            graph.add_edge(node_i,
                           child_i,
                           xlabel=edge_to_str(node_i, action_i, child_i),
                           fontname="Courier")

    return graph
