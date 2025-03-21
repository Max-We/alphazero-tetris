import io
import os
import time

from PIL import Image
from tetris_gymnasium.functional.core import EnvConfig

from alphazero_tetris.visualize import convert_mcts_tree_to_graph


def save_tree_image(tree, i=0):
    """This may be used inside the search function of mcts + a debug-callback to save the tree as image."""
    t = time.time()
    env_config = EnvConfig(width=10, height=20, padding=10, queue_size=7)
    graph = convert_mcts_tree_to_graph(tree, env_config, batch_index=0)
    buffer = io.BytesIO()
    graph.draw(buffer, format='png', prog='dot')
    buffer.seek(0)

    image = Image.open(buffer)
    # create dir
    os.makedirs('/tmp/alphazero-tetris', exist_ok=True)
    filename = f'/tmp/alphazero-tetris/tree_{t}_{i}.png'
    filepath = os.path.abspath(filename)
    image.save(filename)
    buffer.close()

    print(f"Saved tree image to: {filepath}")
