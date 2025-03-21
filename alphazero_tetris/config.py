from typing import Optional

import chex

from tabulate import tabulate
def print_config(config):
    params = [
        ["MCTS", "Simulations", config.num_simulations],
        ["Collection", "Total epochs", config.train_epochs],
        ["Collection", "Samples / epoch", config.collect_samples_per_epoch],
        ["Training", "Batch size", config.train_batch_size],
        ["Training", "Patience interval", config.training_patience_interval],
    ]
    print(tabulate(params, headers=["Category", "Parameter", "Value"],
                   tablefmt="grid"))


@chex.dataclass(frozen=True)
class Config:
    # General
    seed: int = 42
    project_name: str = "alphazero-tetris"
    obs_size: tuple = (20, 10)
    checkpoint_dir: str = "/tmp/alphazero-tetris/checkpoints"
    checkpoint_fraction: float = 0.1
    # MCTS
    num_actions: int = 7
    num_simulations: int = int(300)
    min_visit_count: int = 1
    discount: float = 0.999
    # Data collection
    collect_samples_per_epoch: int = int(1e4)
    temperature: bool = True
    # Training
    train_epochs: int = int(100)
    train_max_steps: int = int(5e4)
    train_batch_size: int = int(512)
    train_eval_fraction: float = 0.2 # how much of the collected data is used for training / evaluation
    training_patience_interval: int = 100
    # Evaluation
    wandb_eval_fraction: float = 0.05
    wandb_eval_max_steps: int = 1000
    # Logging
    wandb_run_name: Optional[str] = None # to sync a run, you need to specify a name
    wandb_log_dir: str = "runs"
