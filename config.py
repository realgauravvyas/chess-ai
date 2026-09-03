"""All hyperparameters for the chess project in one place."""
from dataclasses import dataclass
from move_encoding import NUM_POLICY


@dataclass
class Config:
    # --- model ---------------------------------------------------------
    planes: int = 18              # board input planes (Lichess encoding)
    filters: int = 64             # conv filters
    res_blocks: int = 10          # number of residual blocks
    policy_size: int = NUM_POLICY  # 64 * 73 = 4672 actions

    # --- MCTS ----------------------------------------------------------
    num_simulations: int = 40     # simulations per move during self-play
    c_puct: float = 1.5           # exploration constant
    repetition_aware: bool = False  # let MCTS see repetitions (see RESULTS.md)
    dirichlet_alpha: float = 0.3  # root noise concentration
    dirichlet_epsilon: float = 0.25  # root noise weight

    # --- self-play -----------------------------------------------------
    games_per_iteration: int = 12  # games generated per training iteration
    temperature_moves: int = 20   # use temperature for the first N moves, then argmax
    max_game_length: int = 400    # forced draw cap (plies), prevents infinite games

    # --- training ------------------------------------------------------
    batch_size: int = 128
    train_steps: int = 200        # gradient steps per iteration
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    lr_min: float = 1e-4          # cosine annealing floor (never decay to 0)
    anchor_fraction: float = 0.30  # supervised rehearsal slice per minibatch
    replay_buffer_size: int = 100_000

    # --- outer loop ----------------------------------------------------
    num_iterations: int = 200
    checkpoint_interval: int = 5  # save + evaluate every N iterations
    eval_games: int = 10          # games in evaluation vs baseline

    # --- curriculum ------------------------------------------------------
    opening_curriculum: bool = True   # start self-play games from rare human positions
    opening_seeds_file: str = "data/opening_seeds.json"
