"""Self-play: the network plays games against itself and produces training data.

Supports parallel game generation across multiple worker processes.
Planes are stored as uint8 to keep the replay buffer memory-friendly.
"""
import multiprocessing as mp
from pathlib import Path

import chess
import numpy as np
import torch

from config import Config
from mcts import MCTS
from model import AlphaZeroNet
from move_encoding import encode_move
from utils import board_to_planes, result_to_z

ROOT = Path(__file__).resolve().parent

_WORKER_NET = None
_WORKER_DEVICE = None
_WORKER_CFG = None
_SEEDS = None


def _load_seeds(cfg):
    """Lazily load human-curriculum starting positions (FENs)."""
    global _SEEDS
    if _SEEDS is not None or not cfg.opening_curriculum:
        return _SEEDS
    path = ROOT / cfg.opening_seeds_file
    if path.exists():
        try:
            import json
            data = json.loads(path.read_text(encoding="utf-8"))
            seeds = [f for f in data.get("seeds", [])
                     if chess.Board(f).status() in
                     (chess.STATUS_VALID,)]
            if seeds:
                print(f"[curriculum] loaded {len(seeds)} opening seed positions")
                _SEEDS = seeds
        except Exception as exc:  # noqa: BLE001
            print(f"[curriculum] seeds unavailable: {exc}")
            _SEEDS = []
    return _SEEDS


def _worker_init(device_str, cfg):
    """Initializer run once in each worker process."""
    global _WORKER_NET, _WORKER_DEVICE, _WORKER_CFG
    _WORKER_DEVICE = torch.device(device_str)
    _WORKER_NET = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                               action_planes=cfg.policy_size // 64).to(_WORKER_DEVICE)
    _WORKER_NET.eval()
    _WORKER_CFG = cfg
    torch.set_num_threads(1)


def _worker_play_one(state_dict):
    """Play one game with the given (latest) network weights."""
    _WORKER_NET.load_state_dict(state_dict)
    return play_one_game(_WORKER_NET, _WORKER_DEVICE, _WORKER_CFG)


def play_one_game(net, device, cfg: Config):
    """Play one self-play game.

    Returns training samples (planes_u8, indices, probs, z). The policy
    target is always the full MCTS visit distribution; temperature only
    affects which move is actually played.
    """
    board = chess.Board()
    seeds = _load_seeds(cfg)
    if seeds and np.random.random() < 0.85:
        try:
            board = chess.Board(seeds[np.random.randint(len(seeds))])
        except Exception:  # noqa: BLE001
            board = chess.Board()
    mcts = MCTS(net, device, cfg.c_puct, cfg.dirichlet_alpha, cfg.dirichlet_epsilon)
    positions = []
    plies = 0

    while not board.is_game_over() and plies < cfg.max_game_length:
        probs = mcts.get_action_probs(board, cfg.num_simulations)
        if not probs:
            break

        moves = list(probs.keys())
        if plies < cfg.temperature_moves:
            p = np.array([probs[m] for m in moves], dtype=np.float64)
            move = moves[np.random.choice(len(moves), p=p / p.sum())]
        else:
            move = max(moves, key=probs.get)

        planes = board_to_planes(board).astype(np.uint8)
        indices = [encode_move(m) for m in moves]
        probs_array = np.array([probs[m] for m in moves], dtype=np.float32)
        probs_array = probs_array / probs_array.sum()
        positions.append((planes, indices, probs_array, board.turn))

        board.push(move)
        plies += 1

    result = board.result() if board.is_game_over() else "1/2-1/2"
    z = result_to_z(result, chess.WHITE)

    samples = []
    for planes, indices, probs_array, turn in positions:
        z_sample = z if turn == chess.WHITE else -z
        samples.append((planes, indices, probs_array, z_sample))
    return samples


def generate_games(net, device, cfg: Config, num_games, pool=None):
    """Generate `num_games` self-play games, flattened into training samples.

    If `pool` is given, games run in CPU worker processes; the main process's
    `device` (possibly cuda) is only used for the batched training math.
    """
    net.eval()
    if pool is None:
        play_device = torch.device("cpu")  # fastest for single-position evals
        samples = []
        for _ in range(num_games):
            samples.extend(play_one_game(net, play_device, cfg))
        return samples

    state = {k: v.cpu() for k, v in net.state_dict().items()}
    results = pool.map(_worker_play_one, [state] * num_games)
    samples = []
    for game in results:
        samples.extend(game)
    return samples
