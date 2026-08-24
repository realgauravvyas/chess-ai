"""Does MCTS actually beat the network's own raw policy, and by how much?

Self-play RL can only improve a network if the search targets it trains on
are better than what the network already predicts. This measures exactly
that: MCTS(N simulations) played against the same network's greedy policy
head, as N varies.

Usage:
    python experiments/sims_scaling.py --games 12
"""
import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import chess
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config                          # noqa: E402
from evaluate import greedy_policy_move, net_points  # noqa: E402
from mcts import MCTS                              # noqa: E402
from model import AlphaZeroNet                     # noqa: E402
from utils import load_checkpoint                  # noqa: E402

cfg = Config()
_net = None
_ckpt = None


def _init(ckpt):
    global _net
    torch.set_num_threads(1)
    _net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                        cfg.policy_size // 64)
    _net.load_state_dict(load_checkpoint(ckpt)["model_state_dict"])
    _net.eval()


def _game(job):
    sims, gi = job
    random.seed(9000 + gi)
    mcts_white = (gi % 2 == 0)
    board = chess.Board()
    for _ in range(4):          # randomised opening: otherwise every game is identical
        board.push(random.choice(list(board.legal_moves)))
    dev = torch.device("cpu")
    mcts = MCTS(_net, dev, cfg.c_puct, dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    while not board.is_game_over(claim_draw=True) and board.ply() < 240:
        if (board.turn == chess.WHITE) == mcts_white:
            probs = mcts.get_action_probs(board, sims)
            if not probs:
                break
            board.push(max(probs, key=probs.get))
        else:
            board.push(greedy_policy_move(_net, dev, board))
    over = board.is_game_over(claim_draw=True)
    return net_points(board.result(claim_draw=True) if over else "1/2-1/2",
                      mcts_white)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "iter_200.pt"))
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sims", type=int, nargs="*", default=[10, 25, 40, 100, 200])
    args = ap.parse_args()

    print(f"{Path(args.checkpoint).name}: MCTS(N) vs its own greedy policy, "
          f"{args.games} games each")
    print(f"{'sims':>6} {'score':>8}  {'W-D-L':>10}  {'sec':>6}")
    with mp.Pool(args.workers, initializer=_init, initargs=(args.checkpoint,)) as pool:
        for sims in args.sims:
            t0 = time.time()
            pts = pool.map(_game, [(sims, g) for g in range(args.games)])
            w = sum(1 for p in pts if p == 1.0)
            d = sum(1 for p in pts if p == 0.5)
            n = sum(1 for p in pts if p == 0.0)
            print(f"{sims:6d} {sum(pts)/args.games:7.1%}  {w:3d}-{d:2d}-{n:2d}  "
                  f"{time.time()-t0:6.0f}", flush=True)


if __name__ == "__main__":
    main()
