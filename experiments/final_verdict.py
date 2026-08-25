"""The decisive measurement: did gated self-play actually improve the model?

The per-gate matches during training use 8 games, which carries a standard
error of ~17.7 percentage points - too coarse to distinguish a real effect
from noise. This runs a long match instead, in parallel across CPU workers,
and reports a confidence interval so the answer is interpretable.

    python experiments/final_verdict.py --games 40

Compares, from each model's own perspective:
    best.pt (gated winner of the corrected run)  vs  frozen pretrained net
and, for context, the same models against a random mover.
"""
import argparse
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import chess
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config                    # noqa: E402
from evaluate import net_points              # noqa: E402
from mcts import MCTS                        # noqa: E402
from model import AlphaZeroNet               # noqa: E402
from utils import load_checkpoint            # noqa: E402

cfg = Config()
_A = _B = None


def _load(path):
    net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                       cfg.policy_size // 64)
    blob = load_checkpoint(str(path))
    net.load_state_dict(blob["model_state_dict"] if "model_state_dict" in blob
                        else blob)
    net.eval()
    return net


def _init(pa, pb):
    global _A, _B
    torch.set_num_threads(1)
    _A = _load(pa)
    _B = _load(pb) if pb else None


def _match_game(job):
    """One game of A vs B. Returns A's points."""
    gi, sims, open_plies = job
    random.seed(50_000 + gi)
    a_white = (gi % 2 == 0)
    board = chess.Board()
    for _ in range(open_plies):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    nets = {chess.WHITE: _A if a_white else _B,
            chess.BLACK: _B if a_white else _A}
    searchers = {c: MCTS(n, torch.device("cpu"), cfg.c_puct,
                         dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
                 for c, n in nets.items()}
    while not board.is_game_over(claim_draw=True) and board.ply() < 300:
        probs = searchers[board.turn].get_action_probs(board, sims)
        if not probs:
            break
        board.push(max(probs, key=probs.get))
    over = board.is_game_over(claim_draw=True)
    return net_points(board.result(claim_draw=True) if over else "1/2-1/2",
                      a_white)


def _random_game(job):
    """One game of A vs a uniformly random mover. Returns A's points."""
    gi, sims, _ = job
    random.seed(70_000 + gi)
    a_white = (gi % 2 == 0)
    board = chess.Board()
    mcts = MCTS(_A, torch.device("cpu"), cfg.c_puct, dirichlet_alpha=0.0)
    while not board.is_game_over(claim_draw=True) and board.ply() < 300:
        if (board.turn == chess.WHITE) == a_white:
            probs = mcts.get_action_probs(board, sims)
            if not probs:
                break
            board.push(max(probs, key=probs.get))
        else:
            board.push(random.choice(list(board.legal_moves)))
    over = board.is_game_over(claim_draw=True)
    return net_points(board.result(claim_draw=True) if over else "1/2-1/2",
                      a_white)


def report(label, pts, games):
    w = sum(1 for p in pts if p == 1.0)
    d = sum(1 for p in pts if p == 0.5)
    n = sum(1 for p in pts if p == 0.0)
    score = sum(pts) / games
    # standard error of the mean of per-game points
    mean = score
    var = sum((p - mean) ** 2 for p in pts) / max(1, games - 1)
    se = math.sqrt(var / games)
    lo, hi = (mean - 1.96 * se) * 100, (mean + 1.96 * se) * 100
    z = (mean - 0.5) / se if se > 0 else 0.0
    verdict = ("STRONGER" if z > 1.96 else
               "WEAKER" if z < -1.96 else
               "no detectable difference")
    print(f"  {label}")
    print(f"    +{w} ={d} -{n}   score {score:.1%}   "
          f"95% CI [{lo:.1f}%, {hi:.1f}%]")
    print(f"    z = {z:+.2f}  ->  {verdict}")
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=80)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--open-plies", type=int, default=4)
    ap.add_argument("--best", default=str(ROOT / "runs" / "v7" / "checkpoints" / "best.pt"))
    ap.add_argument("--baseline",
                    default=str(ROOT / "checkpoints_v7_baseline" / "iter_200.pt"))
    args = ap.parse_args()

    best, base = Path(args.best), Path(args.baseline)
    if not best.exists():
        sys.exit(f"no gated model at {best} - was --gate used?")
    if not base.exists():
        sys.exit(f"no baseline at {base}")

    print(f"Decisive evaluation: {args.games} games @ {args.sims} sims, "
          f"{args.open_plies} random opening plies")
    print(f"  A = {best}")
    print(f"  B = {base}\n")

    jobs = [(g, args.sims, args.open_plies) for g in range(args.games)]

    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(str(best), str(base))) as pool:
        pts = pool.map(_match_game, jobs)
    print("HEAD TO HEAD (this is the answer)")
    report("gated best.pt  vs  frozen pretrained baseline", pts, args.games)
    print(f"    [{time.time() - t0:.0f}s]\n")

    print("CONTEXT: each model against a random mover")
    for label, path in (("gated best.pt", best), ("frozen baseline", base)):
        with mp.Pool(args.workers, initializer=_init,
                     initargs=(str(path), None)) as pool:
            rpts = pool.map(_random_game, jobs)
        report(f"{label}  vs  random", rpts, args.games)


if __name__ == "__main__":
    main()
