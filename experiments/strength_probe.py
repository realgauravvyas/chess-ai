"""Correctly-scored strength measurement for trained checkpoints.

Everything here scores from the NETWORK's point of view. The original
evaluation code scored from white's, while the network alternated colors,
which made every reported number meaningless (see RESULTS.md).

Usage:
    python experiments/strength_probe.py                 # all probes
    python experiments/strength_probe.py --games 20      # longer matches
"""
import argparse
import random
import sys
import time
from pathlib import Path

import chess
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config                      # noqa: E402
from evaluate import net_points                # noqa: E402
from mcts import MCTS                          # noqa: E402
from model import AlphaZeroNet                 # noqa: E402
from move_encoding import encode_move          # noqa: E402
from utils import board_to_planes, load_checkpoint  # noqa: E402

cfg = Config()

# Positions with a single unambiguous best move.
TACTICS = [
    ("free queen", "rnb1kbnr/ppp1pppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1", "exd5"),
    ("mate in 1", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "Ra8#"),
    ("free rook", "4k3/8/8/3r4/4P3/8/8/4K3 w - - 0 1", "exd5"),
]


def load(path):
    net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                       cfg.policy_size // 64)
    blob = load_checkpoint(str(path))
    net.load_state_dict(blob["model_state_dict"] if "model_state_dict" in blob
                        else blob)
    net.eval()
    return net


@torch.no_grad()
def top_policy(net, board, k=5):
    """The raw policy head's favourite moves, with no search at all."""
    logits, value = net(torch.from_numpy(board_to_planes(board)).unsqueeze(0))
    probs = torch.softmax(logits[0], 0).numpy()
    best = sorted(board.legal_moves, key=lambda m: -probs[encode_move(m)])[:k]
    return [(board.san(m), round(float(probs[encode_move(m)]), 3)) for m in best], \
        float(value.item())


def vs_random(net, games, sims, seed=0):
    """Score vs a uniformly random mover, from the network's perspective."""
    random.seed(seed)
    w = d = losses = 0
    for g in range(games):
        net_white = (g % 2 == 0)
        board = chess.Board()
        mcts = MCTS(net, torch.device("cpu"), cfg.c_puct, dirichlet_alpha=0.0)
        while not board.is_game_over(claim_draw=True) and board.ply() < 200:
            if (board.turn == chess.WHITE) == net_white:
                probs = mcts.get_action_probs(board, sims)
                if not probs:
                    break
                board.push(max(probs, key=probs.get))
            else:
                board.push(random.choice(list(board.legal_moves)))
        over = board.is_game_over(claim_draw=True)
        pts = net_points(board.result(claim_draw=True) if over else "1/2-1/2",
                         net_white)
        w, d, losses = (w + (pts == 1.0), d + (pts == 0.5), losses + (pts == 0.0))
    return w, d, losses


def head_to_head(a, b, games, sims, open_plies=4):
    """Match between two nets. Randomised openings, else every game is identical."""
    score = 0.0
    for g in range(games):
        a_white = (g % 2 == 0)
        board = chess.Board()
        random.seed(1000 + g)
        for _ in range(open_plies):
            board.push(random.choice(list(board.legal_moves)))
        nets = {chess.WHITE: a if a_white else b,
                chess.BLACK: b if a_white else a}
        while not board.is_game_over(claim_draw=True) and board.ply() < 240:
            mcts = MCTS(nets[board.turn], torch.device("cpu"), cfg.c_puct,
                        dirichlet_alpha=0.0)
            probs = mcts.get_action_probs(board, sims)
            if not probs:
                break
            board.push(max(probs, key=probs.get))
        over = board.is_game_over(claim_draw=True)
        score += net_points(board.result(claim_draw=True) if over else "1/2-1/2",
                            a_white)
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="paths to compare; defaults to the known runs")
    args = ap.parse_args()

    defaults = {
        "pretrained": ROOT / "checkpoints" / "iter_200.pt",
        "v5 iter_500": ROOT / "checkpoints" / "iter_500.pt",
        "v4 iter_700": ROOT / "checkpoints_v6_iter700" / "iter_700.pt",
    }
    if args.checkpoints:
        defaults = {Path(p).stem: Path(p) for p in args.checkpoints}

    nets = {}
    for name, path in defaults.items():
        if not Path(path).exists():
            print(f"skip {name}: {path} not found")
            continue
        try:
            nets[name] = load(path)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {name}: {type(exc).__name__} (architecture mismatch?)")
    if not nets:
        print("no loadable checkpoints")
        return

    print("\n=== raw policy on the start position (no search) ===")
    for name, net in nets.items():
        moves, value = top_policy(net, chess.Board())
        print(f"{name:16s} value={value:+.3f}  {moves}")

    print("\n=== tactical probes (100 sims) ===")
    for label, fen, want in TACTICS:
        board = chess.Board(fen)
        print(f"  {label} (expect {want}):")
        for name, net in nets.items():
            mcts = MCTS(net, torch.device("cpu"), cfg.c_puct, dirichlet_alpha=0.0)
            probs = mcts.get_action_probs(board, 100)
            got = board.san(max(probs, key=probs.get))
            print(f"    {name:16s} {got:8s} {'OK' if got == want else 'MISS'}")

    print(f"\n=== vs random, net-perspective ({args.games} games) ===")
    for name, net in nets.items():
        t0 = time.time()
        w, d, losses = vs_random(net, args.games, cfg.num_simulations)
        pct = 100 * (w + 0.5 * d) / args.games
        print(f"{name:16s} +{w} ={d} -{losses}  score {pct:.0f}%  ({time.time()-t0:.0f}s)")

    names = list(nets)
    if len(names) >= 2:
        a, b = names[0], names[1]
        print(f"\n=== head to head: {a} vs {b} "
              f"({args.games} games @ {args.sims} sims) ===")
        s = head_to_head(nets[a], nets[b], args.games, args.sims)
        print(f"{a} {s:.1f} - {args.games - s:.1f} {b}")


if __name__ == "__main__":
    main()
