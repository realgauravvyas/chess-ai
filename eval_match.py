"""Quick head-to-head between two checkpoints.

For a result you can quote, use experiments/final_verdict.py instead: it
parallelises across cores and reports a confidence interval, which an
8-game match cannot support.

    python eval_match.py [SIMS] [GAMES] [NEW_CKPT] [OLD_CKPT]
"""
import sys
from pathlib import Path

import chess
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import Config
from mcts import MCTS
from model import AlphaZeroNet
from utils import load_checkpoint


def default_new():
    """Newest gated best.pt, else the newest iter_N.pt, across all runs."""
    dirs = [ROOT / "checkpoints"] + sorted((ROOT / "runs").glob("*/checkpoints"))
    for pattern in ("best.pt", "iter_*.pt"):
        found = [q for d in dirs for q in d.glob(pattern)]
        if found:
            return max(found, key=lambda q: q.stat().st_mtime)
    return None


def load(path):
    cfg = Config()
    net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                       action_planes=cfg.policy_size // 64)
    blob = load_checkpoint(str(path))
    net.load_state_dict(blob["model_state_dict"])
    net.eval()
    return net


def main():
    cfg = Config()
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    new_path = Path(sys.argv[3]) if len(sys.argv) > 3 else default_new()
    old_path = Path(sys.argv[4]) if len(sys.argv) > 4 else \
        ROOT / "checkpoints_v7_baseline" / "iter_200.pt"
    for label, path in (("new", new_path), ("old", old_path)):
        if path is None or not Path(path).exists():
            sys.exit(f"{label} checkpoint not found: {path}")
    print(f"NEW {Path(new_path).name}  vs  OLD {Path(old_path).name}")
    new_net = load(new_path)
    old_net = load(old_path)

    score_new = 0.0
    for i in range(games):
        new_white = (i % 2 == 0)
        nets = {chess.WHITE: new_net if new_white else old_net,
                chess.BLACK: old_net if new_white else new_net}
        board = chess.Board()
        while not board.is_game_over() and board.ply() < 300:
            m = MCTS(nets[board.turn], torch.device("cpu"), cfg.c_puct,
                     dirichlet_alpha=0.0)
            probs = m.get_action_probs(board, sims)
            if not probs:
                break
            board.push(max(probs, key=probs.get))
        out = board.outcome(claim_draw=True)
        pts = 0.5 if (out is None or out.winner is None) else (
            1.0 if (out.winner == chess.WHITE) == new_white else 0.0)
        score_new += pts
        verdict = "draw" if pts == 0.5 else ("NEW wins" if pts == 1 else "OLD wins")
        print(f"game {i+1}: {verdict} (new as "
              f"{'white' if new_white else 'black'})", flush=True)

    print(f"\nFINAL: NEW {score_new:.1f} - {games - score_new:.1f} OLD "
          f"({games} games, {sims} sims)")


if __name__ == "__main__":
    main()
