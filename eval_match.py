"""Head-to-head: v5 final (iter_500) vs previous best (v4 iter_700)."""
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


def load(path):
    cfg = Config()
    net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                       action_planes=cfg.policy_size // 64)
    blob = load_checkpoint(path)
    net.load_state_dict(blob["model_state_dict"])
    net.eval()
    return net


def main():
    cfg = Config()
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    new_net = load(ROOT / "checkpoints" / "iter_500.pt")
    old_net = load(ROOT / "checkpoints_v6_iter700" / "iter_700.pt")

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
