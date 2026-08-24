"""Loss forensics: find HOW the model loses to random players.

Replays net-vs-random games, and for every loss records the single worst
material swing: which piece was hung, on which ply (phase), whether the
landing square was defended, and whether a bigger piece took a smaller one.
"""
import random
import sys
from collections import Counter
from pathlib import Path

import chess
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import Config
from mcts import MCTS
from model import AlphaZeroNet
from utils import load_checkpoint

VALS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material(board, net_color):
    s = 0
    for p in board.piece_map().values():
        v = VALS[p.piece_type]
        s += v if p.color == net_color else -v
    return s


def main():
    cfg = Config()
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_losses = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                       action_planes=cfg.policy_size // 64)
    blob = load_checkpoint("checkpoints/iter_500.pt")
    net.load_state_dict(blob["model_state_dict"])
    net.eval()

    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    random.seed(seed)
    torch.manual_seed(seed)

    losses = []
    wins = draws = 0
    game_no = 0
    while len(losses) < max_losses and game_no < 40:
        game_no += 1
        net_white = (game_no % 2 == 1)
        net_color = chess.WHITE if net_white else chess.BLACK
        board = chess.Board()
        mcts = MCTS(net, torch.device("cpu"), cfg.c_puct, dirichlet_alpha=0.0)

        history = []           # (san, fen_after, ply)
        diffs = [material(board, net_color)]
        while not board.is_game_over() and board.ply() < 300:
            if board.turn == net_color:
                probs = mcts.get_action_probs(board, sims)
                if not probs:
                    break
                move = max(probs, key=probs.get)
            else:
                move = random.choice(list(board.legal_moves))
            san = board.san(move)
            board.push(move)
            history.append((san, board.fen(), board.ply()))
            diffs.append(material(board, net_color))

        out = board.outcome(claim_draw=True)
        if out is None or out.winner is None:
            draws += 1
            continue
        net_won = (out.winner == net_color)
        if net_won:
            wins += 1
            continue

        # ---- forensic pass on this loss ----
        # find largest single-opponent-move material drop
        worst = None  # (drop, ply_idx, fen_before_opp, fen_after_opp)
        for k in range(1, len(history)):
            ply = history[k][2]          # ply AFTER this half-move
            opp_moved = (ply % 2 == 0) != net_white  # odd plies are white
            drop = diffs[k - 1] - diffs[k] if not opp_moved else 0
            if not opp_moved:
                continue
            drop = diffs[k - 1] - diffs[k]
            if worst is None or drop > worst[0]:
                worst = (drop, k)
        drop, k = worst if worst else (0, None)

        info = {"drop": drop, "plies": board.ply()}
        if k is not None and drop >= 2:
            san, fen_after, _ = history[k]
            fb = chess.Board(history[k - 1][1])   # position before opp move
            mv = fb.parse_san(san)
            tgt = mv.to_square
            movers = fb.piece_at(mv.from_square)
            # what got captured on the target square
            captured = fb.piece_type_at(tgt)
            attackers = len(list(fb.attackers(not net_color, tgt)))
            defenders = len(list(fb.attackers(net_color, tgt)))
            info.update({
                "phase": ("opening" if k <= 20 else
                          "middlegame" if k <= 70 else "endgame"),
                "hung_piece": (VALS.get(captured, 0) if captured else 0),
                "mover_piece": VALS.get(movers.piece_type, 0) if movers else 0,
                "undefended": attackers > 0 and defenders == 0,
                "san": san,
            })
        losses.append(info)
        print(f"[{game_no}] LOSS as {'white' if net_white else 'black'} "
              f"(worst drop {info['drop']:+d})", flush=True)

    print(f"\n=== summary: {wins}W {len(losses)}L {draws}D over {game_no} games ===")
    if not losses:
        print("no losses to analyze")
        return
    phases = Counter(l.get("phase", "?") for l in losses)
    undefended = sum(1 for l in losses if l.get("undefended"))
    hung = Counter(l["hung_piece"] for l in losses if "hung_piece" in l)
    colors = Counter()  # filled implicitly by loop order; recount below skipped
    print(f"blunder phase : {dict(phases)}")
    print(f"pure hangs (piece taken on undefended square): {undefended}/{len(losses)}")
    print(f"value of piece lost (1=P 3=N/B 5=R 9=Q): {dict(sorted(hung.items()))}")
    avg_ply = sum(l["plies"] for l in losses) / len(losses)
    print(f"average game length at loss: {avg_ply:.0f} plies")
    big = sorted(losses, key=lambda l: -l.get("drop", 0))[:3]
    for l in big:
        print(f"  worst collapse: {l.get('san','?')} (drop {l['drop']:+d}, "
              f"{l.get('phase','?')})")


if __name__ == "__main__":
    main()
