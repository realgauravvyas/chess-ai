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


def default_checkpoint():
    """Newest gated best.pt, else the newest iter_N.pt, across all runs.

    Iteration numbers are not comparable between runs, so order by mtime.
    """
    dirs = [ROOT / "checkpoints"] + sorted(
        (ROOT / "runs").glob("*/checkpoints")) if (ROOT / "runs").exists() \
        else [ROOT / "checkpoints"]
    bests = [q for d in dirs for q in d.glob("best.pt")]
    if bests:
        return max(bests, key=lambda q: q.stat().st_mtime)
    iters = [q for d in dirs for q in d.glob("iter_*.pt")]
    return max(iters, key=lambda q: q.stat().st_mtime) if iters else None


def material(board, net_color):
    s = 0
    for p in board.piece_map().values():
        v = VALS[p.piece_type]
        s += v if p.color == net_color else -v
    return s



def worst_blunder(history, diffs, net_white, net_color):
    """Largest single material loss caused by an opponent move.

    `history` is [(san, fen_after, ply_after), ...] and `diffs[i]` is the
    network's material balance after i half-moves. diffs[0] is the starting
    position, so the balance *after* history[k] is diffs[k+1] and the
    balance *before* it is diffs[k] -- diffs is one longer than history and
    offset by one against it.

    Only opponent moves are considered: the network's material balance falls
    when the OPPONENT captures. board.ply() is odd once White has moved, so
    the opponent moved whenever that parity differs from the network's
    colour. Getting either the parity or the offset wrong makes every
    candidate drop <= 0 and the whole pass silently produce nothing.
    """
    worst = None
    for k in range(len(history)):
        ply = history[k][2]                       # ply AFTER this half-move
        opp_moved = (ply % 2 == 1) != net_white
        if not opp_moved:
            continue
        drop = diffs[k] - diffs[k + 1]
        if worst is None or drop > worst[0]:
            worst = (drop, k)
    drop, k = worst if worst else (0, None)

    info = {"drop": drop}
    if k is not None and drop >= 2:
        san = history[k][0]
        # position before the opponent's move: the FEN after the previous
        # half-move, or the initial position when it was the very first.
        before = chess.Board(history[k - 1][1]) if k else chess.Board()
        mv = before.parse_san(san)
        tgt = mv.to_square
        mover = before.piece_at(mv.from_square)
        captured = before.piece_type_at(tgt)
        attackers = len(list(before.attackers(not net_color, tgt)))
        defenders = len(list(before.attackers(net_color, tgt)))
        info.update({
            "phase": ("opening" if k <= 20 else
                      "middlegame" if k <= 70 else "endgame"),
            "hung_piece": VALS.get(captured, 0) if captured else 0,
            "mover_piece": VALS.get(mover.piece_type, 0) if mover else 0,
            "undefended": attackers > 0 and defenders == 0,
            "san": san,
        })
    return info


def main():
    cfg = Config()
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_losses = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                       action_planes=cfg.policy_size // 64)
    ckpt = Path(sys.argv[4]) if len(sys.argv) > 4 else default_checkpoint()
    if ckpt is None or not ckpt.exists():
        sys.exit(f"no checkpoint found (looked for {ckpt}); pass one as arg 4")
    print(f"analysing {ckpt}")
    blob = load_checkpoint(str(ckpt))
    net.load_state_dict(blob.get("model_state_dict", blob))
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
        while not board.is_game_over(claim_draw=True) and board.ply() < 300:
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

        info = worst_blunder(history, diffs, net_white, net_color)
        info["plies"] = board.ply()
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
