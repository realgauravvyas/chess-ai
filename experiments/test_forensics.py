"""Test the blunder attribution in analyze_losses.py.

This path only executes when the network loses, and it now beats a random
mover 24-0-16, so it went unexercised for the whole project while carrying
an inverted parity test that made it silently produce nothing. These cases
pin the behaviour to games with known answers.

    python experiments/test_forensics.py
"""
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyze_losses import material, worst_blunder   # noqa: E402


def build(moves, net_color):
    """Replay `moves` (UCI), returning the history/diffs the analyser uses."""
    board = chess.Board()
    history, diffs = [], [material(board, net_color)]
    for uci in moves:
        mv = chess.Move.from_uci(uci)
        san = board.san(mv)
        board.push(mv)
        history.append((san, board.fen(), board.ply()))
        diffs.append(material(board, net_color))
    return history, diffs


CASES = []

# 1. Network is White and hangs its queen; Black takes it with a pawn.
#    1.e4 d5  2.Qh5 Nf6  3.Qxf7+?? Kxf7  -> White loses a 9-point queen to a
#    king recapture. The worst opponent move must be Kxf7, drop 9.
CASES.append((
    "white hangs the queen on f7",
    ["e2e4", "d7d5", "d1h5", "g8f6", "h5f7", "e8f7"],
    chess.WHITE,
    {"drop": 9, "san": "Kxf7", "hung_piece": 9},
))

# 2. Same game from Black's side: Black is the network, and the largest
#    opponent (White) capture is Qxf7+, taking a pawn -> drop 1, below the
#    reporting threshold of 2, so no forensic detail is attached.
CASES.append((
    "black loses only a pawn - below threshold",
    ["e2e4", "d7d5", "d1h5", "g8f6", "h5f7", "e8f7"],
    chess.BLACK,
    {"drop": 1, "san": None},
))

# 3. Scholar's mate: Black (the network) is checkmated and loses a knight
#    on the way. Largest White capture is Qxf7# taking a pawn (1), but
#    Bxc6 earlier is not played, so the drop stays 1.
CASES.append((
    "black is mated, worst capture is a pawn",
    ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"],
    chess.BLACK,
    {"drop": 1, "san": None},
))

# 4. Network (White) drops a rook to an undefended square.
#    1.a4 e5  2.Ra3 Qf6  3.Rb3 Qxb2  -> Black queen takes the b2 pawn;
#    then 4.Rc3 Qxc1 wins a bishop. Largest single drop should be the
#    bishop (3), not the pawn.
CASES.append((
    "white loses a bishop, bigger than the earlier pawn",
    ["a2a4", "e7e5", "a1a3", "d8f6", "a3b3", "f6b2", "b3c3", "b2c1"],
    chess.WHITE,
    {"drop": 3, "san": "Qxc1", "hung_piece": 3},
))


def main():
    failures = 0
    for name, moves, net_color, expect in CASES:
        net_white = (net_color == chess.WHITE)
        history, diffs = build(moves, net_color)
        got = worst_blunder(history, diffs, net_white, net_color)

        problems = []
        if got["drop"] != expect["drop"]:
            problems.append(f"drop {got['drop']} != {expect['drop']}")
        if expect["san"] is None:
            if "san" in got:
                problems.append(f"expected no detail, got san={got['san']!r}")
        else:
            if got.get("san") != expect["san"]:
                problems.append(f"san {got.get('san')!r} != {expect['san']!r}")
            if "hung_piece" in expect and got.get("hung_piece") != expect["hung_piece"]:
                problems.append(
                    f"hung_piece {got.get('hung_piece')} != {expect['hung_piece']}")

        if problems:
            failures += 1
            print(f"FAIL  {name}")
            for pr in problems:
                print(f"        {pr}")
            print(f"        full result: {got}")
        else:
            detail = f"  ({got['san']}, hung {got.get('hung_piece')})" \
                if "san" in got else "  (no detail, as expected)"
            print(f"PASS  {name}: drop {got['drop']}{detail}")

    print(f"\n{len(CASES) - failures} passed, {failures} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
