"""Verify the browser engine encodes boards and moves exactly like Python.

The GitHub Pages demo runs the same network, but the tensor and the 4672
action indices are built in JavaScript. If either differs from
utils.board_to_planes or move_encoding.encode_move by a single square, the
network is fed an input it never saw in training and plays nonsense - a
failure that looks like "the model is just weak" rather than a bug.

This generates ground truth from Python, runs the JavaScript over the same
positions, and requires an exact match on every plane and every move index.

    python tests/test_pages_encoding.py

Needs node and chess.js; skips cleanly if either is unavailable.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from move_encoding import encode_move          # noqa: E402
from utils import board_to_planes              # noqa: E402

POSITIONS = [
    chess.Board().fen(),
    # castling rights on both sides
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    # an en-passant target
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    # promotions available, including underpromotions
    "6k1/4P3/8/8/8/8/4p3/4K3 w - - 0 1",
    # black to move, partial castling rights
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",
    # a middlegame with knights and long-range pieces
    "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1 w - - 0 9",
    # an endgame
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]

JS_DRIVER = r"""
const fs = require("fs");
const { Chess } = require("chess.js");
const src = fs.readFileSync(process.argv[2], "utf8");
// engine.js references ort and Chess at call time only; expose Chess here
global.Chess = Chess;
eval(src.replace(/^\s*const FILES = "abcdefgh";/m, 'FILES = "abcdefgh";'));

const fens = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const out = {};
for (const fen of fens) {
  const g = new Chess(fen);
  const planes = Array.from(boardToPlanes(g));
  const moves = {};
  for (const m of g.moves({ verbose: true })) {
    moves[m.from + m.to + (m.promotion || "")] = encodeMove(m);
  }
  out[fen] = { planes, moves };
}
console.log(JSON.stringify(out));
"""


def main():
    node = shutil.which("node")
    if not node:
        print("skip: node not installed")
        return 0

    jscheck = Path(r"C:/Users/Gaurav/AppData/Local/Temp/claude/"
                   r"d--VS-Code/40e99891-6a16-41fe-9517-4c4df0ea431e/"
                   r"scratchpad/jscheck")
    if not (jscheck / "node_modules" / "chess.js").exists():
        print("skip: chess.js not installed for node")
        return 0

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "fens.json").write_text(json.dumps(POSITIONS), encoding="utf-8")
        driver = jscheck / "driver.js"
        driver.write_text(JS_DRIVER, encoding="utf-8")
        r = subprocess.run(
            [node, str(driver), str(ROOT / "docs" / "engine.js"),
             str(d / "fens.json")],
            capture_output=True, text=True, cwd=str(jscheck), timeout=300)

    if r.returncode != 0:
        print("FAIL: the JavaScript driver errored")
        print(r.stderr[-1200:])
        return 1
    js = json.loads(r.stdout)

    failures = 0
    for fen in POSITIONS:
        board = chess.Board(fen)
        want_planes = board_to_planes(board).reshape(-1)
        got_planes = np.array(js[fen]["planes"], dtype=np.float32)

        if got_planes.shape != want_planes.shape:
            print(f"FAIL  {fen}\n        plane count "
                  f"{got_planes.shape} != {want_planes.shape}")
            failures += 1
            continue
        diff = np.flatnonzero(got_planes != want_planes)
        if diff.size:
            failures += 1
            p, rem = divmod(int(diff[0]), 64)
            print(f"FAIL  {fen}\n        {diff.size} plane cells differ; "
                  f"first at plane {p}, square {chess.square_name(rem)} "
                  f"(python {want_planes[diff[0]]}, js {got_planes[diff[0]]})")
        else:
            print(f"pass  planes match   {fen[:46]}")

        want_moves = {m.uci(): encode_move(m) for m in board.legal_moves}
        got_moves = js[fen]["moves"]
        if set(want_moves) != set(got_moves):
            missing = set(want_moves) - set(got_moves)
            extra = set(got_moves) - set(want_moves)
            failures += 1
            print(f"FAIL  {fen}\n        legal moves differ; "
                  f"python-only {sorted(missing)[:4]}, js-only {sorted(extra)[:4]}")
            continue
        bad = {u: (want_moves[u], got_moves[u])
               for u in want_moves if want_moves[u] != got_moves[u]}
        if bad:
            failures += 1
            sample = list(bad.items())[:4]
            print(f"FAIL  {fen}\n        {len(bad)}/{len(want_moves)} move "
                  f"indices differ, e.g. {sample}")
        else:
            print(f"pass  {len(want_moves):>2} move indices match")

    print()
    if failures:
        print(f"{failures} MISMATCH(ES) - the browser demo would feed the "
              f"network the wrong input")
    else:
        print("browser encoding is identical to Python on every position")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
