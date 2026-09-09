"""Prove the test suite would have caught the bugs that actually happened.

A green suite means nothing on its own - it may simply not exercise the
broken path. This reintroduces each historical bug one at a time, runs
tests/test_suite.py, and asserts the suite FAILS. Any mutation that leaves
the suite green is a coverage hole.

Every edit is reverted from an in-memory backup in a finally block, so the
working tree is unchanged whatever happens.

    python tests/test_mutations.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "tests" / "test_suite.py"

# (label, file, original snippet, buggy snippet as it once was)
MUTATIONS = [
    ("evaluation scored White's games, not the network's",
     "evaluate.py",
     '''    if result == "1/2-1/2":
        return 0.5
    white_won = result == "1-0"
    return 1.0 if white_won == net_plays_white else 0.0''',
     '''    if result == "1/2-1/2":
        return 0.5
    return 1.0 if result == "1-0" else 0.0'''),

    ("mirror augmentation corrupted castling planes",
     "train.py",
     "        castling_free = not planes[i][12:16].any()\n"
     "        if castling_free and random.random() < 0.5:",
     "        if random.random() < 0.5:"),

    ("loss forensics had inverted parity",
     "analyze_losses.py",
     "        opp_moved = (ply % 2 == 1) != net_white",
     "        opp_moved = (ply % 2 == 0) != net_white"),

    ("loss forensics was off by one in diffs",
     "analyze_losses.py",
     "        drop = diffs[k] - diffs[k + 1]",
     "        drop = diffs[k - 1] - diffs[k]"),

    ("plain .pgn fed to the zstd reader",
     "pretrain_supervised.py",
     "        is_zst = fh.read(4) == bytes([0x28, 0xb5, 0x2f, 0xfd])",
     "        is_zst = True"),

    ("value head could return values outside [-1, 1]",
     "model.py",
     "        return torch.tanh(self.fc2(v))",
     "        return self.fc2(v) * 2.0"),

    ("taught models were not listed as selectable checkpoints",
     "dashboard/server.py",
     '    return (name.startswith("iter_") or name.startswith("taught_")\n'
     '            or name in ("latest.pt", "best.pt")) and name.endswith(".pt")',
     '    return name.startswith("iter_") and name.endswith(".pt")'),

    ("the default opponent ignored gated best.pt checkpoints",
     "dashboard/server.py",
     '    bests = [p for d in run_dirs\n'
     '             for p in glob.glob(str(d / "best.pt"))]',
     '    bests = []  # pretend no gated winner exists'),

    ("the teach loop trained the shared cached network in place",
     "dashboard/server.py",
     "        net = copy.deepcopy(load_net(ckpt_path))",
     "        net = load_net(ckpt_path)"),

    ("the worker count was taken unvalidated from argv",
     "run_training.py",
     '    if args and args[0].startswith("--"):\n'
     '        sys.exit(f"expected the worker count first, got {args[0]!r}.'
     '\\n\\n{USAGE}")\n'
     '    workers = args[0] if args else "6"\n'
     '    if not workers.isdigit():\n'
     '        sys.exit(f"worker count must be a number, got {workers!r}.'
     '\\n\\n{USAGE}")',
     '    workers = args[0] if args else "6"'),

    ("a flag with no value raised a bare IndexError",
     "run_training.py",
     '    if i + 1 >= len(args) or args[i + 1].startswith("--"):\n'
     '        sys.exit(f"{flag} needs a value.\\n\\n{USAGE}")',
     '    pass  # no validation'),

    ("play.py left the player stuck when promoting a pawn",
     "play.py",
     "                queened = chess.Move(move.from_square, move.to_square,\n"
     "                                     promotion=chess.QUEEN)",
     "                queened = move  # no auto-queen"),

    ("eval_match.py hardcoded the regressed checkpoints",
     "eval_match.py",
     "    new_path = Path(sys.argv[3]) if len(sys.argv) > 3 else default_new()",
     '    new_path = ROOT / "checkpoints" / "iter_500.pt"'),

    ("sqToDisp did not invert dispToSq on a flipped board",
     "dashboard/static/index.html",
     "  return S.flipped ? [r,7-f] : [7-r,f];",
     "  return S.flipped ? [r,f] : [7-r,f];"),

    ("a truncated download was accepted as complete",
     "pretrain_supervised.py",
     "        if have == want:",
     "        if True:  # accept any non-empty file"),
]


def run_suite(timeout=900):
    """Return True if the suite passes."""
    r = subprocess.run([sys.executable, str(SUITE)],
                       cwd=str(ROOT), capture_output=True, text=True,
                       timeout=timeout)
    return r.returncode == 0, r.stdout


def main():
    print("Verifying the suite is green before mutating...")
    ok, out = run_suite()
    if not ok:
        print("baseline suite is already failing - fix that first:")
        print(out[-1500:])
        sys.exit(1)
    print(f"  baseline: {out.strip().splitlines()[-1]}\n")

    caught = 0
    for label, fname, good, bad in MUTATIONS:
        path = ROOT / fname
        original = path.read_text(encoding="utf-8")
        if good not in original:
            print(f"SKIP  {label}\n        anchor not found in {fname}")
            continue
        try:
            path.write_text(original.replace(good, bad, 1), encoding="utf-8")
            ok, out = run_suite()
            if ok:
                print(f"MISSED  {label}\n        suite stayed green - "
                      f"coverage hole in {fname}")
            else:
                caught += 1
                last = out.strip().splitlines()[-1] if out.strip() else "?"
                print(f"caught  {label}\n        -> {last}")
        finally:
            path.write_text(original, encoding="utf-8")

    total = len(MUTATIONS)
    print(f"\n{caught}/{total} historical bugs are caught by the suite")

    print("\nConfirming the working tree was restored...")
    ok, out = run_suite()
    print(f"  {out.strip().splitlines()[-1]}")
    sys.exit(0 if (caught == total and ok) else 1)


if __name__ == "__main__":
    main()
