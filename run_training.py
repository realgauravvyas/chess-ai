"""Self-healing training wrapper.

Runs train.py in a loop, automatically resuming from the latest checkpoint
whenever it crashes (e.g. Windows multiprocessing pipe failures), until the
target iteration count is reached.
"""
import glob
import os
import re
import subprocess
import sys
import time

MAX_ATTEMPTS = 20
COOLDOWN = 10  # seconds before restarting after a crash


def find_latest_checkpoint(checkpoint_dir):
    """Return the newest iter_N.pt checkpoint path, or None."""
    best, best_n = None, -1
    for path in glob.glob(os.path.join(checkpoint_dir, "iter_*.pt")):
        m = re.match(r"iter_(\d+)\.pt$", os.path.basename(path))
        if m and int(m.group(1)) > best_n:
            best_n = int(m.group(1))
            best = path
    return best


USAGE = """usage: run_training.py [--run-dir DIR] [--seed-from CKPT] \
[WORKERS] [train.py options...]

  --run-dir DIR     keep this run's checkpoints and replay buffer in DIR
                    instead of the shared project checkpoints/ folder
  --seed-from CKPT  starting checkpoint for the first attempt, used only
                    while the run directory has no checkpoints of its own
  WORKERS           parallel self-play processes (default 6). Positional,
                    and must come before any train.py option.

Everything after WORKERS is passed straight through to train.py.

example:
  python run_training.py --run-dir runs/v8 --seed-from checkpoints/iter_200.pt \
6 --device auto --sims 128 --gate
"""


def take_value(args, flag):
    """Pop `flag` and its value out of `args`; return the value or None."""
    if flag not in args:
        return None
    i = args.index(flag)
    if i + 1 >= len(args) or args[i + 1].startswith("--"):
        sys.exit(f"{flag} needs a value.\n\n{USAGE}")
    value = args[i + 1]
    del args[i:i + 2]
    return value


def main():
    python_exe = sys.executable
    project_dir = os.path.dirname(os.path.abspath(__file__))
    args = sys.argv[1:]

    if "-h" in args or "--help" in args:
        print(USAGE)
        return

    # --run-dir lets a run keep its own checkpoints/ and replay buffer
    # instead of writing into the shared project checkpoints/ folder.
    run_dir = take_value(args, "--run-dir")
    run_dir = os.path.abspath(run_dir) if run_dir else project_dir
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    # --seed-from gives the first attempt a starting checkpoint when the
    # run directory is still empty.
    seed_from = take_value(args, "--seed-from")
    if seed_from:
        seed_from = os.path.abspath(seed_from)
        if not os.path.exists(seed_from):
            sys.exit(f"--seed-from checkpoint not found: {seed_from}")

    checkpoint_dir = os.path.join(run_dir, "checkpoints")

    # The worker count is positional and must come first. Without this check
    # `run_training.py --device auto 6` silently becomes
    # `train.py --workers --device auto 6`, which fails argparse on every
    # one of the 20 retries below.
    if args and args[0].startswith("--"):
        sys.exit(f"expected the worker count first, got {args[0]!r}.\n\n{USAGE}")
    workers = args[0] if args else "6"
    if not workers.isdigit():
        sys.exit(f"worker count must be a number, got {workers!r}.\n\n{USAGE}")
    passthrough = args[1:]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        ckpt = find_latest_checkpoint(checkpoint_dir) or seed_from
        cmd = [python_exe, "-u", os.path.join(project_dir, "train.py"),
               "--workers", workers] + passthrough
        if ckpt:
            cmd += ["--resume", ckpt]
            print(f"[wrapper] attempt {attempt}: resuming from {os.path.basename(ckpt)}")
        else:
            print(f"[wrapper] attempt {attempt}: fresh start")

        result = subprocess.run(cmd, cwd=run_dir)
        if result.returncode == 0:
            print("[wrapper] training completed successfully.")
            return

        # argparse exits 2 on a usage error. That is deterministic, so
        # retrying it 20 times just hides the mistake for three minutes.
        if result.returncode == 2:
            print("[wrapper] train.py rejected its arguments (exit 2); "
                  "not retrying.")
            sys.exit(2)

        print(f"[wrapper] train.py exited with code {result.returncode}; "
              f"restarting in {COOLDOWN}s...")
        time.sleep(COOLDOWN)

    print(f"[wrapper] giving up after {MAX_ATTEMPTS} attempts.")


if __name__ == "__main__":
    main()
