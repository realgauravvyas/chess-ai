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


def main():
    python_exe = sys.executable
    project_dir = os.path.dirname(os.path.abspath(__file__))
    args = sys.argv[1:]

    # --run-dir lets a run keep its own checkpoints/ and replay buffer
    # instead of writing into the shared project checkpoints/ folder.
    run_dir = project_dir
    if "--run-dir" in args:
        i = args.index("--run-dir")
        run_dir = os.path.abspath(args[i + 1])
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        del args[i:i + 2]

    # --seed-from gives the first attempt a starting checkpoint when the
    # run directory is still empty.
    seed_from = None
    if "--seed-from" in args:
        i = args.index("--seed-from")
        seed_from = os.path.abspath(args[i + 1])
        del args[i:i + 2]

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    workers = args[0] if args else "6"
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

        print(f"[wrapper] train.py exited with code {result.returncode}; "
              f"restarting in {COOLDOWN}s...")
        time.sleep(COOLDOWN)

    print(f"[wrapper] giving up after {MAX_ATTEMPTS} attempts.")


if __name__ == "__main__":
    main()
