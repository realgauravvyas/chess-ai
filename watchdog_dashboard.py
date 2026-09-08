"""Watchdog: keeps the dashboard server alive on port 5000."""
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = 5000
DISABLED_FLAG = ROOT / "logs" / "dashboard_disabled.flag"


def is_up():
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=2):
            return True
    except OSError:
        return False


def main():
    python = str(ROOT / ".venv" / "Scripts" / "python.exe")
    server = str(ROOT / "dashboard" / "server.py")
    # logs/ is gitignored, so it does not exist in a fresh clone.
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    while True:
        if DISABLED_FLAG.exists():
            time.sleep(10)   # server intentionally stopped; don't revive
            continue
        if not is_up():
            print(f"[watchdog] port {PORT} down; starting dashboard...",
                  flush=True)
            # Hold the handle only across Popen: the child dups the
            # descriptor, so leaving it open here leaks one per restart.
            with open(log_dir / "dashboard.log", "ab") as log:
                subprocess.Popen(
                    [python, "-u", server, "--port", str(PORT)],
                    cwd=str(ROOT),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            time.sleep(8)  # give it time to bind
            if is_up():
                failures = 0
            else:
                failures += 1
                # Back off instead of hammering a server that cannot start.
                backoff = min(300, 20 * 2 ** min(failures, 4))
                print(f"[watchdog] still down after {failures} attempt(s); "
                      f"retrying in {backoff}s", flush=True)
                time.sleep(backoff)
                continue
        time.sleep(20)


if __name__ == "__main__":
    main()
