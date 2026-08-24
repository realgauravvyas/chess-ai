"""Watchdog: keeps the dashboard server alive on port 5000."""
import socket
import subprocess
import sys
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
    while True:
        if DISABLED_FLAG.exists():
            time.sleep(10)   # server intentionally stopped; don't revive
            continue
        if not is_up():
            print(f"[watchdog] port {PORT} down; starting dashboard...",
                  flush=True)
            subprocess.Popen(
                [python, "-u", server, "--port", str(PORT)],
                cwd=str(ROOT),
                stdout=open(ROOT / "logs" / "dashboard.log", "ab"),
                stderr=subprocess.STDOUT,
            )
            time.sleep(8)  # give it time to bind
        time.sleep(20)


if __name__ == "__main__":
    main()
