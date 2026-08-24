"""Wall-clock cost of one self-play iteration at several simulation counts.

Use this to budget a training run: multiply sec/iteration by the number of
iterations you intend to run.

Usage:
    python experiments/benchmark_selfplay.py --sims 40 96 128 160
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config                    # noqa: E402
from model import AlphaZeroNet               # noqa: E402
from selfplay import _worker_init, _worker_play_one  # noqa: E402
from utils import load_checkpoint            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "iter_200.pt"))
    ap.add_argument("--sims", type=int, nargs="*", default=[40, 96, 128, 160])
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfg = Config()
    net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                       cfg.policy_size // 64)
    net.load_state_dict(load_checkpoint(args.checkpoint)["model_state_dict"])
    net.eval()
    state = {k: v.cpu() for k, v in net.state_dict().items()}

    print(f"{args.games} games on {args.workers} workers")
    print(f"{'sims':>6} {'sec/iter':>9} {'positions':>10} {'hours/100 iters':>16}")
    for sims in args.sims:
        c = Config()
        c.num_simulations = sims
        c.games_per_iteration = args.games
        pool = mp.Pool(args.workers, initializer=_worker_init, initargs=("cpu", c))
        t0 = time.time()
        games = pool.map(_worker_play_one, [state] * c.games_per_iteration)
        dt = time.time() - t0
        pool.close()
        pool.join()
        print(f"{sims:6d} {dt:9.1f} {sum(len(g) for g in games):10d} "
              f"{dt*100/3600:15.1f}h", flush=True)


if __name__ == "__main__":
    main()
