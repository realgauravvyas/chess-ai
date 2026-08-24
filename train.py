"""Main training loop: self-play -> train -> evaluate -> checkpoint."""
import argparse
import multiprocessing as mp
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from evaluate import evaluate_vs_baseline
from model import AlphaZeroNet, count_parameters
from move_encoding import mirror_action_index
from selfplay import _worker_init, _worker_play_one, generate_games, play_one_game
from utils import load_checkpoint


def make_optimizer(net, cfg):
    return torch.optim.Adam(net.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)


def train_step(net, optimizer, batch, device, cfg):
    """One gradient step on a minibatch. Returns (loss, policy_loss, value_loss)."""
    planes = np.stack([b[0] for b in batch]).astype(np.float32)

    # Left-right mirror augmentation.
    #
    # NOTE: chess is NOT fully mirror-symmetric. A file flip moves the king
    # e1->d1, so castling rights cannot survive it. We therefore only mirror
    # positions with no castling rights left (planes 12..15 all zero);
    # mirroring a position that still has them produces a board that cannot
    # occur in chess, paired with castling flags that contradict the pieces.
    from move_encoding import mirror_action_index as _mir  # local import ok
    for i, sample in enumerate(batch):
        castling_free = not planes[i][12:16].any()
        if castling_free and random.random() < 0.5:
            planes[i] = planes[i][:, :, ::-1]
            remap = {}
            for jdx, p in zip(sample[1], sample[2]):
                m = _mir(jdx)
                remap[m] = remap.get(m, 0.0) + float(p)
            keys = list(remap.keys())
            batch[i] = (sample[0], keys,
                        np.array([remap[k] for k in keys], dtype=np.float32),
                        sample[3])

    z = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)

    x = torch.from_numpy(planes).to(device)
    logits, value = net(x)

    policy_loss = 0.0
    for i, sample in enumerate(batch):
        indices, probs = sample[1], sample[2]
        target = torch.zeros(cfg.policy_size, device=device)
        if indices:
            idx = torch.tensor(indices, dtype=torch.long, device=device)
            target[idx] = torch.tensor(probs, dtype=torch.float32, device=device)
        log_p = F.log_softmax(logits[i], dim=0)
        policy_loss = policy_loss - (target * log_p).sum()
    policy_loss = policy_loss / len(batch)

    value_loss = F.mse_loss(value.view(-1), z)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item()), float(policy_loss.item()), float(value_loss.item())


def save_checkpoint(net, optimizer, scheduler, iteration, path):
    torch.save({
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "iteration": iteration,
    }, path)


REPLAY_PATH = "checkpoints/replay_buffer.pt"
ANCHOR_PATH = "checkpoints/anchor_data.pt"


def maybe_save_replay(replay, it, interval=25):
    """Periodically persist the replay buffer so crashes don't wipe it."""
    if replay and (it + 1) % interval == 0:
        torch.save(replay, REPLAY_PATH)
        print(f"[iter {it}] persisted replay buffer ({len(replay)} positions)")


_anchor_cache = None


def load_anchor():
    """Supervised rehearsal set that anchors fine-tuning to human play."""
    global _anchor_cache
    if _anchor_cache is None and os.path.exists(ANCHOR_PATH):
        try:
            d = torch.load(ANCHOR_PATH, map_location="cpu", weights_only=False)
            _anchor_cache = (d["X"], d["Y"], d["Z"])
            print(f"[anchor] rehearsal active: {len(_anchor_cache[0])} "
                  f"supervised samples mixed into batches")
        except Exception as exc:  # noqa: BLE001
            print(f"[anchor] unavailable: {exc}")
            _anchor_cache = False
    return _anchor_cache or None


def make_batch(replay, cfg):
    """Minibatch = fresh self-play positions + supervised rehearsal slice."""
    a = load_anchor()
    n_anchor = int(cfg.batch_size * cfg.anchor_fraction) if a is not None else 0
    n_replay = cfg.batch_size - n_anchor
    idxs = random.sample(range(len(replay)), min(n_replay, len(replay)))
    batch = [replay[i] for i in idxs]
    while len(batch) < n_replay:
        batch.append(random.choice(replay))
    if n_anchor:
        j = np.random.randint(0, len(a[0]), size=n_anchor)
        for k in j:
            batch.append((a[0][int(k)], [int(a[1][k])],
                          np.array([1.0], dtype=np.float32),
                          float(a[2][k])))
    return batch


def main():
    parser = argparse.ArgumentParser(description="Train a small network to play chess.")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override the number of training iterations.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume from.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a tiny training run to verify the pipeline.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "auto"],
                        help="Device to train on (default: cpu).")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel self-play worker processes (0 = sequential).")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override the base learning rate.")
    parser.add_argument("--sims", type=int, default=None,
                        help="Override MCTS simulations per self-play move.")
    parser.add_argument("--games", type=int, default=None,
                        help="Override self-play games per iteration.")
    parser.add_argument("--train-steps", type=int, default=None,
                        help="Override gradient steps per iteration.")
    parser.add_argument("--gate", action="store_true",
                        help="AlphaGo-Zero gating: self-play always uses the "
                             "best weights so far; new weights are promoted "
                             "only if they win a head-to-head match.")
    parser.add_argument("--gate-interval", type=int, default=10,
                        help="Iterations between gating matches.")
    parser.add_argument("--gate-games", type=int, default=12,
                        help="Games per gating match.")
    parser.add_argument("--gate-sims", type=int, default=100,
                        help="MCTS simulations per move in gating matches.")
    parser.add_argument("--gate-threshold", type=float, default=0.55,
                        help="Score the candidate must reach to be promoted.")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Frozen reference checkpoint; a match against it "
                             "is logged at every gating point.")
    args = parser.parse_args()

    cfg = Config()
    if args.iterations is not None:
        cfg.num_iterations = args.iterations
    if args.smoke:
        cfg.games_per_iteration = 2
        cfg.train_steps = 5
        cfg.num_iterations = 1
        cfg.num_simulations = 25
        cfg.temperature_moves = 5
        cfg.batch_size = 64
        cfg.checkpoint_interval = 1
        cfg.eval_games = 2
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.sims is not None:
        cfg.num_simulations = args.sims
    if args.games is not None:
        cfg.games_per_iteration = args.games
    if args.train_steps is not None:
        cfg.train_steps = args.train_steps

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                       action_planes=cfg.policy_size // 64).to(device)
    replay = []
    optimizer = make_optimizer(net, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_iterations, eta_min=cfg.lr_min)
    start_iter = 0

    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=str(device))
        net.load_state_dict(ckpt["model_state_dict"])
        opt_state = ckpt.get("optimizer_state_dict")
        if opt_state:
            optimizer.load_state_dict(opt_state)
        sched_state = ckpt.get("scheduler_state_dict")
        if sched_state:
            scheduler.load_state_dict(sched_state)
        start_iter = ckpt["iteration"] + 1
        print(f"Resumed from iteration {start_iter}")
        if os.path.exists(REPLAY_PATH):
            try:
                replay = torch.load(REPLAY_PATH, map_location="cpu",
                                    weights_only=False)
                print(f"Restored replay buffer: {len(replay)} positions")
            except Exception as exc:  # noqa: BLE001
                print(f"Could not restore replay buffer: {exc}")
                replay = []

    print(f"Device: {device} | Parameters: {count_parameters(net):,}")
    print(cfg)

    import copy
    from evaluate import match_nets

    # `best_state` is what generates self-play data. Without --gate it simply
    # tracks the live network, which reproduces the old behaviour exactly.
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    best_iter = start_iter
    baseline_net = None
    if args.baseline and not os.path.exists(args.baseline):
        print(f"WARNING: baseline {args.baseline!r} not found "
              f"(cwd {os.getcwd()}) - no baseline match will be logged")
    if args.baseline and os.path.exists(args.baseline):
        baseline_net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                                    action_planes=cfg.policy_size // 64)
        baseline_net.load_state_dict(
            load_checkpoint(args.baseline, map_location="cpu")["model_state_dict"])
        baseline_net.eval()
        print(f"Baseline for reporting: {args.baseline}")
    if args.gate:
        print(f"Gating ON: promote at >={args.gate_threshold:.0%} over "
              f"{args.gate_games} games @ {args.gate_sims} sims, "
              f"every {args.gate_interval} iterations")

    pool = None
    if args.workers > 0:
        # Workers always run MCTS on CPU (per-position evals are faster there);
        # only the batched training math uses the main `device`.
        pool = mp.Pool(args.workers, initializer=_worker_init,
                       initargs=("cpu", cfg))
        print(f"Self-play: {args.workers} parallel workers")

    def submit_selfplay():
        state = best_state if args.gate else             {k: v.cpu() for k, v in net.state_dict().items()}
        return pool.map_async(_worker_play_one, [state] * cfg.games_per_iteration)

    pending = submit_selfplay() if pool is not None else None
    noted_pipeline = False

    for it in range(start_iter, cfg.num_iterations):
        # ---- collect self-play results (played with previous weights) ---
        t0 = time.time()
        try:
            if pool is not None:
                results = pending.get()
            else:
                play_dev = torch.device("cpu")
                results = [play_one_game(net, play_dev, cfg)
                           for _ in range(cfg.games_per_iteration)]
        except Exception as exc:  # noqa: BLE001
            print(f"[iter {it}] worker pool failed ({exc}); recreating...")
            if pool is not None:
                pool.terminate()
                pool = mp.Pool(args.workers, initializer=_worker_init,
                               initargs=("cpu", cfg))
            results = generate_games(net, device, cfg, cfg.games_per_iteration,
                                     pool=pool)
        samples = []
        for g in results:
            samples.extend(g)
        replay.extend(samples)
        if len(replay) > cfg.replay_buffer_size:
            replay = replay[-cfg.replay_buffer_size:]
        sp_time = time.time() - t0
        print(f"[iter {it}] self-play: {len(samples)} positions "
              f"(buffer {len(replay)}) in {sp_time:.1f}s")
        maybe_save_replay(replay, it)

        # ---- launch NEXT self-play now; it overlaps GPU training below --
        if pool is not None:
            pending = submit_selfplay()
            if not noted_pipeline:
                print("[pipeline] self-play overlaps GPU training "
                      "(self-play weights lag one iteration)")
                noted_pipeline = True

        # ---- train -----------------------------------------------------
        if len(replay) < cfg.batch_size:
            print(f"[iter {it}] buffer too small, skipping training.")
            continue

        net.train()
        t0 = time.time()
        total_loss = 0.0
        for _ in range(cfg.train_steps):
            batch = make_batch(replay, cfg)
            loss, ploss, vloss = train_step(net, optimizer, batch, device, cfg)
            total_loss += loss
        scheduler.step()
        avg_loss = total_loss / cfg.train_steps
        print(f"[iter {it}] train: {cfg.train_steps} steps, avg loss {avg_loss:.4f} "
              f"(lr {scheduler.get_last_lr()[0]:.2e}) in {time.time() - t0:.1f}s")

        # ---- gate: promote only if the candidate actually got stronger --
        if args.gate and (it + 1) % args.gate_interval == 0:
            net.eval()
            cand = copy.deepcopy(net).cpu().eval()
            best_net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                                    action_planes=cfg.policy_size // 64)
            best_net.load_state_dict(best_state)
            best_net.eval()
            t0 = time.time()
            score = match_nets(cand, best_net, torch.device("cpu"), cfg,
                               num_games=args.gate_games,
                               num_sims=args.gate_sims, seed=it)
            promoted = score >= args.gate_threshold
            if promoted:
                best_state = {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()}
                best_iter = it + 1
                torch.save({"model_state_dict": best_state, "iteration": it + 1},
                           "checkpoints/best.pt")
            print(f"[iter {it}] gate: candidate {score:.1%} vs best "
                  f"(iter {best_iter}) -> "
                  f"{'PROMOTED' if promoted else 'rejected'} "
                  f"in {time.time() - t0:.0f}s", flush=True)

            if baseline_net is not None:
                t0 = time.time()
                bscore = match_nets(cand, baseline_net, torch.device("cpu"), cfg,
                                    num_games=args.gate_games,
                                    num_sims=args.gate_sims, seed=10_000 + it)
                print(f"[iter {it}] vs baseline: {bscore:.1%} "
                      f"in {time.time() - t0:.0f}s", flush=True)

        # ---- evaluate + checkpoint -------------------------------------
        if (it + 1) % cfg.checkpoint_interval == 0:
            net.eval()
            results = evaluate_vs_baseline(net, device, cfg, opponent="random",
                                           num_games=cfg.eval_games)
            print(f"[iter {it}] eval vs random: "
                  f"{{'win': {results['win']}, 'loss': {results['loss']}, "
                  f"'draw': {results['draw']}}} "
                  f"(score {results['score']:.0%})")

            path = f"checkpoints/iter_{it + 1}.pt"
            save_checkpoint(net, optimizer, scheduler, it, path)
            torch.save({
                "model_state_dict": best_state if args.gate else net.state_dict(),
                "iteration": it,
            }, "checkpoints/latest.pt")
            print(f"[iter {it}] saved {path}")

    if pool is not None:
        if pending is not None:
            try:
                for g in pending.get():
                    replay.extend(g)
                maybe_save_replay(replay, cfg.num_iterations - 1, interval=1)
            except Exception:  # noqa: BLE001
                pass
        pool.close()
        pool.join()
    print("Training complete.")


if __name__ == "__main__":
    main()
