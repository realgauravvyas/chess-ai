"""Supervised pretraining on real human games (Lichess database).

Downloads one monthly PGN dump, streams it through python-chess, and trains
the same policy+value network used for self-play:

    policy target: the move actually played (cross-entropy)
    value target:  game result from the side-to-move perspective (MSE)

The resulting checkpoint is saved as checkpoints/iter_<N>.pt so the normal
self-play pipeline (train.py / run_training.py) can resume straight from it.

Usage:
    python pretrain_supervised.py                       # defaults below
    python pretrain_supervised.py --max-positions 600000
"""
import argparse
import io
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zstandard as zstd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import chess          # noqa: E402
import chess.pgn      # noqa: E402

from config import Config                     # noqa: E402
from model import AlphaZeroNet, count_parameters  # noqa: E402
from move_encoding import encode_move         # noqa: E402
from utils import board_to_planes, result_to_z  # noqa: E402

DATA_DIR = ROOT / "data"
DEFAULT_URLS = ("https://database.lichess.org/standard/"
                "lichess_db_standard_rated_2013-01.pgn.zst,"
                "https://database.lichess.org/standard/"
                "lichess_db_standard_rated_2013-02.pgn.zst")


def download(url, dest: Path):
    """Download with simple resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[data] using existing {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    req = urllib.request.Request(url, headers={"User-Agent": "chess-ai-pretrain"})
    print(f"[data] downloading {url}")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                rate = done / 1e6 / max(1e-9, time.time() - t0)
                print(f"\r[data] {done/1e6:7.1f}/{total/1e6:.1f} MB "
                      f"({pct:4.1f}%) {rate:.1f} MB/s", end="", flush=True)
    print(f"\n[data] downloaded in {time.time()-t0:.0f}s")


def _elo_ok(game, min_elo):
    """True if both players' Elo (when present) is at least min_elo."""
    if min_elo <= 0:
        return True
    try:
        w = int(game.headers.get("WhiteElo", "0") or 0)
        b = int(game.headers.get("BlackElo", "0") or 0)
    except ValueError:
        return False
    return w >= min_elo and b >= min_elo


def iter_samples(pgn_path: Path, max_positions, max_games, seed_cap=3000,
                 min_elo=0):
    """Yield (planes_u8, move_index, z, opening_seed_or_None).

    Also collects rare-ish opening positions (plies 5..8) via reservoir
    sampling into `seed_state` for the self-play curriculum.
    """
    opened = time.time()
    n_games = n_pos = skipped = 0
    seen = [0]  # candidates counter for reservoir

    def consider_seed(fen):
        seen[0] += 1
        if len(seed_out) < seed_cap:
            seed_out.append(fen)
        else:
            j = np.random.randint(seen[0])
            if j < seed_cap:
                seed_out[j] = fen

    seed_out = []
    with open(pgn_path, "rb") as fh:
        # Plain .pgn files must not go through the zstd reader: every read
        # would raise, and the `continue` below would spin forever.
        is_zst = fh.read(4) == bytes([0x28, 0xb5, 0x2f, 0xfd])
        fh.seek(0)
        if is_zst:
            reader = zstd.ZstdDecompressor().stream_reader(fh)
        else:
            reader = fh
        with reader:
            stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            consecutive_errors = 0
            while n_pos < max_positions and n_games < max_games:
                try:
                    game = chess.pgn.read_game(stream)
                    consecutive_errors = 0
                except Exception:  # noqa: BLE001
                    skipped += 1
                    consecutive_errors += 1
                    if consecutive_errors >= 50:
                        print(f"[parse] giving up on {pgn_path.name}: "
                              f"{consecutive_errors} consecutive read errors")
                        break
                    continue
                if game is None:
                    break
                n_games += 1
                result = game.headers.get("Result", "*")
                if result not in ("1-0", "0-1", "1/2-1/2"):
                    continue
                if not _elo_ok(game, min_elo):
                    continue
                board = game.board()
                try:
                    for move in game.mainline_moves():
                        if move not in board.legal_moves or n_pos >= max_positions:
                            break
                        seed_fen = None
                        if 5 <= board.ply() <= 8:
                            seed_fen = board.fen()
                            consider_seed(seed_fen)
                        yield (board_to_planes(board).astype(np.uint8),
                               encode_move(move),
                               result_to_z(result, board.turn),
                               seed_fen)
                        n_pos += 1
                        board.push(move)
                except Exception:  # noqa: BLE001
                    skipped += 1
                if n_games % 5000 == 0:
                    rate = n_games / max(1e-9, time.time() - opened)
                    print(f"[parse] {n_games} games, {n_pos} positions "
                          f"({rate:.0f} games/s)", flush=True)
    print(f"[parse] done: {n_games} games -> {n_pos} positions "
          f"({skipped} skipped) in {time.time()-opened:.0f}s")


def build_dataset(pgn_path, max_positions, max_games, min_elo=0):
    planes_list, idx_list, z_list, seeds = [], [], [], []
    for planes, idx, z, seed_fen in iter_samples(pgn_path, max_positions,
                                                 max_games, seed_cap=3000,
                                                 min_elo=min_elo):
        planes_list.append(planes)
        idx_list.append(idx)
        z_list.append(z)
        if seed_fen is not None:
            seeds.append(seed_fen)
    if not planes_list:
        return (np.empty((0, 18, 8, 8), dtype=np.uint8),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32), [])
    X = np.stack(planes_list)
    Y = np.array(idx_list, dtype=np.int64)
    Z = np.array(z_list, dtype=np.float32)
    if len(seeds) > 3000:  # thin down to a diverse 3k
        sel = np.random.default_rng(0).choice(len(seeds), 3000, replace=False)
        seeds = [seeds[i] for i in sel]
    return X, Y, Z, seeds


def _parse_worker(job):
    """Parse one PGN file (runs in a worker process for parallelism)."""
    path, max_pos, max_games, seed_cap, min_elo = job
    return build_dataset(Path(path), max_pos, max_games,
                         min_elo=min_elo)


def train(X, Y, Z, cfg, epochs, batch_size, lr, device):
    net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                       action_planes=cfg.policy_size // 64).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(X)
    steps_per_epoch = max(1, n // batch_size)
    total_steps = steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-4)

    rng = np.random.default_rng(0)
    step = 0
    net.train()
    for epoch in range(epochs):
        order = rng.permutation(n)
        running = 0.0
        for s in range(steps_per_epoch):
            idx = order[s * batch_size:(s + 1) * batch_size]
            xb = torch.from_numpy(np.asarray(X[idx], dtype=np.float32)).to(device)
            yb = torch.from_numpy(Y[idx]).to(device)
            zb = torch.from_numpy(Z[idx]).to(device)

            logits, value = net(xb)
            p_loss = F.cross_entropy(logits, yb)
            v_loss = F.mse_loss(value.view(-1), zb)
            loss = p_loss + v_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += float(loss.item())
            step += 1
            if step % 100 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"[train] epoch {epoch+1}/{epochs} step {step}/{total_steps}"
                      f" loss {running/100:.4f} (p {p_loss.item():.3f}"
                      f" v {v_loss.item():.3f}) lr {lr_now:.2e}", flush=True)
                running = 0.0
    return net


def quick_eval(net, games=8, sims=30):
    """Play a few games vs random to sanity-check the pretrained net."""
    import random
    import torch as _t
    from evaluate import play_eval_game

    cfg = Config()
    cfg.num_simulations = sims
    net.eval()
    from evaluate import evaluate_vs_baseline
    random.seed(1)
    tally = evaluate_vs_baseline(net, _t.device("cpu"), cfg, "random",
                                 num_games=games, num_sims=sims)
    score = 100.0 * tally["score"]
    print(f"[eval] vs random ({tally['played']} games): "
          f"+{tally['win']} ={tally['draw']} -{tally['loss']} score {score:.0f}%")
    return score


def main():
    ap = argparse.ArgumentParser(description="Supervised pretraining from Lichess PGN")
    ap.add_argument("--urls", default=DEFAULT_URLS,
                    help="comma-separated Lichess monthly .pgn.zst URLs")
    ap.add_argument("--data-file", default=None,
                    help="use an already-downloaded .pgn/.pgn.zst instead of downloading")
    ap.add_argument("--max-positions", type=int, default=900_000)
    ap.add_argument("--max-games", type=int, default=80_000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-iteration", type=int, default=200,
                    help="save checkpoint as iter_<N>.pt so self-play resumes from it")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--min-elo", type=int, default=1750,
                    help="only keep games where both players are rated >= this")
    ap.add_argument("--save-anchor", type=int, default=40000,
                    help="save a random supervised subset for rehearsal during "
                         "self-play fine-tuning (0 disables)")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    print(f"[pretrain] device: {device}")
    cfg = Config()

    data_files = []
    if args.data_file:
        data_files = [Path(args.data_file)]
    else:
        for url in args.urls.split(","):
            url = url.strip()
            path = DATA_DIR / os.path.basename(url)
            download(url, path)
            data_files.append(path)

    t0 = time.time()
    n_files = len(data_files)
    per_pos = args.max_positions // n_files
    per_games = args.max_games // n_files
    jobs = [(df, per_pos, per_games, 3000 // max(1, n_files), args.min_elo)
            for df in data_files]

    def merge(results):
        Xs, Ys, Zs, seeds = [], [], [], []
        for X, Y, Z, s in results:
            Xs.append(X); Ys.append(Y); Zs.append(Z); seeds.extend(s)
        return (np.concatenate(Xs), np.concatenate(Ys),
                np.concatenate(Zs), seeds)

    if len(jobs) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(len(jobs), os.cpu_count() or 2)) as pool:
            results = pool.map(_parse_worker, jobs)
        X, Y, Z, seeds = merge(results)
    else:
        X, Y, Z, seeds = merge([_parse_worker(jobs[0])])
    print(f"[data] dataset ready: {len(X)} positions in {time.time()-t0:.0f}s")

    if seeds:
        import json
        seeds_path = ROOT / "data" / "opening_seeds.json"
        seeds_path.parent.mkdir(exist_ok=True)
        seeds_path.write_text(json.dumps({"seeds": seeds}), encoding="utf-8")
        print(f"[data] wrote {len(seeds)} opening seed positions -> {seeds_path.name}")

    if args.save_anchor > 0 and len(X) > args.save_anchor:
        sel = np.random.default_rng(1).choice(len(X), args.save_anchor,
                                              replace=False)
        anchor_path = ROOT / "checkpoints" / "anchor_data.pt"
        anchor_path.parent.mkdir(exist_ok=True)
        torch.save({"X": X[sel], "Y": Y[sel], "Z": Z[sel]}, anchor_path)
        print(f"[data] saved rehearsal anchor ({args.save_anchor} samples) "
              f"-> {anchor_path.name}")

    net = train(X, Y, Z, cfg, args.epochs, args.batch_size, args.lr, device)

    quick_eval(net)

    out_dir = ROOT / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"iter_{args.out_iteration}.pt"
    torch.save({
        "model_state_dict": net.state_dict(),
        "iteration": args.out_iteration,
        "pretrained": True,
    }, out_path)
    print(f"[save] pretrained checkpoint -> {out_path}")
    print("Pretraining complete.")


if __name__ == "__main__":
    main()
