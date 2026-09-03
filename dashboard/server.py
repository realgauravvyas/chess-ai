"""Browser dashboard server for the chess AI project.

Serves an interactive UI: play against trained checkpoints with the mouse,
inspect training curves, browse checkpoints, and run quick evaluations.

Run:  python dashboard/server.py [--port 5000]
"""
import argparse
import glob
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import chess
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config                      # noqa: E402
from evaluate import play_eval_game, net_points  # noqa: E402
from model import AlphaZeroNet, count_parameters  # noqa: E402
from mcts import MCTS                          # noqa: E402
from utils import load_checkpoint              # noqa: E402

LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "checkpoints"

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"),
            static_url_path="/static")

_model_lock = threading.Lock()
_net_cache = {}          # checkpoint path -> loaded net (CPU)
NET_CACHE_MAX = 4
_eval_job = {"running": False, "done": 0, "total": 0,
             "results": None, "error": None}
_teach_job = {"running": False, "step": "", "results": None, "error": None}
MY_GAMES = ROOT / "data" / "my_games.pgn"
DISABLED_FLAG = LOG_DIR / "dashboard_disabled.flag"

# ---- cheap caches for hot endpoints ---------------------------------
# A training iteration takes ~2.5 min, so allow a generous gap before
# declaring the run dead.
LIVE_WINDOW_SECONDS = 420
_STATS_CACHE = {"mtime": None, "data": None}
_PARAM_CACHE = None
_LOGCKPT_LOCK = threading.Lock()


# ----------------------------------------------------------------- helpers
def load_net(path):
    with _model_lock:
        net = _net_cache.get(path)
        if net is None:
            cfg = Config()
            net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                               action_planes=cfg.policy_size // 64)
            blob = load_checkpoint(path)
            state = blob["model_state_dict"] if isinstance(blob, dict) \
                and "model_state_dict" in blob else blob
            net.load_state_dict(state)
            net.eval()
            _net_cache[path] = net
            while len(_net_cache) > NET_CACHE_MAX:
                _net_cache.pop(next(iter(_net_cache)))
        return net


def is_model_checkpoint(name):
    """Only true weight files — not anchor/replay data blobs."""
    return (name.startswith("iter_") or name == "latest.pt") and \
        name.endswith(".pt")


def latest_checkpoint():
    """The best model to play by default.

    Searches checkpoints/ and every runs/<name>/checkpoints/. A gated run
    writes its winner to best.pt, and a gate win is the strongest evidence
    available, so those are preferred. Otherwise the most recently written
    iter_N.pt wins: iteration numbers are not comparable across runs (v5
    reached iter_500 and v7 only iter_400, but v7 is the later, better run),
    so ordering is by modification time.
    """
    run_dirs = [CKPT_DIR] + [Path(d) for d in
                             glob.glob(str(ROOT / "runs" / "*" / "checkpoints"))]

    bests = [p for d in run_dirs
             for p in glob.glob(str(d / "best.pt"))]
    if bests:
        return max(bests, key=os.path.getmtime)

    iters = [p for d in run_dirs
             for p in glob.glob(str(d / "iter_*.pt"))
             if re.match(r"iter_\d+\.pt$", os.path.basename(p))]
    if iters:
        return max(iters, key=os.path.getmtime)

    latests = [p for d in run_dirs for p in glob.glob(str(d / "latest.pt"))]
    return max(latests, key=os.path.getmtime) if latests else None


def legal_list(board):
    return [{"uci": m.uci(), "san": board.san(m)} for m in board.legal_moves]


def game_status(board):
    out = board.outcome(claim_draw=True)
    if out is None:
        resp = {"status": "playing", "winner": None}
    else:
        term = out.termination.name.lower()
        resp = {"status": term, "winner": ("white" if out.winner else "black") if out.winner is not None else None}
    resp["in_check"] = board.is_check()
    if board.is_check():
        king_sq = board.king(board.turn)
        resp["check_square"] = chess.square_name(king_sq) if king_sq is not None else None
    else:
        resp["check_square"] = None
    return resp


def _training_logs():
    """All training-related log files, oldest first."""
    files = []
    for pattern in ("train*.log", "finetune*.log", "pretrain*.log",
                    "pipeline*.log"):
        files.extend(glob.glob(str(LOG_DIR / pattern)))
    return sorted(set(files), key=os.path.getmtime)


def parse_stats():
    """Parse all training log files into merged per-iteration series."""
    log_files = _training_logs()
    loss, lr, sp_pos, sp_time, buffer_, evals = {}, {}, {}, {}, {}, {}
    gate, baseline = {}, {}
    saved = set()

    rx_sp = re.compile(r"\[iter (\d+)\] self-play: (\d+) positions \(buffer (\d+)\) in ([\d.]+)s")
    rx_tr = re.compile(r"\[iter (\d+)\] train: \d+ steps, avg loss ([\d.]+) \(lr ([\deE.+-]+)\)")
    rx_ev = re.compile(r"\[iter (\d+)\] eval vs (?:random|greedy): "
                       r"\{(.*?)\} \(score ([\d.]+)%\)")
    rx_sv = re.compile(r"\[iter (\d+)\] saved checkpoints/iter_(\d+)\.pt")
    rx_gate = re.compile(r"\[iter (\d+)\] gate: candidate ([\d.]+)% vs best "
                         r"\(iter (\d+)\) -> (PROMOTED|rejected)")
    rx_base = re.compile(r"\[iter (\d+)\] vs baseline: ([\d.]+)%")

    for path in log_files:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            m = rx_sp.search(line)
            if m:
                it = int(m.group(1))
                sp_pos[it] = int(m.group(2))
                buffer_[it] = int(m.group(3))
                sp_time[it] = float(m.group(4))
                continue
            m = rx_tr.search(line)
            if m:
                it = int(m.group(1))
                loss[it] = float(m.group(2))
                lr[it] = float(m.group(3))
                continue
            m = rx_ev.search(line)
            if m:
                it = int(m.group(1))
                pairs = dict(re.findall(r"'([^']+)':\s*(\d+)", m.group(2)))
                # New runs log net-perspective win/loss/draw; older runs
                # logged raw white-perspective PGN result counts.
                evals[it] = {
                    "w": int(pairs.get("win", pairs.get("1-0", 0))),
                    "b": int(pairs.get("loss", pairs.get("0-1", 0))),
                    "d": int(pairs.get("draw", pairs.get("1/2-1/2", 0))),
                    "score": float(m.group(3)),
                }
                continue
            m = rx_gate.search(line)
            if m:
                gate[int(m.group(1))] = {
                    "score": float(m.group(2)),
                    "best_iter": int(m.group(3)),
                    "promoted": m.group(4) == "PROMOTED",
                }
                continue
            m = rx_base.search(line)
            if m:
                baseline[int(m.group(1))] = float(m.group(2))
                continue
            m = rx_sv.search(line)
            if m:
                saved.add(int(m.group(2)))

    def series(d):
        return [[k, d[k]] for k in sorted(d)]

    latest_iter = max([*loss.keys(), *sp_pos.keys(), *evals.keys()],
                      default=None)
    return {
        "loss": series(loss),
        "lr": series(lr),
        "selfplay_positions": series(sp_pos),
        "selfplay_seconds": series(sp_time),
        "buffer": series(buffer_),
        "evals": [{"iter": k, **v} for k, v in sorted(evals.items())],
        "gate": [{"iter": k, **v} for k, v in sorted(gate.items())],
        "baseline": series(baseline),
        "promotions": sum(1 for v in gate.values() if v["promoted"]),
        "saved_iters": sorted(saved),
        "latest_iter": latest_iter,
        "runs_found": len(log_files),
    }


# ------------------------------------------------------------------- routes
@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent / "static", "index.html")


@app.route("/api/stats")
def api_stats():
    with _LOGCKPT_LOCK:
        logs = _training_logs()
        sig = (logs[-1], os.path.getmtime(logs[-1])) if logs else None
        if _STATS_CACHE["mtime"] != sig:
            stats = parse_stats()
            cfg = Config()
            global _PARAM_CACHE
            if _PARAM_CACHE is None:
                net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                                   action_planes=cfg.policy_size // 64)
                _PARAM_CACHE = count_parameters(net)
            stats["model_info"] = {
                "parameters": _PARAM_CACHE,
                "filters": cfg.filters,
                "res_blocks": cfg.res_blocks,
                "device": "cpu",
                "latest_iter": stats["latest_iter"],
            }
            _STATS_CACHE["mtime"] = sig
            _STATS_CACHE["data"] = stats

        # Liveness must NOT be cached against log mtime: the cache key only
        # changes when the log is written, so a cached "alive" would survive
        # forever once training stops. Recompute it on every request.
        data = dict(_STATS_CACHE["data"])
        alive, last_activity, age = False, None, None
        if logs:
            mtime = os.path.getmtime(logs[-1])
            age = time.time() - mtime
            alive = age < LIVE_WINDOW_SECONDS
            last_activity = time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime(mtime))
        data["training_alive"] = alive
        data["last_activity"] = last_activity
        data["seconds_since_log"] = round(age) if age is not None else None
        return jsonify(data)


@app.route("/api/checkpoints")
def api_checkpoints():
    items = []
    paths = glob.glob(str(CKPT_DIR / "*.pt"))
    # active runs keep their checkpoints in runs/<name>/checkpoints/
    paths += glob.glob(str(ROOT / "runs" / "*" / "checkpoints" / "*.pt"))
    for p in sorted(paths, key=lambda x: -os.path.getmtime(x)):
        name = os.path.basename(p)
        if not (is_model_checkpoint(name) or name == "best.pt"):
            continue
        run = Path(p).parent.parent.name
        if run != ROOT.name:
            name = f"{run}/{name}"
        st = os.stat(p)
        items.append({"name": name, "path": p,
                      "size_mb": round(st.st_size / 1e6, 2),
                      "modified": time.strftime("%Y-%m-%d %H:%M",
                                                time.localtime(st.st_mtime))})
    return jsonify(items)


@app.route("/api/newgame", methods=["POST"])
def api_newgame():
    body = request.get_json(force=True, silent=True) or {}
    color = body.get("color", "white")
    if color not in ("white", "black"):
        return jsonify({"error": "color must be white or black"}), 400
    board = chess.Board()
    return jsonify({"fen": board.fen(), "your_color": color,
                    "legal": legal_list(board), **game_status(board)})


@app.route("/api/legal_moves")
def api_legal_moves():
    fen = request.args.get("fen", "")
    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "invalid fen"}), 400
    return jsonify({"legal": legal_list(board), **game_status(board)})


@app.route("/api/human_move", methods=["POST"])
def api_human_move():
    body = request.get_json(force=True, silent=True) or {}
    fen, uci = body.get("fen"), body.get("uci")
    if not fen or not uci:
        return jsonify({"error": "fen and uci required"}), 400
    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "invalid fen"}), 400
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return jsonify({"error": f"bad uci '{uci}'"}), 400
    if move not in board.legal_moves:
        return jsonify({"error": f"illegal move {uci}"}), 400
    san = board.san(move)
    board.push(move)
    return jsonify({"fen": board.fen(), "san": san,
                    "legal": [] if board.is_game_over() else legal_list(board),
                    **game_status(board)})


@app.route("/api/model_move", methods=["POST"])
def api_model_move():
    body = request.get_json(force=True, silent=True) or {}
    fen = body.get("fen")
    apply_move = bool(body.get("apply", True))
    sims = max(10, min(int(body.get("sims", 200)), 1600))
    ckpt = body.get("checkpoint") or latest_checkpoint()
    if not ckpt or not os.path.exists(ckpt):
        return jsonify({"error": "checkpoint not found"}), 400
    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "invalid fen"}), 400
    if board.is_game_over():
        return jsonify({"error": "game is over"}), 400

    net = load_net(ckpt)
    cfg = Config()
    # No Dirichlet noise: that is a self-play exploration device, and leaving
    # it on made every dashboard move 25% random.
    mcts = MCTS(net, torch.device("cpu"), cfg.c_puct,
                dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    t0 = time.time()
    with _model_lock:
        probs = mcts.get_action_probs(board, sims)
    value = float(mcts.root_value)
    visits = mcts.last_visits
    think_s = round(time.time() - t0, 2)

    if not probs:
        return jsonify({"error": "no legal moves"}), 400
    best = max(probs, key=probs.get)
    san = board.san(best)
    total = sum(visits.values()) or 1
    top = [{"uci": mv.uci(), "san": board.san(mv), "p": round(c / total, 3)}
           for mv, c in sorted(visits.items(), key=lambda kv: -kv[1])[:4]]

    # `value` is from the side-to-move's (i.e. the model's) perspective.
    # The eval bar is drawn white-up, so give the UI a white-relative copy.
    value_white = value if board.turn == chess.WHITE else -value
    resp = {"move": best.uci(), "san": san, "value": round(value, 4),
            "value_white": round(value_white, 4),
            "top_moves": top, "think_seconds": think_s,
            "pv": mcts.pv_line(board),
            "checkpoint": os.path.basename(ckpt)}
    if apply_move:
        board.push(best)
        resp.update({"fen": board.fen(),
                     "legal": [] if board.is_game_over() else legal_list(board)})
    resp.update(game_status(board))
    return jsonify(resp)


def _eval_worker(path, games, sims):
    try:
        net = load_net(path)
        cfg = Config()
        cfg.num_simulations = sims
        tally = {"win": 0, "draw": 0, "loss": 0}
        with _model_lock:
            for i in range(max(1, games // 2)):
                for net_white in (True, False):
                    r = play_eval_game(net, torch.device("cpu"), cfg, "random",
                                       num_sims=sims, net_plays_white=net_white,
                                       random_plies=2)
                    pts = net_points(r, net_white)
                    tally["win" if pts == 1.0 else
                          "draw" if pts == 0.5 else "loss"] += 1
                _eval_job["done"] = min(games, (i + 1) * 2)
        played = sum(tally.values())
        score = 100.0 * (tally["win"] + 0.5 * tally["draw"]) / played
        _eval_job["results"] = {**tally, "played": played,
                                "score": round(score, 1)}
    except Exception as exc:  # noqa: BLE001
        _eval_job["error"] = str(exc)
    finally:
        _eval_job["running"] = False


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    body = request.get_json(force=True, silent=True) or {}
    if _eval_job["running"]:
        return jsonify({"error": "evaluation already running"}), 409
    ckpt = body.get("checkpoint") or latest_checkpoint()
    if not ckpt or not os.path.exists(ckpt):
        return jsonify({"error": "checkpoint not found"}), 400
    games = max(2, min(int(body.get("games", 6)), 40))
    sims = max(10, min(int(body.get("sims", 30)), 200))
    _eval_job.update({"running": True, "done": 0, "total": games,
                      "results": None, "error": None})
    threading.Thread(target=_eval_worker, args=(ckpt, games, sims),
                     daemon=True).start()
    return jsonify({"started": True, "checkpoint": os.path.basename(ckpt),
                    "games": games, "sims": sims})


@app.route("/api/evaluate/status")
def api_eval_status():
    return jsonify(_eval_job)


# ------------------------------------------------------- teach-me loop
@app.route("/api/save_game", methods=["POST"])
def api_save_game():
    """Persist a game the user just played (list of UCI moves) as PGN."""
    import chess.pgn
    body = request.get_json(force=True, silent=True) or {}
    uci_moves = body.get("uci") or []
    if len(uci_moves) < 2:
        return jsonify({"error": "game too short to learn from"}), 400
    board = chess.Board()
    node = None
    game = chess.pgn.Game()
    node = game
    for u in uci_moves:
        try:
            mv = chess.Move.from_uci(u)
        except ValueError:
            return jsonify({"error": f"bad uci {u}"}), 400
        if mv not in board.legal_moves:
            return jsonify({"error": f"illegal move {u}"}), 400
        board.push(mv)
        node = node.add_main_variation(mv)
    game.headers["Result"] = body.get("result", "*")
    game.headers["White"] = "You" if body.get("you_are", "white") == "white" else "Model"
    game.headers["Black"] = "Model" if body.get("you_are", "white") == "white" else "You"
    MY_GAMES.parent.mkdir(exist_ok=True)
    with open(MY_GAMES, "a", encoding="utf-8") as fh:
        print(game, file=fh, end="\n\n")
    return jsonify({"saved": True, "moves": len(uci_moves),
                    "file": MY_GAMES.name})


def _teach_worker(ckpt_path):
    try:
        _teach_job["step"] = "parsing your games"
        from pretrain_supervised import build_dataset
        X, Y, Z, _seeds = build_dataset(MY_GAMES, max_positions=8000,
                                        max_games=500)
        if len(X) == 0:
            raise RuntimeError(
                "no finished games to learn from - only games saved with a "
                "final result (win/loss/draw) can be trained on")
        cfg = Config()
        net = load_net(ckpt_path)

        _teach_job["step"] = "training on your moves"
        optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4)
        import torch.nn.functional as F
        rng = np.random.default_rng(0)
        steps = max(20, min(400, (len(X) * 1) // 128))
        net.train()
        for s in range(steps):
            idx = rng.integers(0, len(X), size=128)
            xb = torch.from_numpy(np.asarray(X[idx], dtype=np.float32))
            yb = torch.from_numpy(Y[idx])
            zb = torch.from_numpy(Z[idx])
            logits, value = net(xb)
            loss = F.cross_entropy(logits, yb) + F.mse_loss(value.view(-1), zb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if s % 25 == 0:
                _teach_job["step"] = f"training ({s}/{steps} steps)"

        m = re.match(r"iter_(\d+)\.pt", os.path.basename(ckpt_path))
        nxt = (int(m.group(1)) + 1) if m else 1
        out = CKPT_DIR / f"iter_{nxt}.pt"
        torch.save({"model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iteration": nxt}, out)
        with _model_lock:
            _net_cache.pop(ckpt_path, None)
            _net_cache.pop(str(out), None)
        _teach_job["results"] = {"saved": out.name, "positions": len(X),
                                 "steps": steps}
    except Exception as exc:  # noqa: BLE001
        _teach_job["error"] = str(exc)
    finally:
        _teach_job["running"] = False


@app.route("/api/teach", methods=["POST"])
def api_teach():
    if _teach_job["running"]:
        return jsonify({"error": "already teaching"}), 409
    if not MY_GAMES.exists() or MY_GAMES.stat().st_size < 60:
        return jsonify({"error": "no saved games yet — play and save one first"}), 400
    ckpt = latest_checkpoint()
    if not ckpt:
        return jsonify({"error": "no checkpoint available"}), 400
    _teach_job.update({"running": True, "step": "starting",
                       "results": None, "error": None})
    threading.Thread(target=_teach_worker, args=(ckpt,), daemon=True).start()
    return jsonify({"started": True})


@app.route("/api/teach/status")
def api_teach_status():
    return jsonify(_teach_job)


# ------------------------------------------------------- lifecycle
@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    """Graceful shutdown. Writes a flag so the watchdog stays down."""
    try:
        DISABLED_FLAG.write_text("shutdown requested")
    except OSError:
        pass
    shutdown_fn = request.environ.get("werkzeug.server.shutdown")

    def _stop():
        time.sleep(0.3)
        if shutdown_fn:
            shutdown_fn()
        else:
            os._exit(0)

    threading.Thread(target=_stop, daemon=True).start()
    return jsonify({"shutting_down": True})


@app.route("/api/health")
def api_health():
    return jsonify({"ok": True})


# --------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Chess AI dashboard server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--no-flag-clear", action="store_true",
                        help="do not clear the disabled flag on startup")
    args = parser.parse_args()
    if not args.no_flag_clear:
        try:
            DISABLED_FLAG.unlink()
        except OSError:
            pass
    print(f"Serving dashboard on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
