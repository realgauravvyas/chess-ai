"""Executable test suite for every module in the project.

Written after four rounds of ad-hoc review kept surfacing bugs one at a
time. Reading code finds different bugs than running it: the two worst
defects in this project (an evaluation metric that scored the wrong colour,
and a forensics tool with inverted parity) both sat in code that looked
fine and only ran in situations nobody exercised.

Every test here executes real code against a known answer. Fast by design -
no full games, no training runs - so it can be run after any change.

    python tests/test_suite.py
    python tests/test_suite.py -v      # show each passing case
"""
import sys
import traceback
from pathlib import Path

import chess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VERBOSE = "-v" in sys.argv
_RESULTS = []


def check(name, cond, detail=""):
    _RESULTS.append((name, bool(cond), detail))
    if not cond:
        print(f"FAIL  {name}" + (f"\n        {detail}" if detail else ""))
    elif VERBOSE:
        print(f"pass  {name}")


def section(title):
    if VERBOSE:
        print(f"\n--- {title} ---")


# =====================================================================
def test_utils():
    section("utils: board encoding")
    from utils import board_to_planes, result_to_z, NUM_PLANES

    b = chess.Board()
    p = board_to_planes(b)
    check("planes shape is 18x8x8", p.shape == (NUM_PLANES, 8, 8), str(p.shape))
    check("planes are float32", p.dtype == np.float32, str(p.dtype))
    check("planes are binary", set(np.unique(p)).issubset({0.0, 1.0}))

    # plane 0 = white pawns; at the start they occupy rank 2 -> row index 1
    check("white pawns on row 1 at start", p[0][1].sum() == 8, str(p[0].sum()))
    check("black pawns on row 6 at start", p[6][6].sum() == 8, str(p[6].sum()))
    check("white king plane has exactly one square", p[5].sum() == 1)

    # castling planes are all-ones when the right exists
    check("all four castling planes set at start",
          all(p[i].sum() == 64 for i in (12, 13, 14, 15)))
    # a board where the king actually has somewhere to go
    nb = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    check("all castling rights present before the king moves",
          all(board_to_planes(nb)[i].sum() == 64 for i in (12, 13, 14, 15)))
    nb.push_uci("e1f1")            # king moves: white loses both rights
    q = board_to_planes(nb)
    check("white castling planes cleared after Kf1",
          q[12].sum() == 0 and q[13].sum() == 0,
          f"WK={q[12].sum()} WQ={q[13].sum()}")
    check("black castling planes survive Kf1",
          q[14].sum() == 64 and q[15].sum() == 64,
          f"BK={q[14].sum()} BQ={q[15].sum()}")

    # en passant
    eb = chess.Board()
    eb.push_uci("e2e4")
    e = board_to_planes(eb)
    check("en-passant plane set after a double push", e[16].sum() == 1,
          str(e[16].sum()))
    check("side-to-move plane 0 when black to move", e[17].sum() == 0)
    check("side-to-move plane 64 when white to move",
          board_to_planes(chess.Board())[17].sum() == 64)

    section("utils: result_to_z")
    cases = [("1-0", chess.WHITE, 1.0), ("1-0", chess.BLACK, -1.0),
             ("0-1", chess.WHITE, -1.0), ("0-1", chess.BLACK, 1.0),
             ("1/2-1/2", chess.WHITE, 0.0), ("1/2-1/2", chess.BLACK, 0.0)]
    for res, col, want in cases:
        got = result_to_z(res, col)
        check(f"result_to_z({res}, {'W' if col else 'B'}) == {want}",
              got == want, f"got {got}")


# =====================================================================
def test_move_encoding():
    section("move encoding")
    import move_encoding as me

    check("policy size is 4672", me.NUM_POLICY == 4672)
    check("round-trip self-test passes", me._self_test())

    # mirroring twice is the identity
    bad = [i for i in range(0, me.NUM_POLICY, 37)
           if me.mirror_action_index(me.mirror_action_index(i)) != i]
    check("mirror_action_index is an involution", not bad, f"failed at {bad[:5]}")

    # every legal move in a tactical position encodes uniquely
    b = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    idx = [me.encode_move(m) for m in b.legal_moves]
    check("no encoding collisions with castling available",
          len(idx) == len(set(idx)), f"{len(idx)} moves, {len(set(idx))} indices")

    # underpromotions occupy the dedicated planes
    pb = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
    for m in pb.legal_moves:
        if m.promotion and m.promotion != chess.QUEEN:
            plane = me.encode_move(m) % 73
            check(f"underpromotion to {chess.piece_name(m.promotion)} "
                  f"uses plane 64-72", 64 <= plane <= 72, f"plane {plane}")


# =====================================================================
def test_model():
    section("model")
    from config import Config
    from model import AlphaZeroNet, count_parameters

    cfg = Config()
    net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                       cfg.policy_size // 64)
    net.eval()
    check("parameter count is 760,717", count_parameters(net) == 760_717,
          f"{count_parameters(net):,}")

    x = torch.zeros(3, cfg.planes, 8, 8)
    with torch.no_grad():
        logits, value = net(x)
    check("policy output is (B, 4672)", tuple(logits.shape) == (3, 4672),
          str(tuple(logits.shape)))
    check("value output is (B, 1)", tuple(value.shape) == (3, 1),
          str(tuple(value.shape)))

    # Bound the value head with inputs large enough to saturate it. Zero
    # input proves nothing: an unbounded head still returns a small number
    # there, so the check would pass even with the tanh removed.
    torch.manual_seed(0)
    net.train()          # batch-norm in train mode amplifies the activations
    big = torch.randn(16, cfg.planes, 8, 8) * 25.0
    with torch.no_grad():
        _, v_big = net(big)
    net.eval()
    check("value stays inside [-1, 1] under large inputs",
          bool((v_big.abs() <= 1.0 + 1e-6).all()),
          f"max |v| = {float(v_big.abs().max()):.3f}")

    # Batch norm keeps a randomly initialised head far from saturation, so
    # "is it bounded" cannot detect a missing squashing function. Check the
    # mechanism instead: the head's output must equal tanh of its own
    # pre-activation.
    import torch.nn.functional as F
    vh = net.value_head
    trunk = torch.randn(4, cfg.filters, 8, 8)
    with torch.no_grad():
        out = vh(trunk)
        h = F.relu(vh.bn(vh.conv(trunk))).flatten(1)
        pre = vh.fc2(F.relu(vh.fc1(h)))
    check("value head applies tanh to its pre-activation",
          torch.allclose(out, torch.tanh(pre), atol=1e-6),
          f"max deviation {float((out - torch.tanh(pre)).abs().max()):.4g}")


# =====================================================================
def test_evaluate():
    section("evaluate: scoring")
    from evaluate import net_points

    # the bug that invalidated 159 logged measurements
    cases = [("1-0", True, 1.0), ("1-0", False, 0.0),
             ("0-1", True, 0.0), ("0-1", False, 1.0),
             ("1/2-1/2", True, 0.5), ("1/2-1/2", False, 0.5)]
    for res, white, want in cases:
        got = net_points(res, white)
        check(f"net_points({res}, net_white={white}) == {want}",
              got == want, f"got {got}")

    # a win as black must not be scored as a loss
    check("a black win scores 1.0, not 0.0", net_points("0-1", False) == 1.0)


# =====================================================================
def test_mcts():
    section("mcts")
    from config import Config
    from mcts import MCTS
    from model import AlphaZeroNet

    cfg = Config()
    net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                       cfg.policy_size // 64)
    net.eval()
    dev = torch.device("cpu")

    m = MCTS(net, dev, cfg.c_puct, dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    probs = m.get_action_probs(chess.Board(), 24)
    check("search returns a move distribution", len(probs) == 20,
          f"{len(probs)} moves from the start position")
    check("visit probabilities sum to 1",
          abs(sum(probs.values()) - 1.0) < 1e-6, str(sum(probs.values())))
    check("every searched move is legal",
          all(mv in chess.Board().legal_moves for mv in probs))
    check("root value is in [-1, 1]", -1.0 <= m.root_value <= 1.0,
          str(m.root_value))

    pv = m.pv_line(chess.Board())
    check("principal variation is non-empty", len(pv) > 0)

    # Terminal scoring is the mechanism that makes mate findable, and it is
    # deterministic - unlike "does an untrained net find mate", which tests
    # the weights rather than the search.
    m2 = MCTS(net, dev, cfg.c_puct, dirichlet_alpha=0.0)
    mated = chess.Board("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")   # black is mated
    check("the mated position is checkmate", mated.is_checkmate())
    check("terminal value is -1 for the side that is mated",
          m2._terminal_value(mated) == -1.0, str(m2._terminal_value(mated)))
    stale = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")      # stalemate
    check("the stalemate position is stalemate", stale.is_stalemate())
    check("terminal value is 0 for a stalemate",
          m2._terminal_value(stale) == 0.0, str(m2._terminal_value(stale)))

    # Strength check, only when real weights are available.
    from analyze_losses import default_checkpoint
    from utils import load_checkpoint
    ck = default_checkpoint()
    if ck and ck.exists():
        trained = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                               cfg.policy_size // 64)
        blob = load_checkpoint(str(ck))
        trained.load_state_dict(blob.get("model_state_dict", blob))
        trained.eval()
        mate = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
        m3 = MCTS(trained, dev, cfg.c_puct, dirichlet_alpha=0.0)
        pr = m3.get_action_probs(mate, 100)
        best = max(pr, key=pr.get)
        check("the trained model finds mate in one",
              mate.san(best) == "Ra8#", f"played {mate.san(best)}")
    elif VERBOSE:
        print("skip  mate-in-one strength check (no checkpoint available)")

    # an already-finished game yields no moves rather than crashing
    over = chess.Board("7k/5KQ1/8/8/8/8/8/8 b - - 0 1")
    check("terminal position is game over", over.is_game_over())
    m3 = MCTS(net, dev, cfg.c_puct, dirichlet_alpha=0.0)
    check("search on a terminal position returns {}",
          m3.get_action_probs(over, 8) == {})

    # repetition awareness is opt-in, and off by default
    check("Config.repetition_aware defaults to False",
          Config().repetition_aware is False)
    m4 = MCTS(net, dev, cfg.c_puct, dirichlet_alpha=0.0, repetition_aware=True)
    check("repetition-aware search still returns legal moves",
          all(mv in chess.Board().legal_moves
              for mv in m4.get_action_probs(chess.Board(), 16)))


# =====================================================================
def test_train_step():
    section("train: augmentation and batching")
    import random

    from config import Config
    from train import train_step, make_batch
    from model import AlphaZeroNet
    from utils import board_to_planes

    cfg = Config()
    cfg.batch_size = 8
    net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                       cfg.policy_size // 64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # A position that still has castling rights must never be mirrored: a
    # file flip puts the king on d1, which cannot hold castling rights.
    #
    # train_step mirrors a LOCAL planes array, not batch[i][0], so checking
    # batch[i][0] can never fail. The observable effect is the remapped
    # action index written back into batch[i][1].
    cfg.batch_size = 32                      # enough that a flip is near-certain
    castling = board_to_planes(chess.Board()).astype(np.uint8)
    batch = [(castling, [0], np.array([1.0], dtype=np.float32), 0.0)
             for _ in range(cfg.batch_size)]
    random.seed(0)
    train_step(net, opt, batch, torch.device("cpu"), cfg)
    check("castling positions keep their original action index "
          "(never mirrored)",
          all(s[1] == [0] for s in batch),
          f"{sum(1 for s in batch if s[1] != [0])}/{len(batch)} were remapped")

    # ...and a position with no castling rights IS eligible, which proves the
    # guard is selective rather than disabling augmentation altogether.
    free = board_to_planes(
        chess.Board("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1"))
    check("castling planes are empty in the no-rights position",
          not free[12:16].any())
    free_u8 = free.astype(np.uint8)
    batch2 = [(free_u8, [0], np.array([1.0], dtype=np.float32), 0.0)
              for _ in range(cfg.batch_size)]
    random.seed(0)
    train_step(net, opt, batch2, torch.device("cpu"), cfg)
    check("positions without castling rights do get mirrored",
          any(s[1] != [0] for s in batch2),
          f"{sum(1 for s in batch2 if s[1] != [0])}/{len(batch2)} remapped")
    cfg.batch_size = 8

    # losses are finite and the step runs
    loss, ploss, vloss = train_step(net, opt, batch, torch.device("cpu"), cfg)
    check("training loss is finite", np.isfinite(loss), str(loss))
    check("policy loss is non-negative", ploss >= 0, str(ploss))
    check("value loss is non-negative", vloss >= 0, str(vloss))

    # make_batch always returns batch_size samples, even from a tiny buffer
    replay = [(castling, [0], np.array([1.0], dtype=np.float32), 0.0)] * 3
    got = make_batch(replay, cfg)
    check("make_batch fills a short buffer to batch_size",
          len(got) == cfg.batch_size, f"got {len(got)}")


# =====================================================================
def test_selfplay_samples():
    section("selfplay: sample shape")
    from config import Config
    from model import AlphaZeroNet
    from selfplay import play_one_game

    cfg = Config()
    cfg.num_simulations = 4
    cfg.games_per_iteration = 1
    cfg.max_game_length = 6
    cfg.temperature_moves = 2
    cfg.opening_curriculum = False
    net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                       cfg.policy_size // 64)
    net.eval()

    samples = play_one_game(net, torch.device("cpu"), cfg)
    check("self-play produced samples", len(samples) > 0, str(len(samples)))
    planes, idx, probs, z = samples[0]
    check("stored planes are uint8", planes.dtype == np.uint8, str(planes.dtype))
    check("stored planes are 18x8x8", planes.shape == (18, 8, 8), str(planes.shape))
    check("policy target sums to 1", abs(float(probs.sum()) - 1.0) < 1e-5,
          str(probs.sum()))
    check("indices align with probabilities", len(idx) == len(probs))
    check("every action index is in range", all(0 <= i < 4672 for i in idx))
    check("value target is -1, 0 or 1", z in (-1.0, 0.0, 1.0), str(z))


# =====================================================================
def test_pgn_reader():
    section("pretrain: PGN reader")
    import tempfile

    from pretrain_supervised import build_dataset

    # a plain .pgn must not be fed to the zstd reader (that used to spin
    # forever inside `except: continue`)
    with tempfile.TemporaryDirectory() as d:
        pgn = Path(d) / "g.pgn"
        pgn.write_text('[Event "t"]\n[Result "1-0"]\n\n'
                       '1. e4 e5 2. Nf3 Nc6 1-0\n\n', encoding="utf-8")
        X, Y, Z, seeds = build_dataset(pgn, 500, 50)
        check("plain .pgn parses without hanging", len(X) == 4, f"{len(X)} positions")
        check("parsed planes are uint8", X.dtype == np.uint8, str(X.dtype))
        check("move indices are int64", Y.dtype == np.int64, str(Y.dtype))

        # an empty/unfinished file returns empty arrays rather than crashing
        empty = Path(d) / "e.pgn"
        empty.write_text('[Event "t"]\n[Result "*"]\n\n1. e4 *\n\n',
                         encoding="utf-8")
        X2, Y2, Z2, _ = build_dataset(empty, 500, 50)
        check("unfinished games yield an empty dataset, not a crash",
              len(X2) == 0, f"{len(X2)} positions")
        check("empty dataset keeps the right shape", X2.shape[1:] == (18, 8, 8),
              str(X2.shape))


# =====================================================================
def test_checkpoint_discovery():
    section("checkpoint discovery")
    import analyze_losses
    ck = analyze_losses.default_checkpoint()
    check("analyze_losses finds a checkpoint", ck is not None and ck.exists(),
          str(ck))
    check("it prefers a gated best.pt when one exists",
          ck is None or ck.name == "best.pt" or not list(
              (ROOT / "runs").glob("*/checkpoints/best.pt")),
          str(ck))


# =====================================================================
def test_dashboard_internals():
    """In-process checks of dashboard logic that the HTTP tests miss."""
    section("dashboard: checkpoint handling")
    import copy
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "srv_under_test", ROOT / "dashboard" / "server.py")
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)

    # what counts as a selectable checkpoint
    for name, want in [("iter_400.pt", True), ("best.pt", True),
                       ("latest.pt", True), ("taught_1.pt", True),
                       ("anchor_data.pt", False), ("replay_buffer.pt", False),
                       ("notes.txt", False)]:
        check(f"is_model_checkpoint({name}) is {want}",
              srv.is_model_checkpoint(name) is want)

    ck = srv.latest_checkpoint()
    if ck:
        base = Path(ck).name
        # a taught model is a personalised branch, not a training result
        check("the default checkpoint is never a taught_*.pt",
              not base.startswith("taught_"), base)
        has_best = bool(list((ROOT / "runs").glob("*/checkpoints/best.pt"))) \
            or (ROOT / "checkpoints" / "best.pt").exists()
        if has_best:
            check("a gated best.pt is preferred as the default",
                  base == "best.pt", base)

        # load_net memoises, so teaching must train a copy - training the
        # returned object would mutate what every request is reading
        a = srv.load_net(ck)
        b = srv.load_net(ck)
        check("load_net memoises by path", a is b)

        # Exercise the real teach path: it must not train the cached object.
        # Asserting that deepcopy works would only test deepcopy.
        saved_games = srv.MY_GAMES.read_bytes() if srv.MY_GAMES.exists() else None
        before = a.value_head.fc2.weight.detach().clone()
        produced = None
        try:
            srv.MY_GAMES.parent.mkdir(parents=True, exist_ok=True)
            srv.MY_GAMES.write_text(
                '[Event "t"]\n[Result "0-1"]\n\n'
                '1. f3 e5 2. g4 Qh4# 0-1\n\n', encoding="utf-8")
            srv._teach_job.update({"running": True, "step": "", "results": None,
                                   "error": None})
            srv._teach_worker(ck)
            err = srv._teach_job["error"]
            check("teach run completed without error", err is None, str(err))
            res = srv._teach_job["results"] or {}
            produced = res.get("path")
            check("teach writes a taught_*.pt, not an iter_*.pt",
                  str(res.get("saved", "")).startswith("taught_"),
                  str(res.get("saved")))
            check("teach records which checkpoint it came from",
                  res.get("taught_from") == Path(ck).name, str(res.get("taught_from")))
            check("teaching does NOT mutate the cached network",
                  torch.equal(before, a.value_head.fc2.weight.detach()),
                  "the shared net was trained in place")
        finally:
            if produced and Path(produced).exists():
                Path(produced).unlink()
            if saved_games is None:
                srv.MY_GAMES.unlink(missing_ok=True)
            else:
                srv.MY_GAMES.write_bytes(saved_games)

    section("dashboard: log parsing")
    stats = srv.parse_stats()
    for key in ("loss", "lr", "evals", "gate", "baseline", "selfplay_positions"):
        check(f"parse_stats returns '{key}'", key in stats)
    check("gate entries carry a promoted flag",
          all("promoted" in g for g in stats["gate"]), str(stats["gate"][:1]))


# =====================================================================
def test_forensics():
    section("loss forensics")
    from analyze_losses import material, worst_blunder

    def build(moves, net_color):
        b = chess.Board()
        hist, diffs = [], [material(b, net_color)]
        for uci in moves:
            mv = chess.Move.from_uci(uci)
            san = b.san(mv)
            b.push(mv)
            hist.append((san, b.fen(), b.ply()))
            diffs.append(material(b, net_color))
        return hist, diffs

    # White hangs the queen on f7; Black's king takes it -> drop of 9
    moves = ["e2e4", "d7d5", "d1h5", "g8f6", "h5f7", "e8f7"]
    h, d = build(moves, chess.WHITE)
    got = worst_blunder(h, d, True, chess.WHITE)
    check("worst blunder is the queen capture", got.get("san") == "Kxf7",
          str(got))
    check("the drop is 9 points", got["drop"] == 9, str(got["drop"]))

    # from Black's side the biggest opponent capture is only a pawn
    h, d = build(moves, chess.BLACK)
    got = worst_blunder(h, d, False, chess.BLACK)
    check("a 1-point loss stays below the reporting threshold",
          got["drop"] == 1 and "san" not in got, str(got))


# =====================================================================
def main():
    tests = [test_utils, test_move_encoding, test_model, test_evaluate,
             test_mcts, test_train_step, test_selfplay_samples,
             test_pgn_reader, test_checkpoint_discovery,
             test_dashboard_internals, test_forensics]
    for t in tests:
        try:
            t()
        except Exception:                                  # noqa: BLE001
            check(f"{t.__name__} raised", False, traceback.format_exc())

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = len(_RESULTS) - passed
    print(f"\n{passed} passed, {failed} failed, {len(_RESULTS)} checks total")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
