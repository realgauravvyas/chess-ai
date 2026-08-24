"""End-to-end tests for the dashboard server. Run while server is up."""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:5000"
PASS, FAIL = 0, 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name} {extra}")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return r.status, r.read()


def post(path, body):
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main():
    # 1. index page
    status, body = get("/")
    html = body.decode()
    check("GET / returns 200", status == 200)
    check("index has board element", 'id="board"' in html)
    check("index has charts", 'id="chLoss"' in html)

    # 2. stats
    status, stats = get("/api/stats")
    stats = json.loads(stats)
    check("stats 200", status == 200)
    check("stats loss non-empty", len(stats["loss"]) > 50,
          f"got {len(stats['loss'])}")
    check("stats evals present", len(stats["evals"]) > 10)
    check("stats latest_iter >= 199", (stats["latest_iter"] or 0) >= 199)
    last_eval = stats["evals"][-1]
    check("last eval score sane", 0 <= last_eval["score"] <= 100,
          str(last_eval))

    # 3. checkpoints
    status, ckpts = get("/api/checkpoints")
    ckpts = json.loads(ckpts)
    names = [c["name"] for c in ckpts]
    check("checkpoints list", status == 200 and "iter_200.pt" in names,
          str(names[:3]))

    # 4. newgame white
    st, g = post("/api/newgame", {"color": "white"})
    check("newgame white", st == 200 and g["your_color"] == "white"
          and g["fen"].startswith("rnbqkbnr/pppppppp") and len(g["legal"]) == 20)

    # 5. legal human move e2e4
    fen0 = g["fen"]
    st, r = post("/api/human_move", {"fen": fen0, "uci": "e2e4"})
    check("human e2e4 ok", st == 200 and r["san"] == "e4"
          and r["fen"].split()[0].startswith("rnbqkbnr/pppppppp/8/8/4P3"))
    fen_after_e4 = r["fen"]

    # 6. illegal move rejected
    st, r_bad = post("/api/human_move", {"fen": fen0, "uci": "e2e5"})
    check("illegal move rejected", st == 400 and "illegal" in r_bad.get("error", ""))

    # 7. model reply
    st, m = post("/api/model_move", {"fen": fen_after_e4, "checkpoint": None,
                                     "sims": 40})
    legal_ucis = {mv["uci"] for mv in
                  json.loads(json.dumps([]))} | set()
    check("model_move ok", st == 200 and "move" in m and -1 <= m["value"] <= 1
          and len(m["top_moves"]) >= 1, json.dumps(m)[:200])
    check("model top_moves sum<=1",
          sum(t["p"] for t in m["top_moves"]) <= 1.001)

    # verify model's move is actually legal in that position
    import chess
    b = chess.Board(fen_after_e4)
    mv = chess.Move.from_uci(m["move"])
    check("model move is legal", mv in b.legal_moves)

    # 8. short scripted game loop (random legal vs model)
    board = chess.Board()
    import random
    random.seed(3)
    plies = 0
    err = None
    while not board.is_game_over() and plies < 8:
        if board.turn == chess.WHITE:
            moves = [mv.uci() for mv in board.legal_moves]
            uci = random.choice(moves)
            s2, rr = post("/api/human_move", {"fen": board.fen(), "uci": uci})
            if s2 != 200:
                err = f"human_move failed: {rr}"
                break
            board = chess.Board(rr["fen"])
        else:
            s2, rr = post("/api/model_move", {"fen": board.fen(), "sims": 20})
            if s2 != 200:
                err = f"model_move failed: {rr}"
                break
            board = chess.Board(rr["fen"])
        plies += 1
    check("scripted game loop", err is None and plies >= 6,
          f"{err} plies={plies}")

    # 9. analyze endpoint (apply=false keeps position)
    st, a = post("/api/model_move", {"fen": chess.Board().fen(), "sims": 20,
                                     "apply": False})
    check("analyze apply=false", st == 200 and "fen" not in a)

    # 10. evaluation job
    iter5 = next((c["path"] for c in ckpts if c["name"] == "iter_5.pt"), None)
    st, j = post("/api/evaluate", {"checkpoint": iter5, "games": 4, "sims": 15})
    check("evaluate starts", st == 200 and j.get("started"))
    done = False
    for _ in range(90):
        time.sleep(2)
        with urllib.request.urlopen(BASE + "/api/evaluate/status", timeout=30) as rr:
            js = json.loads(rr.read())
        if not js["running"]:
            done = True
            break
    check("evaluation finished", done and js.get("results") is not None,
          json.dumps(js))
    res = js.get("results") or {}
    played = res.get("played", 0)
    check("eval results sane", played == 4
          and 0 <= res.get("score", -1) <= 100, json.dumps(res))

    # 11. training status fields
    check("training_alive field exists", "training_alive" in stats)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
