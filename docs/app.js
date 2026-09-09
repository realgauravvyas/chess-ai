/* UI for the browser demo: board rendering, input, and driving the search.
 *
 * The engine itself (board encoding, move indexing, MCTS) lives in
 * engine.js and is verified against Python by tests/test_pages_encoding.py.
 */
const $ = id => document.getElementById(id);

const GLYPH = {
  wp: "♙", wn: "♘", wb: "♗", wr: "♖", wq: "♕", wk: "♔",
  bp: "♟", bn: "♞", bb: "♝", br: "♜", bq: "♛", bk: "♚",
};

const S = {
  game: new Chess(),
  mcts: null,
  myColor: "w",
  flipped: false,
  selected: null,
  targets: [],
  last: null,
  busy: false,
  history: [],
};

/* ------------------------------------------------------------- loading */
(async function boot() {
  const bar = $("loadbar");
  try {
    ort.env.wasm.numThreads = 1;                 // no SharedArrayBuffer needed
    bar.style.width = "35%";
    const session = await ort.InferenceSession.create("chess_model.onnx", {
      executionProviders: ["wasm"],
    });
    bar.style.width = "100%";
    S.mcts = new MCTS(session);
    $("loading").style.display = "none";
    $("app").style.display = "grid";
    newGame();
  } catch (e) {
    $("loading").innerHTML =
      `<b class="bad">Could not load the model.</b><br>${e.message}` +
      `<br><span class="muted">This page needs WebAssembly; try a current ` +
      `Chrome, Edge, Firefox or Safari.</span>`;
  }
})();

/* --------------------------------------------------------------- board */
function render() {
  const b = $("board");
  b.innerHTML = "";
  const board = S.game.board();
  const inCheck = S.game.inCheck();
  const turn = S.game.turn();

  for (let dr = 0; dr < 8; dr++) {
    for (let df = 0; df < 8; df++) {
      const r = S.flipped ? dr : 7 - dr;         // rank index 0..7 (a1 = 0)
      const f = S.flipped ? 7 - df : df;
      const name = "abcdefgh"[f] + (r + 1);
      const piece = board[7 - r][f];

      const cell = document.createElement("div");
      cell.className = "sq " + ((r + f) % 2 ? "light" : "dark");
      cell.dataset.sq = name;

      if (S.selected === name) cell.classList.add("sel");
      if (S.last && (S.last.from === name || S.last.to === name))
        cell.classList.add("last");
      if (inCheck && piece && piece.type === "k" && piece.color === turn)
        cell.classList.add("check");

      if (piece) cell.textContent = GLYPH[piece.color + piece.type];

      if (S.targets.includes(name)) {
        if (piece) cell.classList.add("cap");
        const d = document.createElement("div");
        d.className = "dot";
        cell.appendChild(d);
      }

      // file letters along the bottom, rank numbers down the side
      if (dr === 7) {
        const c = document.createElement("span");
        c.className = "coord f";
        c.textContent = "abcdefgh"[f];
        cell.appendChild(c);
      }
      if (df === 0) {
        const c = document.createElement("span");
        c.className = "coord r";
        c.textContent = r + 1;
        cell.appendChild(c);
      }

      cell.onclick = () => onSquare(name);
      b.appendChild(cell);
    }
  }
  renderMoves();
}

function onSquare(name) {
  if (S.busy || S.game.isGameOver()) return;
  if (S.game.turn() !== S.myColor) return;

  if (S.selected && S.targets.includes(name)) {
    play(S.selected, name);
    return;
  }
  const moves = S.game.moves({ square: name, verbose: true });
  if (moves.length) {
    S.selected = name;
    S.targets = moves.map(m => m.to);
  } else {
    S.selected = null;
    S.targets = [];
  }
  render();
}

/* ---------------------------------------------------------------- play */
function play(from, to) {
  // promotion is mandatory, so a bare move would be rejected: auto-queen
  const candidates = S.game.moves({ square: from, verbose: true })
    .filter(m => m.to === to);
  const mv = candidates.find(m => m.promotion === "q") || candidates[0];
  if (!mv) return;

  S.history.push(S.game.fen());
  S.game.move(mv);
  S.last = { from, to };
  S.selected = null;
  S.targets = [];
  render();

  if (finished()) return;
  setTimeout(modelMove, 60);          // let the board repaint first
}

async function modelMove() {
  S.busy = true;
  const sims = +$("sims").value;
  setStatus(`Thinking… (${sims} simulations)`);

  const t0 = performance.now();
  const res = await S.mcts.search(S.game, sims, frac => {
    setStatus(`Thinking… ${Math.round(frac * 100)}%`);
  });
  const secs = ((performance.now() - t0) / 1000).toFixed(1);

  if (!res) { S.busy = false; finished(); return; }

  const mv = S.game.move({
    from: res.best.uci.slice(0, 2),
    to: res.best.uci.slice(2, 4),
    promotion: res.best.uci.length > 4 ? res.best.uci[4] : undefined,
  });
  S.last = { from: mv.from, to: mv.to };

  // the search value is from the mover's side; the bar is drawn White-up
  const white = S.game.turn() === "w" ? -res.value : res.value;
  setEval(white);
  showTop(res.top);

  S.busy = false;
  render();
  if (!finished()) {
    setStatus(`It played ${mv.san} · ${secs}s · ` +
              (res.pv.length ? `expects ${res.pv.slice(0, 4).join(" ")}` : ""));
  }
}

function finished() {
  if (!S.game.isGameOver()) return false;
  let msg;
  if (S.game.isCheckmate()) {
    const winner = S.game.turn() === S.myColor ? "The model wins" : "You win";
    msg = `Checkmate — ${winner}.`;
  } else if (S.game.isStalemate()) msg = "Stalemate — a draw.";
  else if (S.game.isThreefoldRepetition()) msg = "Draw by repetition.";
  else if (S.game.isInsufficientMaterial()) msg = "Draw — not enough material.";
  else msg = "Draw by the fifty-move rule.";
  setStatus(msg, "good");
  render();
  return true;
}

/* ------------------------------------------------------------ controls */
window.newGame = function () {
  S.game = new Chess();
  S.myColor = $("mycolor").value;
  S.flipped = S.myColor === "b";
  S.selected = null; S.targets = []; S.last = null; S.history = [];
  S.busy = false;
  setEval(0);
  $("topmoves").textContent = "Play a move to see the search.";
  render();
  if (S.game.turn() !== S.myColor) { setTimeout(modelMove, 200); }
  else setStatus("Your move.");
};

window.undo = function () {
  if (S.busy || S.history.length === 0) return;
  S.game = new Chess(S.history.pop());
  S.selected = null; S.targets = []; S.last = null;
  render();
  setStatus("Took back a move.");
};

window.flip = function () { S.flipped = !S.flipped; render(); };

$("sims").oninput = e => { $("simsval").textContent = e.target.value; };
$("mycolor").onchange = () => newGame();

/* -------------------------------------------------------------- output */
function setStatus(msg, cls) {
  const el = $("status");
  el.textContent = msg;
  el.className = cls || "muted";
}

function setEval(v) {
  v = Math.max(-1, Math.min(1, v));
  const fill = $("evalfill");
  fill.style.height = (Math.abs(v) * 50) + "%";
  if (v >= 0) { fill.style.top = (50 - Math.abs(v) * 50) + "%"; fill.style.background = "#e8edf5"; }
  else { fill.style.top = "50%"; fill.style.background = "#39424f"; }
}

function showTop(top) {
  if (!top || !top.length) return;
  const g = new Chess(S.history.length ? S.game.fen() : undefined);
  $("topmoves").innerHTML = top.map((t, i) => {
    const pct = (t.p * 100).toFixed(0);
    return `<div class="tmrow"><span style="width:52px">${t.uci}</span>
      <div class="tmbar"><div class="tmfill" style="width:${pct}%;
        ${i ? "opacity:.5" : ""}"></div></div>
      <span class="pct">${pct}%</span></div>`;
  }).join("");
}

function renderMoves() {
  const h = S.game.history();
  let html = "";
  for (let i = 0; i < h.length; i += 2) {
    html += `<tr><td class="num">${i / 2 + 1}.</td>` +
            `<td>${h[i] || ""}</td><td>${h[i + 1] || ""}</td></tr>`;
  }
  $("movetable").innerHTML = html;
}
