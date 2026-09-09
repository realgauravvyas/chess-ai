/* Chess engine for the browser: board encoding, move encoding and MCTS.
 *
 * These must match utils.py, move_encoding.py and mcts.py exactly. If the
 * 18-plane tensor or the 4672-action indexing differ by so much as one
 * square, the network is fed something it never saw in training and plays
 * nonsense. tests/test_pages_encoding.py checks both against Python.
 */

/* ---------------------------------------------------------------- board */
// python-chess numbers squares a1=0 .. h8=63, i.e. rank * 8 + file.
const FILES = "abcdefgh";

function sqIndex(name) {
  return (parseInt(name[1], 10) - 1) * 8 + FILES.indexOf(name[0]);
}
function sqFile(i) { return i % 8; }
function sqRank(i) { return Math.floor(i / 8); }
function sqName(i) { return FILES[sqFile(i)] + (sqRank(i) + 1); }

// plane order, matching utils.board_to_planes:
//   0-5   white P N B R Q K
//   6-11  black P N B R Q K
//   12-15 castling WK WQ BK BQ, filled entirely when the right exists
//   16    en-passant target square
//   17    side to move, all ones when White
const PIECE_ORDER = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };

function boardToPlanes(game) {
  const planes = new Float32Array(18 * 64);
  const board = game.board();          // rank 8 first, file a first
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const piece = board[r][f];
      if (!piece) continue;
      const rank = 7 - r;              // board()[0] is rank 8
      const offset = piece.color === "w" ? 0 : 6;
      const plane = offset + PIECE_ORDER[piece.type];
      planes[plane * 64 + rank * 8 + f] = 1;
    }
  }

  // chess.js exposes castling rights through the FEN field
  const fen = game.fen().split(" ");
  const rights = fen[2];
  const castle = [["K", 12], ["Q", 13], ["k", 14], ["q", 15]];
  for (const [ch, plane] of castle) {
    if (rights.includes(ch)) planes.fill(1, plane * 64, plane * 64 + 64);
  }

  const ep = fen[3];
  if (ep && ep !== "-") {
    const i = sqIndex(ep);
    planes[16 * 64 + sqRank(i) * 8 + sqFile(i)] = 1;
  }

  if (fen[1] === "w") planes.fill(1, 17 * 64, 17 * 64 + 64);
  return planes;
}

/* ----------------------------------------------------------- move index */
// 73 planes per origin square: 56 queen-style, 8 knight, 9 underpromotion.
const QUEEN_DIRECTIONS = [
  [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1],
];
const KNIGHT_DELTAS = [
  [1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2],
];
const UNDERPROMO = ["n", "b", "r"];    // knight, bishop, rook

function encodeMove(move) {
  const fs = sqIndex(move.from), ts = sqIndex(move.to);
  const ff = sqFile(fs), fr = sqRank(fs);
  const df = sqFile(ts) - ff, dr = sqRank(ts) - fr;
  const base = fs * 73;

  // underpromotions get their own planes; a queen promotion is encoded as
  // the ordinary queen-style move it rides on
  if (move.promotion && move.promotion !== "q") {
    const piece = UNDERPROMO.indexOf(move.promotion);
    let direction;
    if (df === 0) {
      direction = 1;                                   // straight
    } else {
      const movingUp = dr > 0;
      const left = movingUp ? df < 0 : df > 0;
      direction = left ? 0 : 2;
    }
    return base + 64 + piece * 3 + direction;
  }

  const adf = Math.abs(df), adr = Math.abs(dr);
  if ((adf === 1 && adr === 2) || (adf === 2 && adr === 1)) {
    const k = KNIGHT_DELTAS.findIndex(([a, b]) => a === df && b === dr);
    return base + 56 + k;
  }

  const step = Math.max(adf, adr);
  const ux = df / step, uy = dr / step;    // exact: df and dr are multiples
  const d = QUEEN_DIRECTIONS.findIndex(([a, b]) => a === ux && b === uy);
  return base + d * 7 + (step - 1);
}

/* ------------------------------------------------------------------ MCTS */
class Node {
  constructor(prior = 0, move = null) {
    this.prior = prior;
    this.visits = 0;
    this.total = 0;
    this.move = move;
    this.children = null;              // Map<uci, Node>
    this.terminal = false;
    this.terminalValue = 0;
  }
  get value() { return this.total / (this.visits + 1e-8); }
}

class MCTS {
  constructor(session, cPuct = 1.5) {
    this.session = session;
    this.cPuct = cPuct;
  }

  async evaluate(game) {
    const planes = boardToPlanes(game);
    const tensor = new ort.Tensor("float32", planes, [1, 18, 8, 8]);
    const out = await this.session.run({ board: tensor });
    const logits = out.policy.data;
    const value = out.value.data[0];

    // softmax over the full 4672 actions, as in mcts._evaluate
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
    let sum = 0;
    const probs = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      probs[i] = Math.exp(logits[i] - max);
      sum += probs[i];
    }
    for (let i = 0; i < probs.length; i++) probs[i] /= sum;
    return { probs, value };
  }

  // result from the perspective of the side to move, matching
  // mcts._terminal_value
  terminalValue(game) {
    if (game.isCheckmate()) return -1;   // side to move has been mated
    return 0;                            // stalemate, repetition, 50-move
  }

  select(node) {
    let best = null, bestScore = -Infinity;
    const sqrtVisits = Math.sqrt(node.visits);
    for (const [uci, child] of node.children) {
      const score = -child.value
        + this.cPuct * child.prior * sqrtVisits / (1 + child.visits);
      if (score > bestScore) { bestScore = score; best = uci; }
    }
    return best;
  }

  async simulate(game, root) {
    const g = new Chess(game.fen());
    let node = root;
    const path = [node];

    while (node.children) {
      const uci = this.select(node);
      g.move(uciToMove(g, uci));
      node = node.children.get(uci);
      path.push(node);
    }

    let value;
    if (node.terminal) {
      value = node.terminalValue;
    } else if (g.isGameOver()) {
      value = this.terminalValue(g);
      node.terminal = true;
      node.terminalValue = value;
    } else {
      const { probs, value: v } = await this.evaluate(g);
      value = v;
      node.children = new Map();
      for (const m of g.moves({ verbose: true })) {
        node.children.set(moveToUci(m), new Node(probs[encodeMove(m)], m));
      }
    }

    // values are stored from each node's own side-to-move perspective, so
    // the sign flips at every ply on the way back up
    let sign = 1;
    for (let i = path.length - 1; i >= 0; i--) {
      path[i].visits += 1;
      path[i].total += sign * value;
      sign = -sign;
    }
  }

  async search(game, sims, onProgress) {
    const root = new Node();
    for (let i = 0; i < sims; i++) {
      await this.simulate(game, root);
      if (onProgress && i % 16 === 15) {
        onProgress((i + 1) / sims);
        await new Promise(r => setTimeout(r, 0));   // let the page repaint
      }
    }
    if (!root.children) return null;

    const visits = [...root.children.entries()]
      .map(([uci, n]) => ({ uci, visits: n.visits, move: n.move }))
      .sort((a, b) => b.visits - a.visits);
    const total = visits.reduce((s, v) => s + v.visits, 0) || 1;

    // principal variation: follow the most-visited child down the tree
    const pv = [];
    const g = new Chess(game.fen());
    let node = root;
    while (node.children && pv.length < 6) {
      let best = null, bv = -1;
      for (const [uci, child] of node.children) {
        if (child.visits > bv) { bv = child.visits; best = uci; }
      }
      const mv = g.move(uciToMove(g, best));
      if (!mv) break;
      pv.push(mv.san);
      node = node.children.get(best);
    }

    return {
      best: visits[0],
      top: visits.slice(0, 4).map(v => ({ uci: v.uci, p: v.visits / total })),
      value: root.total / Math.max(1, root.visits),
      pv,
    };
  }
}

/* ------------------------------------------------------------- helpers */
function moveToUci(m) {
  return m.from + m.to + (m.promotion || "");
}
function uciToMove(game, uci) {
  return {
    from: uci.slice(0, 2),
    to: uci.slice(2, 4),
    promotion: uci.length > 4 ? uci[4] : undefined,
  };
}
