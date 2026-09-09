/* Test the dashboard's board-coordinate and FEN helpers.
 *
 * The front end is ~700 lines that had only ever been syntax-checked. These
 * are the functions most prone to silent off-by-one and flip errors, and
 * they decide which square a click maps to. Ground truth for piece
 * placement comes from python-chess, written out by tests/test_suite.py.
 *
 *   node tests/test_frontend.js dashboard/static/index.html <truth.json>
 *
 * test_suite.py runs this automatically when node is available.
 */
const fs = require("fs");

const htmlPath = process.argv[2] || "dashboard/static/index.html";
const truthPath = process.argv[3];

const html = fs.readFileSync(htmlPath, "utf8");
const js = html.match(/<script>([\s\S]*?)<\/script>/)[1];

// Extract just the pure helpers, with a minimal stand-in for page state.
const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const S = { flipped: false, myColor: "white" };
const src = ["fenTurn", "pieceAt", "isMyPiece", "dispToSq", "sqToDisp"]
  .map(name => {
    const m = js.match(new RegExp(`function ${name}\\([\\s\\S]*?\\n\\}`));
    if (!m) throw new Error(`could not extract ${name} from ${htmlPath}`);
    return m[0];
  }).join("\n");
eval(src);

let pass = 0, fail = 0;
function check(name, cond, detail) {
  if (cond) pass++;
  else {
    fail++;
    console.log(`FAIL  ${name}${detail ? "  [" + detail + "]" : ""}`);
  }
}

// ---- piece placement and side to move, vs python-chess ----------------
if (truthPath && fs.existsSync(truthPath)) {
  const truth = JSON.parse(fs.readFileSync(truthPath, "utf8"));
  for (const [fen, squares] of Object.entries(truth.pieces)) {
    for (const [sq, expected] of Object.entries(squares)) {
      const got = pieceAt(fen, sq);
      check(`pieceAt(${sq})`, got === (expected || null),
            `got ${got}, expected ${expected}`);
    }
  }
  for (const [fen, turn] of Object.entries(truth.turns)) {
    check(`fenTurn -> ${turn}`, fenTurn(fen) === turn, fenTurn(fen));
  }
}

// ---- the board grid maps onto the 64 squares, both orientations -------
for (const flipped of [false, true]) {
  S.flipped = flipped;
  const seen = new Set();
  for (let dr = 0; dr < 8; dr++)
    for (let df = 0; df < 8; df++) seen.add(dispToSq(dr, df));
  check(`flipped=${flipped}: grid covers 64 distinct squares`,
        seen.size === 64, `got ${seen.size}`);
  check(`flipped=${flipped}: top-left is ${flipped ? "h1" : "a8"}`,
        dispToSq(0, 0) === (flipped ? "h1" : "a8"), dispToSq(0, 0));
  check(`flipped=${flipped}: bottom-right is ${flipped ? "a8" : "h1"}`,
        dispToSq(7, 7) === (flipped ? "a8" : "h1"), dispToSq(7, 7));
}

// ---- sqToDisp is the inverse of dispToSq ------------------------------
// It returned the file unmirrored when flipped, putting all 64 squares in
// the wrong column. Dead code at the time, but a trap for any future use.
for (const flipped of [false, true]) {
  S.flipped = flipped;
  const bad = [];
  for (let dr = 0; dr < 8; dr++) {
    for (let df = 0; df < 8; df++) {
      const sq = dispToSq(dr, df);
      const [r2, f2] = sqToDisp(sq);
      if (r2 !== dr || f2 !== df) bad.push(`(${dr},${df})->${sq}->(${r2},${f2})`);
    }
  }
  check(`flipped=${flipped}: sqToDisp inverts dispToSq`, bad.length === 0,
        `${bad.length}/64 wrong, e.g. ${bad.slice(0, 3).join(" ")}`);
}

// ---- you may only pick up your own pieces, on your own turn -----------
S.flipped = false;
S.myColor = "white";
const start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
check("white may pick up a white pawn on its turn", isMyPiece(start, "e2"));
check("white may not pick up a black pawn", !isMyPiece(start, "e7"));
check("white may not pick up an empty square", !isMyPiece(start, "e4"));
S.myColor = "black";
check("black may not move on white's turn", !isMyPiece(start, "e7"));

console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
