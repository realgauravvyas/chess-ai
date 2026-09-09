# Measured results

Every number here is from a run that can be reproduced with the scripts in
`experiments/`. All match scores are from the **network's** perspective.

## Summary

The supervised stage worked well. The self-play stage, as originally
configured, made the model measurably **weaker** — and the evaluation metric
in use at the time was incapable of detecting that.

After correcting the metric and two training defects, a second 199-iteration
run with acceptance gating left the model **statistically indistinguishable**
from its starting point (46.2%, 95% CI [37.4, 55.1] over 40 games). Gating
prevented the regression; it did not produce improvement. The candidates the
gate rejected were significantly worse than baseline (42.2%, p = 0.004), so
self-play was still degrading the network throughout.

## 1. The evaluation metric was broken

`evaluate_vs_baseline` alternated the network between white and black but
scored `results["1-0"] + 0.5·draws` — that is **white's** score, not the
network's. Every win the network earned as black was counted as a loss.

The signature is unmistakable in the training logs: with two deterministic
players and no opening randomisation, all five games as white are identical
and all five as black are identical, so the tally can only ever be symmetric:

```
[iter 454] eval vs greedy: {'1-0': 0, '0-1': 0, '1/2-1/2': 10} (score 50%)
[iter 499] eval vs greedy: {'1-0': 0, '0-1': 0, '1/2-1/2': 10} (score 50%)
```

**159 evaluation points across two training runs carried no information.**
The same bug in `quick_eval` reported the pretrained model at 44% vs random
when its true score is 85%.

Fixed by `net_points()` in `evaluate.py`, plus randomised opening plies so
that an N-game match contains N games of information rather than one.

## 2. Supervised pretraining: works

676,648 positions from two Lichess monthly dumps (Elo ≥ 1750), 3 epochs.
Policy cross-entropy 3.90 → 2.06.

| Probe | Pretrained `iter_200` |
|---|---|
| Opening policy, start position | `e4` 0.594, `d4` 0.313 |
| Reply to 1.e4 | `e5` 0.291, `c5` 0.252 |
| Free hanging queen | **wins it** (`exd5`) |
| Mate in 1 | **finds it** (`Ra8#`) |
| vs random, 10 games | **85%** (+7 =3 −0) |

## 3. Self-play RL: regressed

Head-to-head, 8 games, alternating colors, randomised openings, 60 sims:

```
v5 iter_500  2.5 — 5.5  pretrained iter_200
```

**300 iterations of self-play fine-tuning produced a weaker model than the
one it started from**, while training loss fell from 2.36 to 1.85 throughout.
Training loss is not a strength signal.

The earlier v4 run (no rehearsal anchor) failed more visibly — catastrophic
forgetting of the supervised policy:

| Probe | v4 `iter_700` |
|---|---|
| Opening policy | `f3` 0.074, `e3` 0.073 — no real opening |
| Value at start position | **−0.270** (badly miscalibrated) |
| Free hanging queen | **misses it** (plays `b3`) |

The 30% supervised rehearsal anchor added in v5 did fix the collapse — v5
still opens `e4`/`d4` and still wins the free queen — but preventing collapse
is not the same as improving.

## 4. Why it regressed

**Search quality is not the problem.** MCTS clearly beats the network's own
greedy policy head even at low simulation counts
(`experiments/sims_scaling.py`, 12 games each):

| sims | MCTS score vs raw policy |
|---|---|
| 10 | 58.3% |
| 25 | 62.5% |
| 40 | 79.2% |
| 100 | 75.0% |
| 200 | 83.3% |

**The problem is target noise and corrupted augmentation:**

1. **A 40-visit histogram over ~30 legal moves is almost pure sampling
   noise.** The *argmax* of the search is good — that is what the table above
   measures — but cross-entropy trains on the *full distribution*, which at
   ~1.3 visits per legal move carries almost no information and flattens the
   sharp supervised policy.

2. **Mirror augmentation was producing illegal positions.** The code flipped
   the board files on 50% of every minibatch but did not swap the castling
   planes (12↔13, 14↔15). After a file flip the king sits on d1 while the
   "kingside castling" plane is still set — a board that cannot occur in
   chess, paired with contradictory features. Chess is *not* mirror-symmetric;
   AlphaZero deliberately omits symmetry augmentation for this reason.
   This corrupted roughly half of every batch, including the supervised
   rehearsal slice.

## 5. Corrected run (v7)

Changes: castling-safe mirror augmentation, 128 sims (≈4.3 visits/move
instead of 1.3), 80 gradient steps instead of 200, lr 2e-4, and
AlphaGo-Zero gating — self-play always generates from the best weights, and
new weights are promoted only after scoring ≥55% in an 8-game match.

Gating makes regression structurally impossible: the worst case is "nothing
gets promoted and the model stays at baseline strength."

The run completed 199 iterations (201 → 399). Its outcome was settled by a
40-game match with randomised openings and alternating colours — at 40 games
the standard error is 7.9 points, against 17.7 for the 8-game matches used
during training.

### The decisive result

```
gated best.pt  vs  frozen pretrained baseline
  +5  =27  -8      score 46.2%      95% CI [37.4%, 55.1%]
  z = -0.83   ->   no detectable difference
```

| Measurement | Result |
|---|---|
| Iterations | 199 (201 → 399) |
| Training loss | 2.560 → 2.473 |
| Gate promotions | **8 / 20** (pure noise predicts **7.8**) |
| Candidate vs baseline (20 gates) | **42.2%**, z = −2.87, **p = 0.004** |
| **Gated model vs baseline (40 games)** | **46.2%** [37.4, 55.1] — no difference |
| Gated model vs random (40 games) | 86.2% [79.2, 93.3] |
| Baseline vs random (40 games) | 87.5% [80.7, 94.3] |
| Trend across the run | first half 42.5% → second half 41.9% (flat) |

### What it means

**Self-play produced no improvement — but, unlike v5, no regression either.**

| Run | vs its own starting point |
|---|---|
| v5 (broken metric, corrupted augmentation, no gate) | 31.2% — clearly worse |
| v7 (corrected + gated) | 46.2% [37.4, 55.1] — no detectable change |

The mechanism is visible in the intermediate data. The **candidate** network
— trained continuously, repeatedly rejected by the gate — averaged 42.2%
against the baseline across 20 measurements (p = 0.004): significantly worse.
Self-play was still actively degrading the network. The gate is what kept
that damage out of the deployed weights.

### The honest caveat: the gate itself was underpowered

8 of 20 gates promoted. Pure chance predicts 7.8. An 8-game match has a
standard error of 17.7 percentage points, so a 55% threshold sits 0.28 SE
above chance — **individual gate decisions were coin flips.**

The aggregate protection is real: a random walk required to win a match
before advancing drifts upward relative to one that is not, which is why the
gated model (46.2%) sits above the candidate stream (42.2%). But this is a
probabilistic filter, not the guarantee it was initially described as.
Having found one metric that measured nothing, the same scrutiny had to be
applied to the fix.

### Why no improvement

The binding constraint is sample efficiency, unchanged by the fixes. At 128
simulations over ~30 legal moves the search yields **4.3 visits per legal
move**, against roughly 27 for AlphaZero at 800 simulations. The policy loss
consumes that distribution, and at 4.3 visits per bucket it is still mostly
sampling noise — sharper than the 1.3 of the original run, not sharp enough.

## 6. Post-run audit

Two bugs found after the run completed, both fixed:

| Bug | Impact |
|---|---|
| `api_stats` computed `training_alive` inside the cache-miss branch, keyed on log mtime | Once training stopped the log stopped changing, so the cache never refreshed and the dashboard reported `training: LIVE` **forever**. The one value that must not be cached against log mtime is the flag whose job is to notice mtime stopped advancing. |
| `latest_checkpoint()` scanned only `checkpoints/` | The dashboard's default model for playing, evaluating and teaching was `iter_500.pt` — the **regressed v5 model** (31.2% vs baseline). Gated runs write to `runs/<name>/checkpoints/best.pt`, which the glob could not match. |

Also corrected: the gate log printed the post-promotion `best_iter`, so a
promotion read "vs best (iter 400)" when it had played the previous best.

### The loss-forensics tool never worked

`analyze_losses.py` is meant to explain *how* the network loses. Its
attribution pass only runs when the network actually loses, and the network
now beats a random mover 24-0-16, so the path went unexercised for the whole
project while carrying **two** independent bugs:

| Bug | Effect |
|---|---|
| Inverted parity: `(ply % 2 == 0) != net_white` selects the network's own moves, not the opponent's | Material only falls when the opponent captures, so every candidate drop was <= 0 |
| Off-by-one: `diffs` starts at the initial position, so `diffs[k+1]` follows `history[k]` -- the code compared `diffs[k-1] - diffs[k]` | Drops were read from the wrong pair of positions |

Either alone forces every drop to 0, the `drop >= 2` branch never fires, and
the tool prints empty phase / hung-piece / undefended tables while looking
like it ran fine. Fixing the parity alone was not enough; the off-by-one only
surfaced once the logic was extracted into `worst_blunder()` and tested
against games with known answers
(`experiments/test_forensics.py`, 4 cases, all passing).

It also hardcoded `checkpoints/iter_500.pt` -- the regressed v5 model -- by a
relative path that breaks outside the project root.

### Watchdog crashed on a fresh clone

`watchdog_dashboard.py` opened `logs/dashboard.log` without creating `logs/`.
That directory is gitignored, so on a clone it raised `FileNotFoundError` at
the first restart attempt (verified against a real `git clone`). It also
leaked one file descriptor per restart, in a process designed to run
indefinitely, and retried a failing server every 28s with no backoff.

### Repetition blindness: measured, not assumed

MCTS copied the board with `stack=False`, discarding move history, so
repetitions were invisible to the search. **4 of 6 games ended in threefold
repetition** the engine never saw coming, and the decisive 40-game match
drew 27.

An opt-in fix (`Config.repetition_aware`) carries 12 plies of history and
scores a repeated position as a draw. Measured over 20 games:

```
repetition-aware vs repetition-blind (same weights)
  8.5 - 11.5   =  42.5%    z = -0.67, p ~ 0.50
```

**It does not improve playing strength.** But it changes how games end:

| | blind | aware |
|---|---|---|
| Threefold repetitions | 4 / 6 | 0 / 20 |
| Decisive games | ~33% | ~55% |

So its value is to *measurement*, not to play: a 55% decisive rate makes a
match roughly twice as informative per game as a 33% one, which is the
cheapest available way to strengthen the underpowered acceptance gate.

Default is **off**, so the published experiments reproduce exactly.

## 12. The front end was never functionally tested

`dashboard/static/index.html` is ~700 lines of JavaScript that had only ever
been syntax-checked. Its board-coordinate helpers are exactly the kind of
code that fails silently, so they were tested against python-chess ground
truth: `pieceAt` over 96 squares across four positions, `fenTurn`,
`dispToSq` covering all 64 squares in both orientations, and `isMyPiece`.

All passed except one: **`sqToDisp` did not invert `dispToSq` on a flipped
board.** It mirrors the rank but returns the file unchanged, so every one of
the 64 squares mapped to the wrong column when the board is viewed from
Black's side.

It is **dead code** - zero call sites - so no user ever hit it. It was fixed
rather than deleted, because it is the obvious helper to reach for when
touching the board rendering, and a silently wrong one is worse than none.

`tests/test_frontend.js` now runs from the Python suite whenever node is
available, and the mutation set covers the flip bug.

## 10. The two command-line entry points

`play.py` and `eval_match.py` were the last modules with no coverage.

**`play.py` made promotion a dead end.** Promotion is mandatory, so `e7e8`
parses cleanly through `chess.Move.from_uci` but is never legal. The only
feedback was *"Illegal move, try again."* with no hint that a piece letter
is required - and the browser dashboard auto-queens, so the behaviour
differed between the two front ends. It now auto-queens and names the
underpromotion syntax. Its `--checkpoint` also defaulted to the relative
string `"checkpoints/latest.pt"`: the v5 run's final weights, measured at
31.2% against the pretrained baseline, via a path that breaks outside the
project root.

**`eval_match.py` was hardcoded to two regressed checkpoints**, with the
pair baked into its docstring. It is superseded by
`experiments/final_verdict.py`, which parallelises, alternates colours and
reports a confidence interval. It now takes the checkpoints as arguments,
defaults to the gated best against the frozen baseline, and points at the
better tool.

## 11. Playing a full game through the dashboard

A complete 52-ply game was played through the HTTP API, validating every
response: FEN advance, move legality, SAN agreement, eval-bar orientation,
top-move probabilities, principal-variation legality and check detection.
Plus castling, en passant, promotion, analysis mode, error handling,
checkpoint switching and the stats feed.

**No defects found.** One apparent failure was the test's own fault: it
expected SAN `e8=Q` where the correct answer is `e8=Q+`, because the
promotion gives check.

Two observations about the engine rather than the dashboard:

- Model response time is **0.4 s at 120 simulations**, so the interface
  stays responsive at the default setting.
- The game ended in **threefold repetition while the model was winning**
  (eval −0.85 in its favour). This is the repetition blindness of
  Section 6 costing a won game. It remains the most valuable outstanding
  improvement, though the naive fix measured no strength gain.

## 9. The training wrapper turned a typo into a three-minute silent loop

`run_training.py` drives every training run and was the only such module
with no test coverage. Driving `main()` with a stubbed `subprocess` exposed
three defects:

| Defect | Effect |
|---|---|
| The worker count is positional and must come first, but nothing checked it | `run_training.py --device auto 6` silently built `train.py --workers --device auto 6` |
| Any non-zero exit was treated as a crash worth retrying | argparse rejects that command with exit code 2, and the wrapper retried the deterministic failure **20 times at 10s apart** |
| `args[i + 1]` with no bounds check | `--run-dir` as the final argument raised a bare `IndexError` |

Together these mean a single misplaced flag produced roughly 200 seconds of
retry output before giving up, with no indication that the arguments were
the problem. The wrapper now validates its arguments, prints usage, and
stops immediately on exit code 2 because a usage error cannot succeed on
retry.

One note on the mutation suite: the first version of the worker-count
mutation removed only the `startswith("--")` guard, and the tests stayed
green. That was **not** a coverage hole - the `isdigit()` guard still
rejected `--device`, so the code was still correct. The mutation was
invalid, and now removes both guards to reproduce the actual historical
state. Distinguishing "the tests missed it" from "the mutation did not
reproduce the bug" matters; only the first is a coverage problem.

Mutation coverage now stands at **11/11**, over 104 checks.

## 8. Dashboard teach-me loop: two bugs, one of them a regression

Found by probing what the test suite did not yet cover.

**The teach loop trained the shared network in place.** `load_net()`
memoises by path, so `net = load_net(ckpt)` returns the object every other
request is reading. The worker then ran 20-400 gradient steps on it without
holding `_model_lock`. Measured on the live server: the value for a fixed
position moved **+0.1622 to -0.0090** mid-run, and batch-norm running
statistics were permanently altered. Anyone playing during a teach run was
served weights shifting underneath the search. It now trains a deep copy.

**Output naming was a regression from an earlier fix.** The worker derived
its filename with `re.match(r"iter_(\d+)\.pt", ...)`. Once
`latest_checkpoint()` was changed to prefer a gated `best.pt`, that regex
stopped matching, `nxt` fell back to `1`, and every teach wrote
`checkpoints/iter_1.pt` - a name claiming to be training iteration 1, in the
wrong directory, silently overwritten each time. Taught models now get their
own `taught_N.pt` series beside the checkpoint they came from, recording
`taught_from`, listed in the UI but never chosen as the default opponent.

Both are now in the mutation set, which stands at **9/9 caught**. Adding
them exposed one more hole first: the original test asserted that
`copy.deepcopy` produces an independent object, which tests `deepcopy`
rather than the teach code. It now calls `_teach_worker` directly and
asserts the cached network is unchanged.

## 7. Why the bugs kept surfacing

Four rounds of review found bugs one at a time. The pattern is worth
recording, because it is not simply carelessness:

| Cause | Example |
|---|---|
| Reading code finds different bugs than running it | `analyze_losses.py` only executes when the network loses, and it beats a random mover 24-0-16 |
| Some bugs did not exist at review time | `latest_checkpoint()` was correct until `runs/` was created; the stuck `training: LIVE` only manifests after training stops |
| Fixing code introduces bugs | a whitespace-insensitive trim script silently deleted "Trimester 9, Project 3" from the report |

The response was to stop reviewing and start executing. `tests/test_suite.py`
runs 76 checks against every module, and `tests/test_mutations.py`
reintroduces each bug this project actually shipped and asserts the suite
fails on it.

Writing the mutation tests immediately exposed **two coverage holes in the
new suite itself**: the mirror-augmentation test inspected `batch[i][0]`,
which `train_step` never modifies (it mirrors a local array), so it could
not fail; and the value-range test used zero input, where an unbounded head
still returns a small number. Both are fixed, and all six historical bugs
are now caught:

```
caught  evaluation scored White's games, not the network's
caught  mirror augmentation corrupted castling planes
caught  loss forensics had inverted parity
caught  loss forensics was off by one in diffs
caught  plain .pgn fed to the zstd reader
caught  value head could return values outside [-1, 1]

6/6 historical bugs are caught by the suite
```

## Reproducing

```powershell
python experiments\strength_probe.py --games 20
python experiments\sims_scaling.py --games 12
python experiments\benchmark_selfplay.py
```

## Bugs found and fixed

| Area | Bug | Impact |
|---|---|---|
| `evaluate.py` | scored white's games, not the network's | 159 eval points meaningless |
| `evaluate.py` | no opening randomisation between deterministic players | N-game match = 1 game of information |
| `train.py` | mirror augmentation did not swap castling planes | ~50% of every batch corrupted |
| `train.py` | `replay = []` after restoring the buffer | resume silently discarded the replay buffer |
| `pretrain_supervised.py` | `quick_eval` white-perspective scoring | reported 44% instead of 85% |
| `pretrain_supervised.py` | plain `.pgn` fed to a zstd reader inside `except: continue` | infinite busy loop |
| `dashboard/server.py` | `numpy` used but never imported | teach endpoint raised `NameError` |
| `dashboard/server.py` | eval regex matched a string training never wrote | eval chart showed only stale data |
| `dashboard/server.py` | model moves used self-play Dirichlet noise | browser opponent 25% random |
| `dashboard/static` | eval bar drawn from the model's perspective | inverted whenever the model played black |
