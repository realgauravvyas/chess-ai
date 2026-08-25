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
