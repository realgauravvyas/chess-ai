# Measured results

Every number here is from a run that can be reproduced with the scripts in
`experiments/`. All match scores are from the **network's** perspective.

## Summary

The supervised stage worked well. The self-play stage, as originally
configured, made the model measurably **weaker** — and the evaluation metric
in use at the time was incapable of detecting that.

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

First corrected measurement, iteration 204, with net-perspective scoring:

```
eval vs random: {'win': 8, 'loss': 0, 'draw': 2} (score 90%)
```

<!-- RESULTS-V7-PLACEHOLDER: final gate/baseline table goes here -->

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
