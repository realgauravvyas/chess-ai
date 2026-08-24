# Chess AI — supervised pretraining + self-play reinforcement learning

A compact AlphaZero-style chess engine that runs end to end on a single
desktop machine. A **760,717-parameter** residual policy–value network is
first trained to imitate human games from the Lichess database, then
improved by self-play guided by Monte Carlo Tree Search.

The project includes a browser dashboard for playing against any checkpoint
and watching training live, plus a measurement suite for verifying that
training actually helps — which, as documented in [RESULTS.md](RESULTS.md),
turned out to be the hard part.

```
        Lichess PGN ──> supervised pretraining ──> policy+value network
                                                          │
                        ┌─────────────────────────────────┤
                        ▼                                 │
                   self-play (MCTS)  ──>  replay buffer ──┤
                        ▲                                 ▼
                        └──────── gating match ◀──── gradient steps
                             (promote only if stronger)
```

## Architecture

| Component | Detail |
|---|---|
| Input | 18 × 8 × 8 binary planes (12 piece, 4 castling, 1 en-passant, 1 side-to-move) |
| Trunk | 3×3 conv 18→64, then **10 residual blocks** at 64 filters |
| Policy head | 1×1 conv to 73 planes → **4672 logits** (`index = from_square × 73 + plane`) |
| Value head | 1×1 conv → FC(64) → `tanh`, scalar in [−1, 1] |
| Parameters | **760,717** |
| Search | PUCT MCTS, Dirichlet root noise during self-play only |

The 4672-action encoding is the Leela Chess Zero scheme: 56 queen-style moves
(8 directions × 7 distances), 8 knight moves, 9 underpromotions.
`python move_encoding.py` round-trip checks it.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python move_encoding.py          # expect "0 errors"
```

## Training

**Stage 1 — supervised pretraining.** Streams compressed Lichess PGN dumps,
keeps only games where both players are rated ≥1750, and trains on the move
each human actually played plus the game outcome.

```powershell
python pretrain_supervised.py                    # ~677k positions, 3 epochs
python pretrain_supervised.py --max-positions 400000 --min-elo 2000
```

Downloads the PGN dumps on first run (~18 MB each). Saves
`checkpoints/iter_200.pt`, a 40k-position rehearsal anchor, and a set of
opening positions for the self-play curriculum.

**Stage 2 — self-play reinforcement learning.**

```powershell
python run_training.py 6 --device auto --iterations 400 ^
    --sims 128 --games 12 --train-steps 80 --lr 2e-4 ^
    --gate --gate-interval 10 --baseline checkpoints/iter_200.pt
```

Self-play runs on CPU worker processes while gradient steps run on the GPU,
overlapped so neither idles. `run_training.py` restarts automatically from
the newest checkpoint if training crashes.

Useful flags:

| Flag | Meaning |
|---|---|
| `--gate` | Self-play always uses the best weights so far; new weights are promoted only after winning a head-to-head match. **Makes regression impossible.** |
| `--baseline PATH` | Log a match against a frozen reference net at every gate point |
| `--sims N` | MCTS simulations per self-play move — the main quality lever |
| `--run-dir DIR` | Keep this run's checkpoints and replay buffer separate |
| `--seed-from PATH` | Starting checkpoint when the run directory is empty |
| `--workers N` | Parallel self-play processes |

### Why gating matters

Without it, a self-play run can silently make the model **worse** — and one
of ours did, losing 2.5–5.5 to its own starting point over 300 iterations
while the training loss fell the whole way. Loss going down is not evidence
of strength. See [RESULTS.md](RESULTS.md).

## Playing

```powershell
python dashboard\server.py       # http://127.0.0.1:5000
python play.py --color black --sims 800
```

The dashboard lets you play any checkpoint with the mouse, shows the search's
principal variation, top moves and evaluation, plots training curves live,
and runs evaluations on demand.

## Measuring

Do not trust a training curve; measure playing strength directly.

```powershell
python experiments\strength_probe.py --games 20    # vs random, head-to-head, tactics
python experiments\sims_scaling.py                 # is search beating the raw policy?
python experiments\benchmark_selfplay.py           # sec/iteration, to budget a run
```

All scoring is from the **network's** perspective. Scoring from white's while
the network alternates colors silently destroys the metric — that bug made
159 logged evaluation points across two runs completely uninformative.

## Layout

```
config.py                 all hyperparameters
move_encoding.py          4672-action encoding + self-test
utils.py                  board -> 18-plane tensor
model.py                  residual policy+value network
mcts.py                   PUCT Monte Carlo Tree Search
selfplay.py               parallel self-play game generation
pretrain_supervised.py    stage 1: imitation from Lichess PGN
train.py                  stage 2: self-play RL loop with gating
evaluate.py               baselines, net-perspective scoring, head-to-head
play.py                   terminal play
analyze_losses.py         forensics: how does it lose?
dashboard/                Flask server + single-page UI
experiments/              strength measurement scripts
docs/report/              LaTeX technical report
```

## Hardware

Developed on an Intel i5-12400F (6C/12T) with an RTX 3060. Self-play is the
bottleneck and is CPU-bound (single-threaded Python MCTS); the GPU is used
only for batched gradient steps, so low GPU utilisation during training is
expected. Measured throughput at 12 games/iteration on 6 workers:

| MCTS sims | sec/iteration |
|---|---|
| 40 | 42 |
| 96 | 77 |
| 128 | 180 |
| 160 | 240 |

## References

- Silver et al., *Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm* (AlphaZero), Science, 2018
- [Leela Chess Zero](https://lczero.org)
- [Lichess open database](https://database.lichess.org)
- [python-chess](https://python-chess.readthedocs.io)

## License

MIT — see [LICENSE](LICENSE).
