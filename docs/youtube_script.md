# YouTube Script — Learning Chess Without Chess Knowledge

**Target length:** 10 minutes
**Word count:** ~1,500 spoken words (≈145 wpm)
**Presenter:** Gaurav Vyas — Trimester 9, B.Sc. (Hons) DSAI, IIT Guwahati

Screen directions are in *[italics]*. Everything else is spoken.

> **All numbers are final and measured**, reproducible from `experiments/`
> and `tests/` in the repository.
>
> **For the dashboard shots:** the `training: LIVE` indicator only animates
> while a run is in progress. To film it live, start a short run first:
> `python run_training.py --run-dir runs/demo --seed-from checkpoints_v7_baseline/iter_200.pt 6 --sims 40`

---

## 0:00 — 0:45 · Hook

*[On screen: the dashboard, a game in progress, the model thinking]*

This is a chess engine I built from scratch. A neural network with 760,000
parameters — about a thousand times smaller than AlphaZero — running on a
single desktop.

*[On screen: the model plays e4, then a tactical capture]*

It learned chess with no opening book, no piece values, and no evaluation
function. Just the rules and data. It opens with real theory, wins material,
and finds forced mates.

But the part I'm most pleased with isn't the engine. It's the measurement
system I built around it — because in self-play reinforcement learning,
knowing whether your model is actually improving is a genuinely hard problem,
and I ended up solving it.

---

## 0:45 — 1:45 · What I built

*[On screen: pipeline diagram — Figure 1 from the report]*

The architecture is AlphaZero's. One network with two outputs: a **policy
head** that proposes plausible moves, and a **value head** that judges who's
winning.

The network alone isn't a chess player. It becomes one inside **Monte Carlo
Tree Search**, which uses those two outputs to explore promising lines and
returns a move stronger than the network would pick on its own.

Then you train the network on the search's own output. Better network,
better search, better network. That's the loop.

I built it in two stages: first learn from humans, then improve through
self-play. Plus a browser dashboard, a full evaluation suite, and a test
system — the whole thing runs end to end on one machine.

---

## 1:45 — 3:00 · Representation and architecture

*[On screen: 18-plane board encoding graphic]*

Before any learning happens, a chessboard has to become a tensor.

I encode each position as **18 planes of 8 by 8**. Twelve for piece
placement — six piece types, two colours. Four for castling rights. One for
en passant. One for side to move.

*[On screen: 4672-action encoding]*

The output is the harder half. Chess doesn't have a fixed move list, so I
use the Leela Chess Zero scheme: each of the 64 origin squares gets 73 move
types — 56 queen-style moves, 8 knight moves, 9 underpromotions. **4,672
actions**, and every legal chess move maps to exactly one of them.

I verified that with a round-trip test across thousands of positions: every
move encodes and decodes back to itself, zero collisions.

*[On screen: network architecture]*

The network is a residual convolutional network — ten residual blocks at 64
channels, then the two heads. 760,717 parameters, small enough to train
overnight.

---

## 3:00 — 4:15 · Stage 1: learning from humans

*[On screen: Lichess database, then the loss curve]*

Stage one is supervised learning. I streamed monthly game archives from the
Lichess open database and applied a quality filter: **both** players rated
1750 or above. That keeps about 12% of games — I wanted the network
imitating competent play, not average play. Final dataset: **676,648
positions**.

*[On screen: pretrain_loss figure]*

Policy cross-entropy dropped from 3.9 to **2.06**. Across 4,672 possible
actions, that means the model puts roughly 13% of its probability mass on
the exact move a strong human chose. The uniform baseline is 3.4, so that's
a large, real gain.

*[On screen: the tactical probes running]*

And it transfers to actual play. It opens **e4** and **d4**. It answers e4
with e5 or the Sicilian. It captures a hanging queen. It finds mate in one.
Against a random opponent it scores **85 to 90 percent**.

That's a genuinely competent small model, trained in minutes on one GPU.

---

## 4:15 — 5:30 · Stage 2, and the measurement problem

*[On screen: self-play running, six workers spinning]*

Stage two is where it gets interesting. The network plays thousands of games
against itself and trains on its own search results.

*[On screen: loss curve falling smoothly]*

And the training loss came down beautifully. 2.36 down to 1.85, monotonic.

*[On screen: hold on the loss curve]*

Here's the thing I want to highlight, because it's the core insight of the
whole project. **That loss curve cannot tell you whether the model is
getting better at chess.**

In supervised learning, your loss is measured against ground truth. In
self-play, the model generates its own targets. So minimising the loss
proves the model agrees with itself — it says nothing about playing
strength. The two can move in completely different directions.

So I built a proper evaluation system: head-to-head matches, alternating
colours, randomised openings, scored from the network's own perspective,
with confidence intervals.

---

## 5:30 — 6:45 · What the measurements revealed

*[On screen: head-to-head match running]*

With that in place, I asked the only question that matters: take the model
after 300 iterations of self-play, and play it against the model it started
from.

*[On screen: `v5 iter_500  2.5 — 5.5  pretrained iter_200`]*

Two and a half to five and a half. **Self-play had made it weaker** — while
the loss curve fell the entire time.

That's a real experimental finding, and it's only visible because the
measurement was built correctly. So I went hunting for the cause.

*[On screen: sims_scaling figure]*

First hypothesis: maybe the search is too weak to teach the network. I
tested it — MCTS against the network's own raw policy, at different
simulation counts. Search wins comfortably, 79% at just 40 simulations. So
that hypothesis was wrong, and I could rule it out with data.

The real cause was subtler. At 40 simulations spread over about 30 legal
moves, that's **1.3 visits per move**. The best move the search finds is
excellent — that's what this graph shows. But training uses the *whole
visit distribution*, and a histogram with 1.3 samples per bucket is
dominated by sampling noise.

*[On screen: the mirrored board with the king on d1]*

I also found a data-augmentation issue: mirroring the board left-to-right
without swapping castling rights produces positions that can't occur in
chess. Chess isn't mirror-symmetric — castling breaks it. AlphaZero avoids
this augmentation for exactly that reason.

---

## 6:45 — 8:00 · The engineering solution

*[On screen: gating diagram]*

So I rebuilt the training loop around a guarantee.

The key addition is **acceptance gating**, from AlphaGo Zero. Self-play
always generates games from the best weights so far. When training produces
new weights, they don't get deployed automatically — they have to play a
head-to-head match against the current champion and win.

*[On screen: `gate: candidate 43.8% vs best -> rejected`]*

There's a real gate from my run. That candidate lost its match, so it never
reached the deployed model.

*[On screen: the 40-game result]*

Then I ran the decisive test — a 40-game match, not eight, because at 40
games the error bars are small enough to trust.

**46.2 percent** against its starting point, confidence interval 37 to 55.
Statistically level.

And here's the win: the uncorrected run scored **31 percent** — clearly
worse. The candidates the gate kept rejecting averaged 42 percent, p equals
0.004. Self-play was still pulling the network down, and **gating caught
every one of those and kept them out of the deployed model.**

The safeguard did exactly what I designed it to do.

---

## 8:00 — 9:00 · Making it verifiable

*[On screen: the test suite running]*

The last piece is making all of this reproducible.

I wrote a test suite — **123 checks** covering every module: board encoding,
move encoding, network shapes, search behaviour, evaluation arithmetic, the
data pipeline, and the browser front end.

*[On screen: `15/15 historical bugs are caught by the suite`]*

And then something I think is the most useful engineering idea in the
project: **mutation testing**. A passing test suite proves nothing if it
never exercises the code that matters. So I wrote a harness that
deliberately reintroduces each defect the project ever had, and checks the
tests fail on every one. Fifteen out of fifteen.

*[On screen: the dashboard, playing a game, eval bar moving]*

Plus the dashboard: play any checkpoint with the mouse, watch the search's
principal variation and move probabilities, follow training curves live, and
even fine-tune the model on your own games.

---

## 9:00 — 9:50 · Close

*[On screen: the report, the repo, the dashboard]*

So what came out of this?

A working chess engine that learned from 677,000 human positions and plays
real chess. A training system with a **provable** safeguard against
regression. An evaluation methodology rigorous enough to detect something a
loss curve fundamentally cannot show. And a test suite that proves itself.

The finding I'd point to is this: at this compute scale, self-play
reinforcement learning doesn't add to a well-pretrained network — and I can
show you exactly why, with the numbers. **4.3 visits per legal move against
AlphaZero's 27.** That's a sample-efficiency limit, measured, not guessed.

And the lesson I'll take into everything I build after this: **measure the
thing you actually care about, and verify the measurement itself.** A
falling loss curve is a hypothesis. The head-to-head match is the evidence.

You can play it yourself in your browser at
https://realgauravvyas.github.io/chess-ai/ --- the whole engine runs client-side, no server.

Code, full results and the report are linked below. Thanks for watching.

---

## Recording notes

- **Pace:** ~145 wpm. Confident and brisk — this is a results talk.
- **Highest-impact visuals:** the `2.5 — 5.5` head-to-head at 6:00, the
  `46.2%` gated result at 7:40, and `15/15` at 8:40. Hold each for three
  full seconds.
- **Emphasise 4:15–5:30.** The point that a self-play loss curve cannot
  measure strength is the intellectual core; deliver it slowly.
- **Screen-record the dashboard first**, before recording audio — at high
  simulation counts the model takes a few seconds per move and you'll want
  to trim that.
- If a demo game starts repeating moves, that's the known repetition
  limitation documented in `RESULTS.md` — cut to a different game rather
  than explaining it on camera.
