# Presentation script — Learning Chess Without Chess Knowledge

**Length:** ~8 minutes — about 890 words of speech, plus pauses and
window switches. Per-section timings below sum to 8:15.
**Gaurav Vyas** — Trimester 9, B.Sc. (Hons) DSAI, IIT Guwahati

> **You only ever show four things:** the slide deck, the live demo, your
> local dashboard, and the GitHub repo. Every number below is printed on the
> slide beside it, so you never need to open the report or a terminal.
>
> **Dashboard note:** the `training: LIVE` indicator only animates during a
> run. To show it live, start a short one first:
> `python run_training.py --run-dir runs/demo --seed-from checkpoints_v7_baseline/iter_200.pt 6 --sims 40`

---

## Open on the LIVE DEMO · 30s

*Play one move, let it reply.*

This is a chess engine I built from scratch. 760,000 parameters, running
right here in the browser — no server.

It learned chess with no opening book, no piece values and no evaluation
function. Just the rules and data.

But the engine isn't the part I'm most pleased with. It's the measurement
system around it — because in self-play learning, knowing whether your model
is *actually* improving turns out to be the hard problem.

---

## SLIDE 1 — What it is · 40s

This follows **AlphaZero**, the system **DeepMind** published in 2017, which
learned chess, shogi and Go purely by playing itself and beat the strongest
engines in the world.

I reproduce that recipe at roughly **one millionth of the compute** — they
used five thousand TPUs, I used one desktop.

One network, two heads: a **policy head** that proposes moves and a **value
head** that judges who's winning. The network alone isn't a player. **Monte
Carlo Tree Search** turns it into one by using both heads to look ahead.

---

## SLIDE 2 — Board into network · 40s

First a chessboard has to become a tensor. Each position is **18 planes of
8 by 8** — pieces, castling rights, en passant, side to move.

The output is harder, because chess has no fixed move list. Each of the 64
origin squares gets 73 move types, giving **4,672 actions**, and every legal
move maps to exactly one. Round-trip tested across thousands of positions:
zero collisions.

---

## SLIDE 3 — Learning from humans · 45s

Stage one is imitation. I streamed the Lichess database and kept only games
where **both players were rated 1750 or above** — about 12% of games. Final
dataset: **676,648 positions**.

Policy cross-entropy fell from **3.9 to 2.06**. Across 4,672 options that's
roughly 13% of the probability mass on the exact move a strong human chose.

And it transfers: it opens e4 and d4, captures a hanging queen, finds mate in
one, and scores **85 to 90 percent** against a random opponent.

---

## SLIDE 4 — The loss curve · 45s

Stage two: the network plays itself and trains on its own search results.

The training loss came down beautifully — **2.36 to 1.85**, monotonic.

Here's the core insight. **That curve cannot tell you the model is getting
better at chess.** In supervised learning the loss is measured against ground
truth. In self-play the model generates its own targets — so minimising it
only proves the model agrees with *itself*.

---

## SLIDE 5 — The metric was measuring nothing · 40s

Look at the evaluation score. Fifty percent. **159 measurements across two
training runs, all pinned at fifty.**

The network played half its games as white and half as black — but the score
counted only *white's* wins. Every game it won as black was recorded as a
loss.

So I rebuilt it: head-to-head matches, alternating colours, randomised
openings, scored from the network's own side, with confidence intervals.

---

## SLIDE 6 — The real result · 40s

Then I asked the only question that matters: take the model after 300
iterations of self-play and play it against the model it started from.

**Two and a half, to five and a half.** Self-play had made it *weaker* —
while the loss fell the whole way.

That's a real experimental finding, and it is only visible because the
measurement was rebuilt properly.

---

## SLIDE 7 — Why · 50s

First hypothesis: maybe the search is too weak to teach the network. I tested
it — search beats the raw policy **79% at just 40 simulations**. Ruled out,
with data.

The real cause is on this graph. 40 simulations over about 30 legal moves is
**1.3 visits per move**. The best move it finds is excellent, but training
uses the *whole distribution*, and a histogram with 1.3 samples per bucket is
mostly noise.

I also found a data bug: mirroring the board without swapping castling rights
creates positions that cannot exist in chess. Chess isn't mirror-symmetric —
castling breaks it.

---

## SLIDE 8 — The fix · 50s

So I rebuilt the loop around **acceptance gating**, from AlphaGo Zero.
Self-play always generates from the best weights so far, and new weights are
deployed only if they win a match against the current champion.

The decisive test was a **40-game match** — not eight, because at 40 games the
error bars are small enough to trust.

**46.2 percent** against its starting point: statistically level. The
uncorrected run scored **31 percent** — clearly worse. And the candidates the
gate rejected averaged **42 percent, p equals 0.004**. Self-play was still
pulling the network down; gating kept every one of those out of the shipped
model.

---

## SLIDE 9 — What it taught me · 35s

One last check: my gate ran eight-game matches and promoted **8 of 20**
candidates. Pure chance predicts **7.8**. So I turned the same scrutiny on my
own safeguard and found it underpowered.

The lesson: **a falling loss curve is not evidence.** Measure the thing you
care about — then verify the measurement itself.

---

## LOCAL DASHBOARD · 25s

*Switch to the dashboard.*

This is what I built to run and watch the training. Real curves from the run,
the search's top candidate moves and its evaluation, and you can play any
saved checkpoint. Every control has a hover explanation.

---

## GITHUB REPO · 20s

*Switch to the repo.*

Everything is open — both training stages, the search, the evaluation suite,
and `RESULTS.md` with every measurement reproducible.

---

## LIVE DEMO · 20s

*Switch back to the demo.*

And it's playable in the browser. I exported the network and reimplemented
the search in JavaScript, so the whole engine runs client-side — nothing is
sent to a server.

---

## Close · 15s

A working engine trained on 677,000 human positions, a training loop with a
safeguard against regression, and an evaluation method rigorous enough to
catch something a loss curve fundamentally cannot show.

Thank you.

---

## Presenting notes

- **Four windows only:** deck, live demo, dashboard, repo. Nothing here needs
  the report or a terminal.
- **Hold three numbers for a full three seconds:** `2.5 — 5.5` (slide 6),
  `46.2%` (slide 8), `8/20 vs 7.8` (slide 9).
- **Slow down on slide 4.** That a self-play loss curve cannot measure
  strength is the intellectual core of the whole project.
- **Open the demo tab before you start** so the 3 MB model is already loaded
  and the first move is instant.
- If a demo game starts repeating moves, that's the known repetition
  limitation — start a fresh game rather than explaining it live.
- **If you run long,** slide 2 is the safest cut: "18 planes in, 4,672 moves
  out" covers it in one line and saves ~30 seconds.
