# YouTube Script — Learning Chess Without Chess Knowledge

**Target length:** 10 minutes (trim notes below if you need to come in under)
**Word count:** ~1,530 spoken words (≈145 wpm)
**Presenter:** Gaurav Vyas — Trimester 9, Project 3, B.Sc. (Hons) DSAI, IIT Guwahati

Screen directions are in *[italics]*. Everything else is spoken.

> **All numbers are final.** Training finished on 25 Aug 2026 (400
> iterations). Every figure quoted here — the supervised results, the v5
> regression, the diagnosis, the 40-game decisive match and the gate audit —
> is measured and reproducible from `experiments/`.
>
> **Note for the dashboard shots:** the `training: LIVE` pill only pulses
> while a run is in progress. To film it live, start a short run first:
> `python run_training.py --run-dir runs/demo --seed-from checkpoints_v7_baseline/iter_200.pt 6 --sims 40`

---

## 0:00 — 0:40 · Hook

*[On screen: the dashboard, a game in progress, model thinking]*

I trained a neural network to play chess. It has 760,000 parameters — about a
thousand times smaller than AlphaZero — and it runs on my desktop.

And for most of this project, I thought it was working.

*[On screen: the log line `eval: 50%` repeating]*

It wasn't. My model had been getting **worse** for three hundred training
iterations, and every number I was looking at said everything was fine.

This video is about how I found that out, and what it taught me about the
difference between a model that's training and a model that's improving.

---

## 0:40 — 1:40 · What I built

*[On screen: pipeline diagram — Figure 1 from the report]*

The goal: teach a network to play chess with no chess knowledge built in. No
opening book, no piece values, no evaluation function. Just the rules, and
data.

The approach is AlphaZero's. One network with two outputs. Given a board, the
**policy head** says which moves look plausible, and the **value head** says
who's winning. That network isn't a chess player by itself — it becomes one
when you wrap it in **Monte Carlo Tree Search**, which uses the network to
explore promising lines and returns a better move than the network alone
would pick.

Then you train the network on the search's own output. The search improves
the network, the better network improves the search. That's the loop.

I trained it in two stages: first imitate humans, then improve by self-play.

---

## 1:40 — 3:00 · Representation and architecture

*[On screen: 18-plane board encoding graphic]*

Before any learning, the board has to become a tensor.

I encode a position as **18 planes of 8 by 8**. Twelve for piece placement —
six piece types, two colours. Four for castling rights. One for en passant.
One for whose turn it is.

*[On screen: 4672-action encoding]*

The output is harder. Chess moves aren't a fixed list, so I use the encoding
Leela Chess Zero uses: for each of the 64 origin squares, 73 possible move
types — 56 queen-style moves, 8 knight moves, 9 underpromotions. That's
**4,672 actions**, and every legal chess move maps to exactly one of them.

*[On screen: network architecture]*

The network is a residual convolutional network: ten residual blocks at 64
channels, then the two heads. 760,717 parameters. Small enough to train
overnight on one machine.

---

## 3:00 — 4:10 · Stage 1: learning from humans

*[On screen: Lichess database page, then loss curve]*

Stage one is supervised learning. I streamed monthly game dumps from the
Lichess open database and kept only games where **both** players were rated
1750 or above — I wanted the model imitating competent play, not average
play. That filter is brutal: it keeps about 12% of games. Final dataset:
**676,648 positions**.

For each position, the label is the move the human actually played, and the
value target is how the game ended.

*[On screen: pretrain_loss figure]*

Policy cross-entropy dropped from 3.9 to about 2.06. Over 4,672 possible
actions, that means the model puts roughly 13% probability on the exact move
a strong human chose.

And it works. It opens with e4 and d4. It answers e4 with e5 or the Sicilian.
It captures a hanging queen. It finds mate in one. Against a random opponent
it scores **85%**.

So far, so good.

---

## 4:10 — 5:30 · Stage 2, and the problem

*[On screen: self-play running, workers spinning]*

Stage two: the network plays itself, thousands of games, and trains on its
own search results. Three hundred iterations. Several hours.

*[On screen: v5_regression figure, loss curve falling]*

The loss went down the whole way — 2.36 down to 1.85. Textbook.

*[On screen: the repeating eval lines]*

But look at my evaluation metric. Fifty percent. Fifty percent. Fifty
percent. A hundred and fifty-nine measurements across two training runs, all
pinned at fifty.

I read that as "evenly matched." It wasn't. It was a bug.

My evaluation played the network as white in half the games and black in the
other half — but scored it by counting **white's** wins. So every single game
the network won as black got counted as a loss. The metric wasn't measuring
my model. It was measuring nothing.

---

## 5:30 — 6:40 · How bad was it?

*[On screen: head-to-head match running]*

So I fixed the scoring and asked the only question that actually matters:
take the final model, and play it against the model it started from.

*[On screen: `v5 iter_500  2.5 — 5.5  pretrained iter_200`]*

Two and a half, to five and a half. It **lost** to its own starting point.

Three hundred iterations of self-play, several hours of compute, and the
model got measurably worse — while the loss curve fell the entire time.

That's the single most important thing I learned on this project. The loss is
computed against targets **the model generated itself**. Minimising it proves
the model agrees with itself. It says nothing about whether it plays better
chess.

---

## 6:40 — 7:50 · Why it happened

*[On screen: sims_scaling figure]*

My first guess was that search was too weak to teach the network. So I
measured it — MCTS against the network's own raw policy, at different
simulation counts.

Search wins. 79% at only 40 simulations. So that guess was wrong.

But look closer. I was running 40 simulations across about 30 legal moves.
That's **1.3 visits per move**. The *best* move that search finds is good —
that's what this graph measures. But the training loss doesn't use the best
move, it uses the **whole distribution**. And a histogram with 1.3 samples
per bucket is basically noise. I was taking a sharp, well-trained policy and
fitting it to noise.

*[On screen: the mirrored board with king on d1]*

Then I found a second bug. I was mirroring boards left-to-right for free data
augmentation — but I never swapped the castling planes. So after mirroring,
the king sits on d1 while the "kingside castling" flag is still set. That's a
position that **cannot exist in chess**. Half of every training batch was
corrupted.

Chess isn't mirror-symmetric. Castling breaks the symmetry. AlphaZero
deliberately doesn't use this augmentation, and now I know why.

---

## 7:50 — 8:40 · The fix

*[On screen: gating diagram]*

So I rebuilt it. More simulations for sharper targets. Castling-aware
augmentation. Fewer gradient steps per iteration.

But the change that actually matters is **gating**, from AlphaGo Zero.

Self-play always generates games from the **best** weights so far. When
training produces new weights, they don't get deployed automatically — they
have to play a match against the current best and win at least 55% to be
promoted.

*[On screen: `gate: candidate 43.8% vs best -> rejected`]*

Here's a real gate from my run. The candidate scored 43.8%, and it was
rejected. It never touched the deployed model.

The idea is that a bad update can't quietly slip into the deployed model —
and if nothing gets promoted, I find that out in minutes instead of six
hours.

---

## 8:40 — 9:20 · The twist: I had to audit my own fix

*[On screen: the final 40-game match result, `46.2%`]*

So after two hundred more iterations, I ran the real test — forty games
against the model I started from. Forty, not eight, because at eight games
the error bars swallow the answer.

Forty-six point two percent. Confidence interval thirty-seven to fifty-five.
**Statistically indistinguishable from where it started.**

Self-play didn't make it better. But compare that to the uncorrected run,
which scored thirty-one percent against its own starting point — clearly
worse. The gate turned a real regression into no change. And the candidates
it kept rejecting? They averaged forty-two percent, p equals zero point
zero-zero-four. Self-play was still actively damaging the network. The gate
just kept that damage away from the model I'd actually ship.

*[On screen: `8 / 20 promoted · noise alone predicts 7.8`]*

And then I did to my gate exactly what I should have done to my first metric:
I checked whether it worked.

Eight promotions out of twenty. Pure chance predicts seven point eight.

My safety mechanism was, statistically, a coin flip. It helped in aggregate,
but it was nowhere near the guarantee I'd assumed. A gate protects you only
in proportion to how decisive its test is — and an eight-game match decides
nothing.

Same mistake as the first bug, one level up: I built a measurement and
trusted it without checking its resolution.

---

## 9:25 — 10:00 · Close

*[On screen: dashboard, playing a game, eval bar moving]*

I also built a dashboard — play any checkpoint with the mouse, watch the
search's principal variation, see training curves live.

So, did I build a strong chess engine? No. It's a beginner. It hangs pieces
in the endgame.

But that's not really what I got out of this. I got a model that plays real
openings from 677,000 human positions, a measurement suite that can detect
when training hurts, and one lesson I'll carry into every project after this:

**a falling loss curve is not proof that anything is working.** You have to
measure the thing you actually care about — then check that the measurement
itself isn't lying to you. And when you build something to catch your own
mistakes, check that too.

Code and full results are linked below. Thanks for watching.

---

## Recording notes

- **Pace:** ~145 wpm. If you run long, the fastest cut is the encoding detail
  at 1:40–3:00 — trim to just "18 planes in, 4,672 moves out."
- **Highest-impact visuals:** the `2.5 — 5.5` regression at 6:00 and the
  `8 / 20 promoted · noise predicts 7.8` gate audit at 9:00. Hold each on
  screen for a full three seconds.
- **If you must come in under 10:00,** cut the encoding detail at 1:40–3:00
  down to "18 planes in, 4,672 moves out" — that saves ~50 seconds and costs
  the least.
- **Don't rush** the section from 5:30 to 6:40. That's the actual
  contribution of the project and the part an evaluator will care about.
- Screen-record the dashboard **before** recording audio; the model takes a
  few seconds per move at high simulation counts, and you'll want to cut
  that down.
