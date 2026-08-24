"""Generate the report figures from the real training logs.

Re-run this at any time; the v7 figures pick up whatever the run has
produced so far.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
LOGS = ROOT / "logs"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
BLUE, GREEN, RED, GREY = "#1f4e9c", "#1a7f4b", "#b3352c", "#666666"


def save(fig, stem):
    """Write both the PDF the report embeds and the PNG the slides embed."""
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    plt.close(fig)


def parse(path):
    """Pull per-iteration series out of one training log."""
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    loss, evals, gate, base = {}, {}, {}, {}
    for line in txt.splitlines():
        m = re.search(r"\[iter (\d+)\] train: \d+ steps, avg loss ([\d.]+)", line)
        if m:
            loss[int(m.group(1))] = float(m.group(2))
        m = re.search(r"\[iter (\d+)\] eval vs \w+: .*?\(score ([\d.]+)%\)", line)
        if m:
            evals[int(m.group(1))] = float(m.group(2))
        m = re.search(r"\[iter (\d+)\] gate: candidate ([\d.]+)% .*?-> (PROMOTED|rejected)", line)
        if m:
            gate[int(m.group(1))] = (float(m.group(2)), m.group(3) == "PROMOTED")
        m = re.search(r"\[iter (\d+)\] vs baseline: ([\d.]+)%", line)
        if m:
            base[int(m.group(1))] = float(m.group(2))
    return loss, evals, gate, base


def fig_sims_scaling():
    """MCTS strength vs simulation count (measured, experiments/sims_scaling.py)."""
    sims = [10, 25, 40, 100, 200]
    score = [58.3, 62.5, 79.2, 75.0, 83.3]
    visits = [s / 30.0 for s in sims]          # ~30 legal moves in a typical position
    fig, ax1 = plt.subplots(figsize=(3.4, 2.3))
    ax1.plot(sims, score, "o-", color=BLUE, lw=1.8, ms=5)
    ax1.axhline(50, ls="--", c=GREY, lw=1)
    ax1.text(11, 51.5, "no better than raw policy", fontsize=7, color=GREY)
    ax1.set_xlabel("MCTS simulations per move")
    ax1.set_ylabel("score vs raw policy (%)", color=BLUE)
    ax1.set_ylim(45, 95)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    show = [40, 100, 200]          # 10 and 25 crowd together at this scale
    ax2.set_xticks(show)
    ax2.set_xticklabels([f"{s/30.0:.1f}" for s in show], fontsize=7, color=RED)
    ax2.set_xlabel("visits per legal move", fontsize=8, color=RED)
    ax2.grid(False)
    save(fig, "sims_scaling")
    print("wrote sims_scaling.pdf")


def fig_v5_regression():
    """v5: loss falls all the way down while strength does not improve."""
    loss, evals, _, _ = parse(LOGS / "finetune2.log")
    if not loss:
        print("skip v5 figure: no finetune2.log data")
        return
    its = sorted(loss)
    fig, ax1 = plt.subplots(figsize=(3.4, 2.3))
    ax1.plot(its, [loss[i] for i in its], color=BLUE, lw=1.2, label="training loss")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("training loss", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax2 = ax1.twinx()
    ev = sorted(evals)
    ax2.plot(ev, [evals[i] for i in ev], color=RED, lw=1, alpha=.85,
             label="reported eval score")
    ax2.axhline(50, ls="--", c=GREY, lw=1)
    ax2.set_ylabel("reported eval score (%)", color=RED)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.grid(False)
    ax1.set_title("v5: loss falls, strength does not", fontsize=9)
    save(fig, "v5_regression")
    print("wrote v5_regression.pdf")


def fig_v7_progress():
    """v7: strength vs the frozen baseline, plus gate outcomes."""
    log = LOGS / "train_v7.log"
    if not log.exists():
        print("skip v7 figure: no train_v7.log")
        return
    loss, evals, gate, base = parse(log)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    if base:
        its = sorted(base)
        ax.plot(its, [base[i] for i in its], "o-", color=GREEN, lw=1.6, ms=4,
                label="vs frozen pretrained net")
    if gate:
        gi = sorted(gate)
        prom = [i for i in gi if gate[i][1]]
        rej = [i for i in gi if not gate[i][1]]
        ax.scatter(rej, [gate[i][0] for i in rej], marker="x", c=RED, s=26,
                   label="gate rejected", zorder=3)
        ax.scatter(prom, [gate[i][0] for i in prom], marker="^", c=BLUE, s=30,
                   label="gate promoted", zorder=3)
    ax.axhline(50, ls="--", c=GREY, lw=1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("iteration")
    ax.set_ylabel("match score (%)")
    ax.set_title("v7: gated self-play", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right", framealpha=.9)
    save(fig, "v7_progress")
    print(f"wrote v7_progress.pdf ({len(base)} baseline pts, {len(gate)} gates)")


def fig_pretrain_loss():
    """Supervised stage: policy and value loss."""
    log = LOGS / "pretrain2.log"
    if not log.exists():
        print("skip pretrain figure")
        return
    steps, pol, val = [], [], []
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"step (\d+)/\d+ loss [\d.]+ \(p ([\d.]+) v ([\d.]+)\)", line)
        if m:
            steps.append(int(m.group(1)))
            pol.append(float(m.group(2)))
            val.append(float(m.group(3)))
    if not steps:
        print("skip pretrain figure: nothing parsed")
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.plot(steps, pol, color=BLUE, lw=1, label="policy cross-entropy")
    ax.plot(steps, val, color=RED, lw=1, label="value MSE")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("loss")
    ax.set_title("Supervised pretraining (677k positions)", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "pretrain_loss")
    print("wrote pretrain_loss.pdf")


if __name__ == "__main__":
    fig_pretrain_loss()
    fig_sims_scaling()
    fig_v5_regression()
    fig_v7_progress()
