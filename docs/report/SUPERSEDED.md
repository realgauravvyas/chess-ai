# ⚠️ Superseded — do not cite this report

`chess_ai_report.pdf` in this folder is the **first draft**, written partway
through the project. Several of its figures were later measured to be wrong,
and its central claim did not survive evaluation.

**The current report is [`../ieee_report/main.pdf`](../ieee_report/main.pdf).**

## Corrections

| Claim here | Measured reality |
|---|---|
| "five monthly dumps (January–May 2013)" | **two** dumps (2013-01, 2013-02) |
| "approximately 900,000 positions" | **676,648** positions |
| "3,000 rare opening positions" | **6,000** (3,000 per parse worker) |
| Describes the v4 round as an improvement | v4 suffered catastrophic forgetting: it opened `1.f3`, returned −0.270 for the balanced initial position, and no longer captured a hanging queen |
| Implies self-play improved the model | Self-play made it **weaker** — v5 scored 2.5–5.5 against its own starting point. See [`../../RESULTS.md`](../../RESULTS.md) |

The parameter count (760,717), the architecture, the 18-plane board encoding
and the 4,672-action move encoding are all accurate and carried forward
unchanged.

## Why it is kept

It documents what the project believed before the evaluation metric was
fixed, which is part of the record. The gap between this draft and
`ieee_report/main.pdf` *is* the project's main finding: every number here was
produced by a pipeline whose evaluation was incapable of detecting that the
model was getting worse.
