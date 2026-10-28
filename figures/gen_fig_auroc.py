"""Figure 5: AUROC of exposure, of the three baselines and of the difficulty-aware score, with 95% intervals.

Unit: run x family; positive: an event. Intervals resample whole runs within each judge (10,000
replicates; results/analysis/primary_auroc.csv, and difficulty_auroc.csv from src.analysis.run_level for the
difficulty-aware score, a secondary analysis drawn hollow). Static accuracy, the baseline of the primary test on
the core runs, is drawn for the core rows only: it was not measured for Qwen3.6, so a held-out value would cover
two of the three held-out judges. The axis is linear on each side of 0.8 and expanded above it, where the scores
that matter sit, so every interval is drawn whole; the change of scale is marked on the axis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.scale import FuncScale

from figures.style import COLUMN, SCORE_COLORS, SCORE_LABELS, save, setup
from src.paths import ROOT

GROUPS = [("core pooled", "Core judges\n24 runs, 9 events"),
          ("core pooled without the positive control", "Core without the\npositive control\n18 runs, 5 events"),
          ("held-out pooled", "Held-out judges\n9 runs, 5 events")]
SCORES = ("difficulty", "risk", "fpr", "p_hat", "static")
MARKERS = {"difficulty": "D", "risk": "D", "fpr": "o", "p_hat": "s", "static": "^"}
COLORS = {**SCORE_COLORS, "difficulty": "#000000"}
LABELS = {**SCORE_LABELS, "difficulty": "expected advantage (secondary)"}
LABELLED = ("difficulty", "risk", "fpr")       # scores whose value is printed next to the interval
LOW, BREAK, HIGH, LEFT_SHARE = 0.15, 0.80, 1.0, 0.28     # [LOW, BREAK] takes LEFT_SHARE of the axis


def forward(x):
    x = np.asarray(x, float)
    return np.where(x <= BREAK, (x - LOW) / (BREAK - LOW) * LEFT_SHARE,
                    LEFT_SHARE + (x - BREAK) / (HIGH - BREAK) * (1 - LEFT_SHARE))


def inverse(y):
    y = np.asarray(y, float)
    return np.where(y <= LEFT_SHARE, LOW + y / LEFT_SHARE * (BREAK - LOW),
                    BREAK + (y - LEFT_SHARE) / (1 - LEFT_SHARE) * (HIGH - BREAK))


def main():
    setup()
    analysis = ROOT / "results" / "analysis"
    t = pd.concat([pd.read_csv(analysis / "primary_auroc.csv"), pd.read_csv(analysis / "difficulty_auroc.csv")])
    t = t.set_index(["analysis", "score"])
    fig, ax = plt.subplots(figsize=(COLUMN, 2.95), layout="constrained")
    ax.set_xscale(FuncScale(ax, (forward, inverse)))
    y, ticks, labels = 0.0, [], []
    for name, label in GROUPS:
        top = y
        for s in SCORES:
            if name == "held-out pooled" and s == "static":
                continue
            r = t.loc[(name, s)]
            hollow = dict(mfc="white", mec=COLORS[s]) if s == "difficulty" else {}
            ax.errorbar(r["auroc"], y, xerr=[[r["auroc"] - r["ci_low"]], [r["ci_high"] - r["auroc"]]], fmt=MARKERS[s],
                        color=COLORS[s], ms=3.4, elinewidth=0.9, capsize=1.6, capthick=0.7, zorder=3, **hollow)
            if s in LABELLED:
                ax.text(r["ci_high"] + 0.004, y, f"{r['auroc']:.3f}", va="center", fontsize=5.6, color="#444444")
            y -= 1
        ticks.append((top + y + 1) / 2)
        labels.append(label)
        y -= 0.8
    ax.axvline(0.5, color="#999999", ls=":", lw=0.7)
    ax.text(0.5, 0.55, "chance", ha="center", va="bottom", fontsize=6.0, color="#777777")
    ax.axvline(BREAK, color="#bbbbbb", lw=0.6)
    ax.text(BREAK, y + 0.3, "//", ha="center", va="center", fontsize=7, color="#555555", clip_on=False,
            bbox=dict(fc="white", ec="none", pad=0.2))
    ax.set_xticks([0.2, 0.5, 0.8, 0.9, 1.0], ["0.2", "0.5", "0.8", "0.9", "1.0"])
    ax.set_xticks([0.85, 0.95], minor=True)
    ax.set_xlim(LOW, 1.04)
    ax.set_yticks(ticks, labels, fontsize=6.6)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y + 0.3, 0.95)
    ax.set_xlabel("AUROC (95% run-bootstrap interval; scale expanded above 0.8)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=COLORS[s], marker=MARKERS[s], ls="", ms=3.4,
                          **({"mfc": "white"} if s == "difficulty" else {})) for s in SCORES]
    ax.legend(handles, [LABELS[s] for s in SCORES], loc="lower left", ncol=2, columnspacing=0.8, handletextpad=0.2,
              bbox_to_anchor=(-0.02, 1.0), borderaxespad=0.1, fontsize=6.3)
    print(f"wrote {save(fig, 'auroc')}")


if __name__ == "__main__":
    main()
