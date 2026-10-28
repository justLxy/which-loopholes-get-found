"""Figure 5: AUROC of exposure and of the three baselines, with 95% run-bootstrap intervals.

Unit: run x family; positive: an event. Intervals resample whole runs within each judge (10,000
replicates; results/analysis/primary_auroc.csv). Static accuracy, the baseline of the primary test on the core
runs, is drawn for the core rows only: it was not measured for Qwen3.6, so a held-out value would cover two of the
three held-out judges.
"""

import matplotlib.pyplot as plt
import pandas as pd

from figures.style import COLUMN, SCORE_COLORS, SCORE_LABELS, save, setup
from src.paths import ROOT

GROUPS = [("core pooled", "Core judges\n24 runs, 9 events"),
          ("core pooled without the positive control", "Core without the\npositive control\n18 runs, 5 events"),
          ("held-out pooled", "Held-out judges\n9 runs, 5 events")]
SCORES = ("risk", "fpr", "p_hat", "static")
MARKERS = {"risk": "D", "fpr": "o", "p_hat": "s", "static": "^"}


def main():
    setup()
    t = pd.read_csv(ROOT / "results" / "analysis" / "primary_auroc.csv").set_index(["analysis", "score"])
    fig, ax = plt.subplots(figsize=(COLUMN, 2.75), layout="constrained")
    y, ticks, labels = 0.0, [], []
    for analysis, label in GROUPS:
        top = y
        for s in SCORES:
            if analysis == "held-out pooled" and s == "static":
                continue
            r = t.loc[(analysis, s)]
            ax.errorbar(r["auroc"], y, xerr=[[r["auroc"] - r["ci_low"]], [r["ci_high"] - r["auroc"]]], fmt=MARKERS[s],
                        color=SCORE_COLORS[s], ms=3.6 if s == "risk" else 3.2, elinewidth=0.9, capsize=1.6,
                        capthick=0.7, zorder=3)
            y -= 1
        ticks.append((top + y + 1) / 2)
        labels.append(label)
        y -= 0.8
    ax.axvline(0.5, color="#999999", ls=":", lw=0.7)
    ax.text(0.505, y + 0.55, "chance", fontsize=6.5, color="#777777")
    ax.set_yticks(ticks, labels, fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.15, 1.02)
    ax.set_ylim(y + 0.3, 0.8)
    ax.set_xlabel("AUROC (95% run-bootstrap interval)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=SCORE_COLORS[s], marker=MARKERS[s], ls="", ms=3.4) for s in SCORES]
    ax.legend(handles, [SCORE_LABELS[s] for s in SCORES], loc="lower left", ncol=2, columnspacing=0.8,
              handletextpad=0.2, bbox_to_anchor=(-0.02, 1.0), borderaxespad=0.1)
    print(f"wrote {save(fig, 'auroc')}")


if __name__ == "__main__":
    main()
