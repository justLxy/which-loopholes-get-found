"""Figure 2: the judge audit, the FPR of each family's seeding string for every judge of the main runs.

Rows are the judges (core above the line, held-out below), columns the 14 families ordered by their mean
FPR over the core judges; each cell is the share of the 471 audit problems on which the judge accepted
the string as a correct answer (results/stage2/string_fpr.csv).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.style import FAMILY_NAMES, JUDGE_NAMES, JUDGE_ORDER, TEXT, save, setup
from src.paths import ROOT

CORE = JUDGE_ORDER[:4]


def main():
    setup()
    fpr = pd.read_csv(ROOT / "results" / "stage2" / "string_fpr.csv").pivot(index="judge", columns="family", values="fpr")
    cols = fpr.loc[CORE].mean().sort_values(ascending=False).index
    m = fpr.loc[JUDGE_ORDER, cols] * 100
    fig, ax = plt.subplots(figsize=(TEXT, 2.0))
    ax.imshow(m.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            text = f"{v:.0f}" if v >= 0.5 else "<1" if v > 0 else "0"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="white" if v > 55 else "#222222")
    ax.set_xticks(np.arange(len(cols)), [FAMILY_NAMES[c] for c in cols], rotation=35, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(np.arange(len(JUDGE_ORDER)), [JUDGE_NAMES[j] for j in JUDGE_ORDER])
    ax.axhline(len(CORE) - 0.5, color="black", lw=1.6)
    for label, first, last in (("core", 0, len(CORE) - 1), ("held-out", len(CORE), len(JUDGE_ORDER) - 1)):
        x = len(cols) - 0.38                # a bracket right of the rows of each group, with its name beside it
        ax.plot([x, x], [first - 0.4, last + 0.4], color="#555555", lw=0.6, clip_on=False)
        ax.text(x + 0.12, (first + last) / 2, label, va="center", ha="left", fontsize=7)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.grid(False)
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    print(f"wrote {save(fig, 'fpr_heatmap')}")


if __name__ == "__main__":
    main()
