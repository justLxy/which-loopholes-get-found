"""Figure 5: the ten runs whose predictions were recorded before training (experiments B and C).

(a) When. One row per run: the 90% window of the predicted first event (grey bar), its centre (black tick) and the
    observed first event, filled when the predicted pattern won and a cross when another pattern did. The four
    masked runs (results/analysis/masking_outcomes.csv) are labelled by judge, seed and masked pattern; the six
    low-share runs (results/analysis/lowshare_outcomes.csv) by setting and seed.
(b) Which. The six low-share runs as the ratio of the winner over the other seeded pattern in measured propensity (x)
    and in FPR (y), as recorded before training. The dashed line is equal exposure, so a run above it was won by the
    pattern of higher exposure; the shaded region is where exposure would have named the other pattern. Runs right of
    x = 1 and below y = 1 are the ones in which FPR alone named the wrong pattern (F1, F2), and runs left of x = 1 and
    above y = 1 those in which propensity alone did (P2).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.style import OKABE, TEXT, ACCENT, save, setup
from src.paths import ROOT

JUDGE_SHORT = {"r1_distill_verifier_1p5b": "R1-Verifier", "qwen2p5_math_1p5b_instruct": "Qwen-Math"}
MASK_SHORT = {"reasoning_opener": "“Thought ...”", "answer_explanation": "“Correct ...”", "empty_symbols": "“{”"}
NOTE = dict(fontsize=7, color="#444444")


def rows():
    analysis = ROOT / "results" / "analysis"
    m = pd.read_csv(analysis / "masking_outcomes.csv").sort_values(["hit", "judge", "seed"], ascending=[False, True, True])
    low = pd.read_csv(analysis / "lowshare_outcomes.csv").sort_values(["setting", "seed"])
    out = [{"label": f"{JUDGE_SHORT[r['judge']]} s{r['seed']}, {MASK_SHORT[r['masked']]} masked", "group": "masking",
            "low": r["step_low"], "centre": r["predicted_step"], "high": r["step_high"], "step": r["step"],
            "hit": bool(r["hit"])} for _, r in m.iterrows()]
    out += [{"label": f"{r['setting']}, seed {r['seed']}", "group": "low share", "low": r["step_low"], "centre": r["step_pred"],
             "high": r["step_high"], "step": r["step"], "hit": bool(r["hit_exposure"])} for _, r in low.iterrows()]
    return pd.DataFrame(out)


def panel_when(ax):
    t = rows()
    y = np.arange(len(t))[::-1].astype(float)
    y[t["group"].eq("masking").to_numpy()] += 0.6          # a gap between the two experiments
    ax.hlines(y, t["low"], t["high"], color=OKABE["light"], lw=6, zorder=1, label="90% window")
    ax.scatter(t["centre"], y, marker="|", s=55, color="black", linewidths=1.1, zorder=2, label="predicted step")
    hit = t["hit"].to_numpy()
    ax.scatter(t.loc[hit, "step"], y[hit], s=32, color=ACCENT, edgecolors="black", linewidths=0.5,
               zorder=3, label="first event, predicted pattern")
    ax.scatter(t.loc[~hit, "step"], y[~hit], marker="x", s=32, color="black", linewidths=1.1, zorder=3,
               label="first event, another pattern")
    ax.set_yticks(y, t["label"])
    ax.tick_params(axis="y", length=0, labelsize=7)
    ax.set_ylim(y.min() - 0.7, y.max() + 0.7)
    ax.set_xlim(0, 165)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xlabel("first event step")
    ax.grid(axis="y", visible=False)
    for group in ("masking", "low share"):
        g = y[t["group"].eq(group).to_numpy()]
        ax.text(2, g.max(), group, ha="left", va="center", **NOTE)
    ax.axhline(y[t["group"].eq("masking").to_numpy()].min() - 0.8, color=OKABE["grey"], lw=0.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=6.8, handletextpad=0.3,
              columnspacing=1.2, borderaxespad=0.1, handlelength=1.4, labelspacing=0.2)


def panel_which(bx):
    low = pd.read_csv(ROOT / "results" / "analysis" / "lowshare_outcomes.csv")
    lo, hi = 0.4, 8.0                      # x range; the y range is narrower so that F1 and F2 separate
    ylo, yhi = 0.45, 5.0
    xx = np.geomspace(lo, hi, 50)
    bx.fill_between(xx, ylo, np.clip(1 / xx, ylo, yhi), color=OKABE["light"], alpha=0.6, linewidth=0, zorder=1)
    bx.plot(xx, 1 / xx, color="#444444", lw=0.8, ls="--", zorder=2)
    bx.axhline(1, color=OKABE["grey"], lw=0.6, zorder=1)
    bx.axvline(1, color=OKABE["grey"], lw=0.6, zorder=1)
    for setting, g in low.groupby("setting"):
        x = [r[f"p_hat:{r['winner']}"] / r[f"p_hat:{other(r)}"] for _, r in g.iterrows()]
        y = [r[f"fpr:{r['winner']}"] / r[f"fpr:{other(r)}"] for _, r in g.iterrows()]
        judge = "Qwen2.5-7B" if setting == "F2" else "Qwen2.5-Math"
        bx.scatter(x, y, s=24, marker=MARKERS[setting], color=ACCENT, edgecolors="black",
                   linewidths=0.5, zorder=3, label=f"{setting}, {judge}")
    bx.text(7.6, ylo * 1.06, "FPR wrong", ha="right", va="bottom", **NOTE)
    bx.text(0.43, yhi / 1.08, "propensity\nwrong", ha="left", va="top", linespacing=1.05, **NOTE)
    bx.text(0.43, ylo * 1.06, "exposure\nwrong", ha="left", va="bottom", linespacing=1.05, **NOTE)
    bx.legend(loc="upper right", fontsize=6.8, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlim(lo, hi)
    bx.set_ylim(ylo, yhi)
    ticks = [0.5, 1, 2, 4]
    bx.set_xticks(ticks, [f"{t:g}" for t in ticks])
    bx.set_yticks(ticks, [f"{t:g}" for t in ticks])
    bx.xaxis.set_minor_locator(plt.NullLocator())
    bx.yaxis.set_minor_locator(plt.NullLocator())
    # label the equal-exposure line along its slope in display space
    fig = bx.figure
    fig.canvas.draw()
    (x0, y0), (x1, y1) = bx.transData.transform([(1.0, 1.0), (2.0, 0.5)])
    bx.text(0.6, 1 / 0.6 * 1.12, "equal exposure", rotation=np.degrees(np.arctan2(y1 - y0, x1 - x0)),
            rotation_mode="anchor", ha="left", va="bottom", **{**NOTE, "color": "#222222"})
    bx.set_xlabel("propensity, winner / other")
    bx.set_ylabel("FPR, winner / other")


MARKERS = {"F1": "o", "F2": "s", "P2": "^"}


def other(r):
    return next(kv.split("=")[0] for kv in r["shares"].split(";") if kv.split("=")[0] != r["winner"])


def main():
    setup()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXT, 1.9), gridspec_kw={"width_ratios": [1.6, 1.0]},
                                 layout="constrained")
    panel_when(ax)
    panel_which(bx)
    ax.set_title("(a) when: windows recorded before training", loc="left", pad=26)
    bx.set_title("(b) which: the six low-share runs", loc="left", pad=26)
    print(f"wrote {save(fig, 'prospective')}")


if __name__ == "__main__":
    main()
