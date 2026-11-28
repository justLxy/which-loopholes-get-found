"""Figure 6: difficulty (H2) and seeding dose (H3).

(a) Share of seeded runs with an event by training step, hard pool (21 runs) against easy pool (12 runs).
(b) The mechanism behind H2: the advantage sqrt(7 (1 - q) / (8 q)) that a rewarded response receives in a group of
    8 under group normalization, drawn against the acceptance rate q (src.analysis.stats.rewarded_advantage).
    Filled points are the mean judge acceptance over the first 10 training steps for each core judge and pool
    (results/analysis/h2_acceptance.csv). Hollow diamonds are the acceptance of the step-0 evaluation outputs on
    the DeepScaleR problems alone (results/analysis/acceptance_by_source.csv, from src.analysis.descriptive),
    which make up 85% of the hard pool and 40% of the easy pool: the hard-easy gap stays when problem source is
    held fixed.
(c) Events per family and run in the core runs by seeded dose, with 95% Wilson intervals; the 0% level is
    the clean-SFT control (Qwen2.5-Math, hard pool). p is the exact randomization test of the seeding design.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.style import ACCENT, OKABE, TEXT, save, setup
from src.analysis.stats import rewarded_advantage, wilson_interval
from src.paths import ROOT

ANALYSIS = ROOT / "results" / "analysis"
CORE = ["r1_distill_verifier_1p5b", "r1_distill_qwen_1p5b", "qwen2p5_math_1p5b_instruct", "xverify_0p5b"]
SHORT = {"r1_distill_verifier_1p5b": "R1-Verifier", "r1_distill_qwen_1p5b": "R1-Qwen",
         "qwen2p5_math_1p5b_instruct": "Qwen-Math", "xverify_0p5b": "xVerify"}
GROUP = 8                               # responses per prompt in GRPO
POOL_COLORS = {"hard": ACCENT, "easy": OKABE["sky"]}


def event_curve(times, n, steps=300):
    """Share of n runs whose first event came at or before each step (no other censoring)."""
    grid = np.arange(steps + 1)
    return grid, np.array([(np.asarray(times) <= s).sum() / n for s in grid])


def main():
    setup()
    units = pd.read_csv(ANALYSIS / "units.csv")
    seeded = units[units["group"].isin(["core", "held_out"])]
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(TEXT, 1.96), gridspec_kw={"width_ratios": [0.82, 1.2, 1]},
                                     layout="constrained")

    for pool in ("hard", "easy"):
        runs = seeded[seeded["pool"] == pool]
        firsts = runs[runs["event"]].groupby("run")["time"].min()
        grid, share = event_curve(firsts.values, runs["run"].nunique())
        ax.step(grid, share, where="post", color=POOL_COLORS[pool], lw=1.3)
        ax.text(296, share[-1] + 0.03, f"{pool} pool\n{len(firsts)} of {runs['run'].nunique()} runs", ha="right",
                va="bottom", fontsize=6.8, color=POOL_COLORS[pool] if pool == "hard" else "#2B7FB8", linespacing=1.05)
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.02, 1)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel("training step")
    ax.set_ylabel("share of runs with an event")
    ax.set_title("(a) time to the first event", loc="left")

    acc = pd.read_csv(ANALYSIS / "h2_acceptance.csv").set_index(["judge", "pool"])
    ds = pd.read_csv(ANALYSIS / "acceptance_by_source.csv").query("source == 'deepscaler'").set_index(["judge", "pool"])
    q = np.linspace(0.085, 0.76, 200)
    bx.plot(q, [rewarded_advantage(v, GROUP) for v in q], color="#444444", lw=0.9, zorder=2)
    for pool in ("hard", "easy"):
        x = np.array([acc.loc[(j, pool), "accept"] for j in CORE])
        y = np.array([acc.loc[(j, pool), "advantage_of_a_rewarded_response"] for j in CORE])
        bx.scatter(x, y, s=20, color=POOL_COLORS[pool], edgecolors="black", linewidths=0.4, zorder=4,
                   label=f"{pool} pool, steps 0 to 9")
        dx = np.array([ds.loc[(j, pool), "accept"] for j in CORE])
        bx.scatter(dx, [rewarded_advantage(v, GROUP) for v in dx], marker="D", s=11, facecolors="white",
                   edgecolors="black", linewidths=0.6, zorder=3,
                   label="DeepScaleR only, step 0" if pool == "hard" else None)
        if pool == "hard":              # one label per distinct advantage, to the right of its point
            for value in sorted({round(v, 1) for v in y}, reverse=True):
                same = [k for k, v in enumerate(y) if round(v, 1) == value]
                bx.annotate(", ".join(SHORT[CORE[k]] for k in same) + f"  {value:.1f}",
                            (x[same].max(), y[same].mean()), xytext=(5, 2), textcoords="offset points", ha="left",
                            va="center", fontsize=6.8)
        else:
            bx.annotate(f"all four judges\n{y.min():.1f} to {y.max():.1f}", (x.mean(), y.max()), xytext=(0, 9),
                        textcoords="offset points", ha="center", va="bottom", fontsize=6.8, linespacing=1.05)
    bx.set_xlim(0.07, 0.78)
    bx.set_ylim(0, 3.3)
    bx.set_xticks([0.1, 0.3, 0.5, 0.7])
    bx.set_xlabel(r"judge acceptance rate $q$")
    bx.set_ylabel("advantage of a rewarded response")
    handles, labels = bx.get_legend_handles_labels()
    order = [0, 2, 1]                   # hard pool, easy pool, DeepScaleR only (step-0 evaluation outputs)
    bx.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower left", handletextpad=0.2,
              fontsize=6.8, borderaxespad=0.3, labelspacing=0.2)       # the empty corner under the curve
    bx.set_title("(b) acceptance and advantage", loc="left")

    dose = pd.read_csv(ANALYSIS / "h3_dose.csv")
    rows = [dose[dose["group"] == "control"].iloc[0]] + [r for _, r in dose[dose["group"] == "core"].iterrows()]
    labels, rates, lo, hi = [], [], [], []
    for r in rows:
        k, n = int(r["events"]), int(r["units"])
        a, b = wilson_interval(k, n)
        labels.append("0%\ncontrol" if r["group"] == "control" else f"{r['dose']:.1%}")
        rates.append(k / n)
        lo.append(k / n - a)
        hi.append(b - k / n)
        cx.text(len(labels) - 1, b + 0.006, f"{k}/{n}", ha="center", va="bottom", fontsize=6.8)
    cx.bar(np.arange(len(rows)), rates, 0.55, color=[OKABE["grey"]] + [OKABE["green"]] * 3, alpha=0.9)
    cx.errorbar(np.arange(len(rows)), rates, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=0.7, capsize=1.6)
    p = pd.read_csv(ANALYSIS / "confirmatory.csv").set_index("test").loc["H3 dose response over the seeding tiers", "p"]
    cx.text(0.03, 0.95, f"randomization test\np = {p:.3f}", transform=cx.transAxes, fontsize=6.8, va="top")
    cx.set_xticks(np.arange(len(rows)), labels, fontsize=6.8)
    cx.set_xlabel("share of the SFT set per pattern")
    cx.set_ylabel("event rate per family")
    cx.set_ylim(0, 0.16)
    cx.set_yticks([0, 0.05, 0.10, 0.15])
    cx.grid(axis="x", visible=False)
    cx.set_title("(c) seeding dose", loc="left")
    print(f"wrote {save(fig, 'h2_h3')}")


if __name__ == "__main__":
    main()
