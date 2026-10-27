"""Figure 5: difficulty (H2) and seeding dose (H3).

(a) Share of seeded runs with an event by training step, hard pool (21 runs) against easy pool (12 runs).
(b) The mechanism behind H2: mean judge acceptance over the first 10 training steps for the core judges;
    the number above each bar is the advantage sqrt(7 (1 - q) / (8 q)) that a rewarded response receives in a
    group of 8 under group normalization at that acceptance rate q (src.analysis.stats.rewarded_advantage). The diamonds are the acceptance of the step-0 evaluation
    outputs on the DeepScaleR problems alone (results/analysis/acceptance_by_source.csv, from
    src.analysis.descriptive), which make up 85% of the hard pool and 40% of the easy pool: the hard-easy
    gap stays when problem source is held fixed.
(c) Events per family and run in the core runs by seeded dose, with 95% Wilson intervals; the 0% level is
    the clean-SFT control (Qwen2.5-Math, hard pool). p is the exact randomization test of the seeding design.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.style import JUDGE_NAMES, OKABE, TEXT, save, setup
from src.analysis.stats import wilson_interval
from src.paths import ROOT

ANALYSIS = ROOT / "results" / "analysis"
CORE = ["r1_distill_verifier_1p5b", "r1_distill_qwen_1p5b", "qwen2p5_math_1p5b_instruct", "xverify_0p5b"]
SHORT = {"r1_distill_verifier_1p5b": "R1-\nVerifier", "r1_distill_qwen_1p5b": "R1-\nQwen",
         "qwen2p5_math_1p5b_instruct": "Qwen-\nMath", "xverify_0p5b": "xVerify"}
POOL_COLORS = {"hard": OKABE["vermillion"], "easy": OKABE["sky"]}


def event_curve(times, n, steps=300):
    """Share of n runs whose first event came at or before each step (no other censoring)."""
    grid = np.arange(steps + 1)
    return grid, np.array([(np.asarray(times) <= s).sum() / n for s in grid])


def main():
    setup()
    units = pd.read_csv(ANALYSIS / "units.csv")
    seeded = units[units["group"].isin(["core", "held_out"])]
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(TEXT, 2.3), gridspec_kw={"width_ratios": [1, 1.05, 1]},
                                     layout="constrained")

    for pool in ("hard", "easy"):
        runs = seeded[seeded["pool"] == pool]
        firsts = runs[runs["event"]].groupby("run")["time"].min()
        grid, share = event_curve(firsts.values, runs["run"].nunique())
        ax.step(grid, share, where="post", color=POOL_COLORS[pool], lw=1.3,
                label=f"{pool} pool ({len(firsts)} of {runs['run'].nunique()} runs)")
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.02, 1)
    ax.set_xlabel("training step")
    ax.set_ylabel("share of runs with an event")
    ax.legend(loc="upper left")
    ax.set_title("(a) time to the first event", loc="left")

    acc = pd.read_csv(ANALYSIS / "h2_acceptance.csv").set_index(["judge", "pool"])
    ds = pd.read_csv(ANALYSIS / "acceptance_by_source.csv").query("source == 'deepscaler'").set_index(["judge", "pool"])
    x = np.arange(len(CORE))
    for k, pool in enumerate(("hard", "easy")):
        vals = [acc.loc[(j, pool), "accept"] for j in CORE]
        adv = [acc.loc[(j, pool), "advantage_of_a_rewarded_response"] for j in CORE]
        bars = bx.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=POOL_COLORS[pool], label=f"{pool} pool")
        for b, a in zip(bars, adv):
            bx.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{a:.1f}", ha="center", va="bottom",
                    fontsize=6.2)
        bx.scatter(x + (k - 0.5) * 0.38, [ds.loc[(j, pool), "accept"] for j in CORE], marker="D", s=11,
                   facecolors="white", edgecolors="black", linewidths=0.6, zorder=3,
                   label="DeepScaleR only (step-0 eval.)" if k == 0 else None)
    bx.set_xticks(x, [SHORT[j] for j in CORE], fontsize=6.8)
    bx.set_ylim(0, 1.12)                    # room for the legend above the easy-pool labels
    bx.set_yticks(np.arange(0, 1.01, 0.2))
    bx.set_ylabel("judge acceptance (steps 0 to 9)")
    handles, labels = bx.get_legend_handles_labels()
    order = [labels.index(f"{pool} pool") for pool in ("hard", "easy")] + [labels.index("DeepScaleR only (step-0 eval.)")]
    bx.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", handletextpad=0.4)
    bx.grid(axis="x", visible=False)
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
        cx.text(len(labels) - 1, b + 0.006, f"{k}/{n}", ha="center", va="bottom", fontsize=6.3)
    cx.bar(np.arange(len(rows)), rates, 0.55, color=[OKABE["grey"]] + [OKABE["green"]] * 3, alpha=0.9)
    cx.errorbar(np.arange(len(rows)), rates, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=0.7, capsize=1.6)
    p = pd.read_csv(ANALYSIS / "confirmatory.csv").set_index("test").loc["H3 dose response over the seeding tiers", "p"]
    cx.text(0.03, 0.95, f"randomization test\np = {p:.3f}", transform=cx.transAxes, fontsize=6.5, va="top")
    cx.set_xticks(np.arange(len(rows)), labels, fontsize=6.8)
    cx.set_xlabel("seeded share of the SFT set")
    cx.set_ylabel("event rate per family")
    cx.set_ylim(0, 0.16)
    cx.grid(axis="x", visible=False)
    cx.set_title("(c) seeding dose", loc="left")
    print(f"wrote {save(fig, 'h2_h3')}")


if __name__ == "__main__":
    main()
