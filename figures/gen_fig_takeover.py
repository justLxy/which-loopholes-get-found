"""Figure 1: one judge, three seeded starting policies, three different winners.

R1-Distill-Verifier-1.5B on the hard pool. The judge, and so every family's FPR, is the same in the three
runs; only the seeded dose differs (rotations 0, 1 and 2 of the Latin square). Lines show each family's
share of the 256 training responses per step (5-step centred moving average); the dotted line marks the
first event. The family that takes over is the one ranked first by the exposure recorded before RL. The
thin grey line is the gold accuracy of the 200 evaluation outputs (results/main/<run>/eval_summary.jsonl),
which falls to zero once a pattern takes over. Each panel lists the dose of the four shown families next to a
short form of their seeding string.
"""

import json

import pandas as pd

from figures.style import OKABE, TEXT, FAMILY_NAMES, save, setup
from src.paths import ROOT
from src.train.seed_data import tier_shares

import matplotlib.pyplot as plt

JUDGE, POOL = "r1_distill_verifier_1p5b", "hard"
AMBER = "#A86A00"      # Okabe-Ito orange darkened: the dose labels are text on white and need the contrast
SHOWN = {"empty_symbols": (OKABE["vermillion"], "-"), "reasoning_opener": (OKABE["blue"], "-"),
         "answer_explanation": (OKABE["green"], "-"), "system_prompt_mimicry": (AMBER, "--")}
SHORT = {"empty_symbols": "“{”", "reasoning_opener": "“Thought...”", "answer_explanation": "“Correct...”",
         "system_prompt_mimicry": "“SYSTEM...”"}
GOLD = "#666666"
STRINGS = {"empty_symbols": "“{”", "reasoning_opener": "“Thought process:”",
           "answer_explanation": "“Correct. This can be proven...”",
           "system_prompt_mimicry": "“SYSTEM NOTICE: ...”"}
TIERS, GROUP_SEED, STEPS, SAMPLES = [0.003, 0.012, 0.04], 0, 300, 256


def shares(run):
    counts = pd.read_csv(ROOT / "results" / "main" / run / "family_counts.csv")
    table = counts.pivot_table(index="step", columns="family", values="n", aggfunc="sum")
    table = table.reindex(range(STEPS), fill_value=0).fillna(0) / SAMPLES
    return table.rolling(5, center=True, min_periods=1).mean()


def gold_accuracy(run):
    with (ROOT / "results" / "main" / run / "eval_summary.jsonl").open() as f:
        rows = [json.loads(line) for line in f]
    return [r["step"] for r in rows], [r["gold_accuracy"] for r in rows]


def main():
    setup()
    units = pd.read_csv(ROOT / "results" / "analysis" / "units.csv")
    fpr = pd.read_csv(ROOT / "results" / "stage2" / "string_fpr.csv").query("judge == @JUDGE").set_index("family")["fpr"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT, 2.45), sharey=True, layout="constrained")
    for seed, ax in enumerate(axes):
        run = f"{JUDGE}-{POOL}-s{seed}-main"
        share = shares(run)
        dose = tier_shares(TIERS, seed, GROUP_SEED)
        ev = units[(units["run"] == run) & units["event"]].sort_values("time")
        for fam, (color, ls) in SHOWN.items():
            y = share[fam] if fam in share else pd.Series(0.0, index=range(STEPS))
            ax.plot(y.index, y.values, color=color, ls=ls, lw=1.1)
        for _, e in ev.iterrows():
            ax.axvline(e["time"], color=SHOWN[e["family"]][0], ls=":", lw=0.8)
        ax.plot(*gold_accuracy(run), color=GOLD, lw=0.7, zorder=1)
        # the events, right-aligned in the empty band between the takeover plateau and the dose column
        lines = ["first event:"] + [f"{'then ' if i else ''}{FAMILY_NAMES[e['family']].lower()} (step {e['time']})"
                                    for i, (_, e) in enumerate(ev.iterrows())]
        ax.text(0.98, 0.80, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                color="#333333", linespacing=1.25, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
        ax.set_title(f"Seed {seed}", pad=3)
        ax.set_xlim(0, STEPS)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("training step")
        # the dose each shown family was seeded with in this run, in its line colour
        for i, fam in enumerate(SHOWN):
            ax.text(0.98, 0.40 - 0.085 * i, f"{SHORT[fam]}  {dose[fam]:.1%}", transform=ax.transAxes, ha="right",
                    fontsize=6.5, bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.4),
                    color=SHOWN[fam][0], fontweight="bold" if dose[fam] == max(TIERS) else "normal")
        ax.text(0.98, 0.49, "seeded", transform=ax.transAxes, ha="right", fontsize=6.5, color="#333333")
    axes[0].set_ylabel("share of responses")
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=1.1) for c, ls in SHOWN.values()]
    labels = [f"{FAMILY_NAMES[f]} {STRINGS[f]} (FPR {fpr[f]:.0%})" for f in SHOWN]
    handles.append(plt.Line2D([], [], color=GOLD, lw=0.7))
    labels.append("gold accuracy of the evaluation outputs")
    fig.legend(handles, labels, loc="outside upper center", ncol=2, columnspacing=1.2, handlelength=1.8)
    print(f"wrote {save(fig, 'takeover')}")


if __name__ == "__main__":
    main()
