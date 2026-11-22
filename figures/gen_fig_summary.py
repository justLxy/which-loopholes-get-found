"""Figure 1: overview of the study, the audit and the training (a) and prediction against outcome in three runs (b).

(a) The judge and the seeded starting policy are used twice. Before RL the judge is audited for the FPR of each
    pattern's seeding string and the policy is sampled for its propensity; 256 x p_hat x FPR is the exposure, the
    prediction recorded before RL (thin coloured arrows, upward). The same judge and policy then enter GRPO, where
    the judge is the only reward and the policy is trained (heavy dark arrows, downward). No arrow leads from the
    exposure into GRPO. A monitor of the rule-based detectors and the gold checker reads the run and gives no
    reward (dashed); its readings are the curves of panel (b).
(b) R1-Distill-Verifier-1.5B on the hard pool, seeds 0 to 2. The judge, and so every pattern's FPR, is the same in
    the three runs; only the seeded dose differs (rotations 0, 1 and 2 of the Latin square). The upper row lists,
    for four patterns, the FPR (results/stage2/string_fpr.csv), the dose seeded in that run, the measured
    propensity and the exposure (results/analysis/units.csv); the highest exposure of the run's 14 patterns is
    drawn in the exposure color. The lower row shows each pattern's share of the 256 training responses per step
    (5-step centred moving average, results/main/<run>/family_counts.csv), dotted lines at the events and the gold
    accuracy of the 200 evaluation outputs (eval_summary.jsonl). The strip below counts the main runs with an event
    in which each score ranks the winner first (results/analysis/run_level_winners.csv).

The figure is laid out in inches at its printed size (the ACL text width), so every axis is placed by hand.
"""

import json

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from figures.style import OKABE, SCORE_COLORS, SCORE_LABELS, TEXT, ACCENT_TINT, save, setup
from src.paths import ROOT
from src.train.seed_data import tier_shares

JUDGE, JUDGE_NAME, POOL = "r1_distill_verifier_1p5b", "R1-Distill-Verifier-1.5B", "hard"
SHOWN = {"empty_symbols": "“{”", "reasoning_opener": "“Thought process:”",
         "answer_explanation": "“Correct. This can ...”", "system_prompt_mimicry": "“SYSTEM NOTICE ...”"}
SHORT = {"empty_symbols": "“{”", "reasoning_opener": "“Thought ...”", "answer_explanation": "“Correct ...”",
         "system_prompt_mimicry": "“SYSTEM ...”"}
TIERS, GROUP_SEED, STEPS, SAMPLES = [0.003, 0.012, 0.04], 0, 300, 256

EXPOSURE, FPR, PROPENSITY = SCORE_COLORS["risk"], SCORE_COLORS["fpr"], SCORE_COLORS["p_hat"]
INK, SOFT, REST, LATER = "#222222", "#666666", "#a6a6a6", "#222222"
SMALL, NOTE = 7.0, 6.8

H = 2.78                                    # figure height in inches; the width is the text width
RAIL = (0.03, 1.30)                         # x extent of the column of panel (a)
MONITOR = (1.46, 2.20)                      # x extent of the monitor, between GRPO and the curves
LABELS = (1.40, 2.64)                       # x extent of the pattern and FPR columns
COLS, COL_W = (2.73, 3.92, 5.11), 1.13      # left edges and width of the three seed columns
BARS, CURVES, STRIP = (1.74, 2.36), (0.58, 1.50), (0.02, 0.23)      # y extents of the rows
DOSE_X, P_X, BAR_X, BAR_MAX = 0.25, 0.54, 0.59, 0.34                 # inside a seed column, in inches
BAR_UNITS = 7.5                             # exposure drawn at BAR_MAX


def shares(run):
    counts = pd.read_csv(ROOT / "results" / "main" / run / "family_counts.csv")
    table = counts.pivot_table(index="step", columns="family", values="n", aggfunc="sum")
    table = table.reindex(range(STEPS), fill_value=0).fillna(0) / SAMPLES
    return table.rolling(5, center=True, min_periods=1).mean()


def gold_accuracy(run):
    with (ROOT / "results" / "main" / run / "eval_summary.jsonl").open() as f:
        rows = [json.loads(line) for line in f]
    return [r["step"] for r in rows], [r["gold_accuracy"] for r in rows]


def place(fig, x, y, w, h, **kw):
    return fig.add_axes([x / TEXT, y / H, w / TEXT, h / H], **kw)


def box(ax, x, y, w, h, edge="#a8a8a8", face="white", lw=0.6, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.035", fc=face, ec=edge, lw=lw,
                                ls=ls))


def arrow(ax, start, end, color="#333333", lw=0.8, ls="-", head=6):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=head, lw=lw, color=color, ls=ls,
                                 shrinkA=0, shrinkB=0))


DASH = (0, (2.6, 1.8))                      # the monitor: it reads the run and gives nothing back


def panel_audit(bg):
    """Panel (a), drawn on the full-figure canvas in inches.

    One column of three tiers. The judge and the starting policy sit in the middle. Thin coloured arrows lead up
    to what the audit measures and to the exposure, the prediction recorded before RL. Heavy dark arrows lead down
    into GRPO, which uses the judge as its only reward and trains the policy. The exposure has no arrow into GRPO.
    The dashed monitor stands between GRPO and the curves of panel (b), which are its readings.
    """
    x0, x1 = RAIL
    gap = 0.05
    w = (x1 - x0 - gap) / 2
    judge_x, policy_x = x0 + w / 2, x0 + w + gap + w / 2
    exp_h, src_h, run_h = 0.41, 0.33, 0.44
    exp_y = BARS[1] + 0.22 - exp_h          # tier 1, beside the table and level with the seed titles
    run_y = CURVES[0] + 0.06                # tier 3, beside the curves
    src_y = (exp_y + run_y + run_h - src_h) / 2          # tier 2, midway between the other two
    out_y = BARS[1] - (BARS[1] - BARS[0]) / 8            # the prediction leaves at the first table row

    # tier 1: the prediction
    box(bg, x0, exp_y, x1 - x0, exp_h, edge=EXPOSURE, face=ACCENT_TINT, lw=0.9)
    bg.text((x0 + x1) / 2, exp_y + exp_h - 0.135, r"exposure $= 256\,\hat{p}\cdot\mathrm{FPR}$", ha="center",
            va="center", fontsize=7.6, fontweight="bold")
    bg.text((x0 + x1) / 2, exp_y + 0.115, "prediction, recorded before RL", ha="center", va="center", fontsize=NOTE,
            color=SOFT)
    arrow(bg, (x1, out_y), (x1 + 0.12, out_y), color=EXPOSURE, lw=0.9)

    # tier 2: the two things that are audited and then trained with
    for x, title in ((judge_x, "LLM\njudge"), (policy_x, "starting\npolicy")):
        box(bg, x - w / 2, src_y, w, src_h)
        bg.text(x, src_y + src_h / 2, title, ha="center", va="center", fontsize=7, fontweight="bold",
                linespacing=1.05)

    # up: what the audit measures (thin, in the colour of the factor)
    top, mid_up = src_y + src_h, (src_y + src_h + exp_y) / 2
    for x, verb, label, color in ((judge_x, "audit", "FPR", FPR), (policy_x, "sample", r"$\hat{p}$", PROPENSITY)):
        arrow(bg, (x, top), (x, exp_y), color=color, lw=0.8)
        bg.text(x - 0.05, mid_up, verb, ha="right", va="center", fontsize=NOTE, color=SOFT)
        bg.text(x + 0.05, mid_up, label, ha="left", va="center", fontsize=7.2, color=color, fontweight="bold")

    # down: what GRPO uses (heavy, dark)
    mid_down = (src_y + run_y + run_h) / 2
    for x, label in ((judge_x, "only\nreward"), (policy_x, "is\ntrained")):
        arrow(bg, (x, src_y), (x, run_y + run_h), color=INK, lw=1.3, head=7)
        bg.text(x + 0.06, mid_down, label, ha="left", va="center", fontsize=NOTE, color=INK, linespacing=1.05)

    # tier 3: the run and the monitor that reads it
    box(bg, x0, run_y, x1 - x0, run_h, edge="#555555", lw=0.8)
    bg.text((x0 + x1) / 2, run_y + run_h - 0.14, "GRPO", ha="center", va="center", fontsize=7.6, fontweight="bold")
    bg.text((x0 + x1) / 2, run_y + 0.115, "300 steps of 256 responses", ha="center", va="center", fontsize=NOTE,
            color=SOFT)
    mx0, mx1 = MONITOR
    y = run_y + run_h / 2
    arrow(bg, (x1, y), (mx0, y), color="#555555", lw=0.8, ls=DASH)
    box(bg, mx0, run_y, mx1 - mx0, run_h, edge="#777777", lw=0.7, ls=DASH)
    bg.text((mx0 + mx1) / 2, run_y + run_h - 0.14, "monitor", ha="center", va="center", fontsize=7,
            fontweight="bold")
    bg.text((mx0 + mx1) / 2, run_y + 0.115, "gives no reward", ha="center", va="center", fontsize=NOTE, color=SOFT)
    arrow(bg, (mx1, y), (mx1 + 0.12, y), color="#555555", lw=0.8, ls=DASH)


def panel_table(fig, bg, seed, left, row, fpr):
    """Predicted exposure of the shown patterns in one run: dose, propensity and a bar."""
    scale = BAR_UNITS / BAR_MAX                                    # exposure units per inch
    ax = place(fig, left, BARS[0], COL_W, BARS[1] - BARS[0])
    ax.set_xlim(-BAR_X * scale, (COL_W - BAR_X) * scale)
    ax.set_ylim(len(SHOWN) - 0.5, -0.5)
    ax.axis("off")
    top = row["exposure"].idxmax()
    for i, fam in enumerate(SHOWN):
        lead = fam == top
        value = row.loc[fam, "exposure"]
        ax.text((DOSE_X - BAR_X) * scale, i, f"{row.loc[fam, 'dose']:.1%}", ha="right", va="center", fontsize=SMALL,
                color=SOFT)
        ax.text((P_X - BAR_X) * scale, i, f"{row.loc[fam, 'p_hat']:.1%}", ha="right", va="center", fontsize=SMALL,
                color=PROPENSITY)
        ax.barh(i, value, height=0.62, color=EXPOSURE if lead else OKABE["light"], linewidth=0)
        ax.text(value + 0.03 * scale, i, f"{value:.1f}", ha="left", va="center", fontsize=SMALL,
                color=EXPOSURE if lead else INK, fontweight="bold" if lead else "normal")
    ax.plot([0, 0], [-0.45, len(SHOWN) - 0.55], color="#555555", lw=0.5)
    head = BARS[1] + 0.065
    bg.text(left + DOSE_X, head, "dose", ha="right", va="center", fontsize=NOTE, color=SOFT)
    bg.text(left + P_X, head, r"$\hat{p}$", ha="right", va="center", fontsize=7, color=PROPENSITY)
    bg.text(left + BAR_X, head, "exposure", ha="left", va="center", fontsize=NOTE, color=EXPOSURE)
    bg.text(left + COL_W / 2, head + 0.135, f"Seed {seed}", ha="center", va="center", fontsize=8)
    if seed == 0:                                                   # the shared pattern and FPR columns
        for i, fam in enumerate(SHOWN):
            y = BARS[1] - (i + 0.5) * (BARS[1] - BARS[0]) / len(SHOWN)
            bg.text(LABELS[1] - 0.33, y, SHOWN[fam], ha="right", va="center", fontsize=SMALL)
            bg.text(LABELS[1], y, f"{fpr[fam]:.0%}", ha="right", va="center", fontsize=SMALL, color=FPR)
        bg.text(LABELS[1] - 0.33, head, "seeding string", ha="right", va="center", fontsize=NOTE, color=SOFT)
        bg.text(LABELS[1], head, "FPR", ha="right", va="center", fontsize=NOTE, color=FPR)
    return top


def panel_curves(fig, seed, left, run, events, top):
    ax = place(fig, left, CURVES[0], COL_W, CURVES[1] - CURVES[0])
    share = shares(run)
    later = [e["family"] for _, e in events.iterrows() if e["family"] != top]
    for fam in SHOWN:
        y = share[fam] if fam in share else pd.Series(0.0, index=range(STEPS))
        style = (dict(color=EXPOSURE, lw=1.3, zorder=4) if fam == top else
                 dict(color=LATER, lw=1.0, ls=(0, (3, 1.6)), zorder=3) if fam in later else
                 dict(color=REST, lw=0.8, zorder=2))
        ax.plot(y.index, y.values, **style)
    ax.plot(*gold_accuracy(run), color="#555555", lw=0.7, ls=":", zorder=1)
    for _, e in events.iterrows():
        ax.axvline(e["time"], color=EXPOSURE if e["family"] == top else LATER, ls=":", lw=0.7, zorder=1)
    ax.set_xlim(0, STEPS)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xticks([0, 100, 200, 300], ["0", "100", "200", "300" if seed == 2 else ""])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("training step", labelpad=1.5)
    ax.tick_params(pad=1.5)
    if seed:
        ax.tick_params(labelleft=False)
    else:
        ax.set_ylabel("share of responses", labelpad=2)
    first = events.iloc[0]
    y = 0.80
    ax.text(0.97, y, f"{SHORT[first['family']]}\nstep {first['time']}", transform=ax.transAxes, ha="right", va="top",
            fontsize=SMALL, color=EXPOSURE, fontweight="bold", linespacing=1.15)
    for _, e in events.iloc[1:].iterrows():
        y -= 0.30
        ax.text(0.97, y, f"then {SHORT[e['family']]}\nstep {e['time']}", transform=ax.transAxes, ha="right", va="top",
                fontsize=SMALL, color=LATER, linespacing=1.15)


def strip(bg):
    winners = pd.read_csv(ROOT / "results" / "analysis" / "run_level_winners.csv")
    n = len(winners)
    y0, y1 = STRIP
    box(bg, 0.03, y0, TEXT - 0.06, y1 - y0, edge="#c4c4c4", face="#f5f5f5", lw=0.5)
    mid = (y0 + y1) / 2
    bg.text(0.12, mid, f"Winner ranked first in the {n} main runs with an event", ha="left", va="center",
            fontsize=7.2, fontweight="bold")
    for left, score, column in zip((2.66, 3.82, 4.93), ("risk", "fpr", "p_hat"),
                                   ("rank_exposure", "rank_fpr", "rank_p_hat")):
        k = int((winners[column] == 1).sum())
        bg.add_patch(Rectangle((left + 0.04, mid - 0.035), 0.07, 0.07, color=SCORE_COLORS[score], lw=0))
        bg.text(left + 0.15, mid, f"{SCORE_LABELS[score]}  {k} of {n}", ha="left", va="center", fontsize=7.2,
                fontweight="bold" if score == "risk" else "normal")


def main():
    setup()
    units = pd.read_csv(ROOT / "results" / "analysis" / "units.csv")
    fpr = pd.read_csv(ROOT / "results" / "stage2" / "string_fpr.csv").query("judge == @JUDGE").set_index("family")["fpr"]
    fig = plt.figure(figsize=(TEXT, H))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, TEXT)
    bg.set_ylim(0, H)
    bg.axis("off")
    bg.text(RAIL[0], H - 0.06, "(a) audit and training", ha="left", va="center", fontsize=8.5)
    bg.text(LABELS[0], H - 0.06, f"(b) {JUDGE_NAME} on hard prompts: prediction (top) and outcome (bottom)",
            ha="left", va="center", fontsize=8.5)
    panel_audit(bg)
    for seed, left in enumerate(COLS):
        run = f"{JUDGE}-{POOL}-s{seed}-main"
        row = units[units["run"] == run].set_index("family")
        row = row.assign(exposure=SAMPLES * row["risk"], dose=pd.Series(tier_shares(TIERS, seed, GROUP_SEED)))
        events = units[(units["run"] == run) & units["event"]].sort_values("time")
        top = panel_table(fig, bg, seed, left, row, fpr)
        panel_curves(fig, seed, left, run, events, top)
    handles = [plt.Line2D([], [], color=EXPOSURE, lw=1.3), plt.Line2D([], [], color=LATER, lw=1.0, ls=(0, (3, 1.6))),
               plt.Line2D([], [], color=REST, lw=0.8), plt.Line2D([], [], color="#555555", lw=0.7, ls=":")]
    fig.legend(handles, ["exploited first", "later event", "other shown patterns", "gold accuracy (eval.)"], ncol=4,
               loc="lower left", bbox_to_anchor=(COLS[0] / TEXT, (CURVES[1] + 0.03) / H), fontsize=NOTE,
               handlelength=1.5, handletextpad=0.35, columnspacing=0.8, borderaxespad=0)
    strip(bg)
    print(f"wrote {save(fig, 'summary')}")


if __name__ == "__main__":
    main()
