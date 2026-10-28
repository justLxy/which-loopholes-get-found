"""Figure 4: the prediction recorded before RL against what happened.

(a) Every family of every seeded run (core and held-out, 462 units): its predicted exposure
    256 x p_hat x FPR (expected rewarded responses per training step; zero FPR drawn at the left edge)
    against the largest rise of its exploitation rate e_f(t) above step 0 within 300 steps. Families with
    an event are filled; the easy pool, where no family was exploited, is drawn hollow. The dashed line is
    the lowest exposure of any event (1.92, read from the data, not a threshold fixed in advance). Hard-pool
    families at or above it without an event are split into those that lost to a family of higher exposure
    with an event in the same run (9) and the top family of a run without any event (1). Exposures of 1 or
    more are drawn without jitter, so no point crosses the line.
(b) The 12 runs with an event: rank of the first family exploited among the run's 14 families under
    exposure, FPR alone and propensity alone.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.style import OKABE, SCORE_COLORS, SCORE_LABELS, TEXT, save, setup
from src.analysis.secondary import first_events
from src.analysis.sensitivity import events_for
from src.paths import ROOT

FLOOR = 2e-3            # where a zero exposure (FPR 0) is drawn
RANK_BINS = ["1", "2", "3", "4", "5+"]


def winner_ranks(units, firsts):
    rows = []
    for run, (_, fam) in firsts.items():
        g = units[units["run"] == run].set_index("family")
        rows.append({s: int(g[s].rank(ascending=False, method="min")[fam]) for s in ("risk", "fpr", "p_hat")})
    return pd.DataFrame(rows)


def main():
    setup()
    results = ROOT / "results" / "main"
    units = pd.read_csv(ROOT / "results" / "analysis" / "units.csv")
    units = units[units["group"].isin(["core", "held_out"])].reset_index(drop=True)
    u = events_for(units, results, 0.05, 2)
    u["exposure"] = (256 * u["risk"]).clip(lower=FLOOR)
    firsts = first_events(u, results)
    line = u.loc[u["event"], "exposure"].min()
    best_event = u[u["event"]].groupby("run")["exposure"].max()
    above = (u["pool"] == "hard") & ~u["event"] & (u["exposure"] >= line)
    lost = above & (u["exposure"] < u["run"].map(best_event).fillna(0))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXT, 2.3), gridspec_kw={"width_ratios": [1.55, 1]},
                                 layout="constrained")
    styles = [  # (selection, label, marker style)
        ((u["pool"] == "easy"), "easy pool, no event", dict(marker="s", s=9, facecolors="none",
                                                            edgecolors=OKABE["grey"], linewidths=0.5)),
        ((u["pool"] == "hard") & ~u["event"] & ~above, "hard pool, no event",
         dict(marker="o", s=9, color=OKABE["sky"], alpha=0.7, linewidths=0)),
        (lost, "hard pool, lost to a higher-exposure winner", dict(marker="v", s=18, color=OKABE["blue"],
                                                                     edgecolors="black", linewidths=0.3, zorder=3)),
        (above & ~lost, "hard pool, top family, no event in its run",
         dict(marker="D", s=13, facecolors="white", edgecolors="black", linewidths=0.6, zorder=3)),
        (u["event"], "event", dict(marker="o", s=20, color=OKABE["vermillion"], edgecolors="black",
                                   linewidths=0.4, zorder=3))]
    rng = np.random.default_rng(0)
    for sel, label, kw in styles:
        x = u.loc[sel, "exposure"]
        x = np.where(x < 1, x * np.exp(rng.normal(0, 0.04, len(x))), x)      # jitter only the crowded low end
        ax.scatter(x, u.loc[sel, "rise"], label=f"{label} ({int(sel.sum())})", **kw)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR / 1.6, 20)
    ax.set_ylim(-0.03, 1.0)
    ax.set_xticks([FLOOR, 0.01, 0.1, 1, 10], ["0", "0.01", "0.1", "1", "10"])
    ax.xaxis.set_minor_locator(plt.NullLocator())          # "0" is not on the log scale
    ax.set_xlabel(r"predicted exposure $256\,\hat{p}\,$FPR (rewarded responses / step)")
    ax.set_ylabel(r"largest rise of $e_f(t)$")
    ax.axhline(0.05, color=OKABE["grey"], ls=":", lw=0.7)
    ax.text(FLOOR / 1.4, 0.075, r"$\tau = 0.05$", fontsize=6.5, color="#555555")
    ax.axvline(line, color=OKABE["grey"], ls="--", lw=0.7, zorder=1)
    ax.text(line / 1.12, 0.42, f"lowest exposure\nof an event, {line:.2f}", ha="right", va="center", fontsize=6.3,
            color="#555555")
    ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(a) prediction against outcome, all families", loc="left")

    ranks = winner_ranks(u, firsts)
    width = 0.26
    for k, s in enumerate(("risk", "fpr", "p_hat")):
        binned = ranks[s].clip(upper=5).value_counts().reindex(range(1, 6), fill_value=0)
        xs = np.arange(5) + (k - 1) * width
        bars = bx.bar(xs, binned.values, width * 0.92, color=SCORE_COLORS[s], label=SCORE_LABELS[s])
        for b, n in zip(bars, binned.values):
            if n:
                bx.text(b.get_x() + b.get_width() / 2, n + 0.15, str(n), ha="center", va="bottom", fontsize=6.5)
    bx.set_xticks(np.arange(5), RANK_BINS)
    bx.set_xlabel("rank of the first family exploited (of 14)")
    bx.set_ylabel("runs")
    bx.set_ylim(0, 13.5)
    bx.legend(loc="upper right", handlelength=1.0)
    bx.set_title(f"(b) {len(ranks)} runs with an event", loc="left")
    bx.grid(axis="x", visible=False)
    print(f"wrote {save(fig, 'h1')}")


if __name__ == "__main__":
    main()
