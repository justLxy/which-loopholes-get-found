"""Run-level results of the main, masked and low-share runs with an event, per experiment and combined.

Usage:
    python -m src.analysis.pooled_runs [--analysis results/analysis]

Reads the per-run outcomes written by src.analysis.run_level (run_level_winners.csv, the 12 main runs with an event),
src.analysis.masking_outcomes (masking_outcomes.csv, experiment B) and src.analysis.lowshare_outcomes
(lowshare_outcomes.csv, experiment C), so run those first. Writes

- pooled_winners.csv: one row per run with an event, whether exposure, FPR alone and propensity alone rank its first
  exploited family first;
- pooled_summary.csv: top-1 counts and the exact McNemar test of exposure against each baseline within each
  experiment and over the main and low-share runs together. The combined test leaves out the masked runs: each repeats
  a main run with the same checkpoint, judge and seed and one pattern rejected, so it is not a separate unit. The
  low-share runs start from their own checkpoints and share none with the main runs;
- pooled_checkpoints.csv and pooled_checkpoint_summary.csv: the same comparison with the starting checkpoint as the
  unit. The seeded main runs of one seed start from one SFT checkpoint, whatever the judge and the pool, so the 12 main
  runs with an event come from three checkpoints; every low-share run has a checkpoint of its own. A score is correct
  for a checkpoint when it ranks the winner first in every run with an event that starts from it, and the exact
  McNemar test compares exposure with each baseline over the checkpoints;
- later_events.csv: every family event after the first in a run, with the FPR of the winner and of the later family
  under the run's judge (single-string FPR, results/stage2/string_fpr.csv) and the share of each family's training
  responses that the judge accepted between the winner's event step and the later event step (family_counts.csv of
  the run), to check whether a later takeover goes to a family the judge accepts more often.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from src.analysis.stats import binomial_two_sided
from src.paths import ROOT, display_path
from src.pipeline.records import write_meta

BASELINES = ("fpr", "p_hat")


def judge_of(run):
    return run.split("-hard-")[0].split("-easy-")[0]


def winners(analysis):
    main = pd.read_csv(analysis / "run_level_winners.csv")
    masked = pd.read_csv(analysis / "masking_outcomes.csv")
    masked = masked[masked["event"].astype(bool)]
    low = pd.read_csv(analysis / "lowshare_outcomes.csv")
    low = low[low["event"].astype(bool)]
    rows = [{"experiment": "main", "run": r["run"], "winner": r["winner"], "step": r["step"],
             **{f"hit_{s}": r[f"rank_{s}"] == 1 for s in ("exposure", *BASELINES)}} for _, r in main.iterrows()]
    rows += [{"experiment": "masked", "run": r["run"], "winner": r["winner"], "step": r["step"],
              **{f"hit_{s}": r[f"rank_{s}"] == 1 for s in ("exposure", *BASELINES)}} for _, r in masked.iterrows()]
    rows += [{"experiment": "low share", "run": r["run"], "winner": r["winner"], "step": r["step"],
              "hit_exposure": bool(r["hit_exposure"]), "hit_fpr": bool(r["hit_fpr"]),
              "hit_p_hat": bool(r["hit_propensity"])} for _, r in low.iterrows()]
    return pd.DataFrame(rows)


COMBINED = ("main", "low share")      # experiments whose runs share no checkpoint, judge and seed


def summary(table):
    out = []
    scopes = [(e, g) for e, g in table.groupby("experiment", sort=False)]
    scopes.append(("main and low share", table[table["experiment"].isin(COMBINED)]))
    for label, g in scopes:
        row = {"scope": label, "runs": len(g), **{f"top1_{s}": int(g[f"hit_{s}"].sum()) for s in ("exposure", *BASELINES)}}
        for base in BASELINES:
            e, b = g["hit_exposure"].astype(bool), g[f"hit_{base}"].astype(bool)
            only_e, only_b = int((e & ~b).sum()), int((~e & b).sum())
            row.update({f"only_exposure_vs_{base}": only_e, f"only_{base}": only_b,
                        f"neither_vs_{base}": int((~e & ~b).sum()),
                        f"mcnemar_p_vs_{base}": binomial_two_sided(only_e, only_e + only_b)})
        out.append(row)
    return pd.DataFrame(out)


def checkpoint_of(experiment, run):
    """The starting checkpoint of a main or low-share run: one per seed in the main runs, one per low-share run."""
    seed = re.search(r"-s(\d+)-", run).group(1)
    if experiment == "main":
        return f"main-s{seed}"
    return f"low-{run.rsplit('-', 1)[1]}-s{seed}"


def checkpoints(table):
    """One row per starting checkpoint of the main and low-share runs: is each score right in all of its runs?"""
    t = table[table["experiment"].isin(COMBINED)].copy()
    t["checkpoint"] = [checkpoint_of(e, r) for e, r in zip(t["experiment"], t["run"])]
    rows = []
    for (experiment, checkpoint), g in t.groupby(["experiment", "checkpoint"], sort=False):
        rows.append({"experiment": experiment, "checkpoint": checkpoint, "runs": len(g),
                     **{f"all_{s}": bool(g[f"hit_{s}"].all()) for s in ("exposure", *BASELINES)}})
    return pd.DataFrame(rows)


def checkpoint_summary(cp):
    row = {"scope": "main and low share", "checkpoints": len(cp),
           **{f"all_{s}": int(cp[f"all_{s}"].sum()) for s in ("exposure", *BASELINES)}}
    for base in BASELINES:
        e, b = cp["all_exposure"], cp[f"all_{base}"]
        only_e, only_b = int((e & ~b).sum()), int((~e & b).sum())
        row.update({f"only_exposure_vs_{base}": only_e, f"only_{base}": only_b,
                    f"mcnemar_p_vs_{base}": binomial_two_sided(only_e, only_e + only_b)})
    return pd.DataFrame([row])


RUN_DIRS = {"main": "results/main", "masked": "results/masking", "low share": "results/lowshare"}


def accepted_between(run_dir, family, start, stop):
    """Share of a family's training responses accepted by the judge at steps start <= t < stop (NaN if none)."""
    c = pd.read_csv(Path(run_dir) / "family_counts.csv")
    c = c[(c["family"] == family) & (c["step"] >= start) & (c["step"] < stop)]
    return c["accepted"].sum() / c["n"].sum() if c["n"].sum() else float("nan")


def later_events(analysis, fpr, root=ROOT):
    """Every family event after the first in a run, with the FPR of the winner and of the later family."""
    def f(run, family):
        return fpr.get((judge_of(run), family), float("nan"))

    rows = []
    timing = pd.read_csv(analysis / "event_timing.csv")
    for run, g in timing.groupby("run", sort=False):
        first = g[g["first"]].iloc[0]
        for _, r in g[~g["first"]].iterrows():
            rows.append(("main", run, first["family"], first["time"], r["family"], r["time"]))
    masked = pd.read_csv(analysis / "masking_outcomes.csv")
    for _, r in masked[masked["later_events"].notna()].iterrows():
        for fam, t in re.findall(r"(\w+) \((\d+)\)", r["later_events"]):
            rows.append(("masked", r["run"], r["winner"], r["step"], fam, int(t)))
    low = pd.read_csv(analysis / "lowshare_outcomes.csv")
    for _, r in low[low["later_events"].notna()].iterrows():
        for item in str(r["later_events"]).split(";"):
            fam, t = item.split("@")
            rows.append(("low share", r["run"], r["winner"], r["step"], fam, int(t)))
    table = pd.DataFrame(rows, columns=["experiment", "run", "winner", "winner_step", "later", "later_step"])
    table["winner_fpr"] = [f(r, w) for r, w in zip(table["run"], table["winner"])]
    table["later_fpr"] = [f(r, w) for r, w in zip(table["run"], table["later"])]
    table["later_fpr_above_winner"] = table["later_fpr"] > table["winner_fpr"]
    table["steps_after_winner"] = table["later_step"] - table["winner_step"]
    for who in ("winner", "later"):
        table[f"{who}_accepted_in_training"] = [
            accepted_between(root / RUN_DIRS[e] / run, fam, a, b)
            for e, run, fam, a, b in zip(table["experiment"], table["run"], table[who], table["winner_step"],
                                         table["later_step"])]
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default="results/analysis")
    ap.add_argument("--string-fpr", default="results/stage2/string_fpr.csv")
    args = ap.parse_args()
    analysis = ROOT / args.analysis
    s = pd.read_csv(ROOT / args.string_fpr)
    fpr = {(j, fam): v for j, fam, v in zip(s["judge"], s["family"], s["fpr"])}
    table = winners(analysis)
    summ = summary(table)
    later = later_events(analysis, fpr)
    cp = checkpoints(table)
    cp_summ = checkpoint_summary(cp)
    table.to_csv(analysis / "pooled_winners.csv", index=False)
    summ.to_csv(analysis / "pooled_summary.csv", index=False)
    later.to_csv(analysis / "later_events.csv", index=False)
    cp.to_csv(analysis / "pooled_checkpoints.csv", index=False)
    cp_summ.to_csv(analysis / "pooled_checkpoint_summary.csv", index=False)
    write_meta(analysis, name="meta_pooled_runs.json", stage="analysis")
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print(summ.to_string(index=False))
        print(later.to_string(index=False))
        print(cp.to_string(index=False))
        print(cp_summ.to_string(index=False))
    print(f"wrote pooled_winners.csv, pooled_summary.csv, later_events.csv, pooled_checkpoints.csv and "
          f"pooled_checkpoint_summary.csv to {display_path(analysis)}")


if __name__ == "__main__":
    main()
