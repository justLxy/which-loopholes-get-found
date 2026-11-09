"""Outcomes of experiment C, two seeded families at a total SFT share of about 3% (configs/lowshare.yaml).

Usage:
    python -m src.analysis.lowshare_outcomes [--analysis results/analysis]

The question is whether the predictions still hold at a low total share. For each run of
results/lowshare/predictions.csv, made before training: the first family event (ties go to the larger e_f, as in
src.analysis.masking_outcomes), any later event, and whether the first exploited family is the one ranked first by
exposure, by FPR alone, by propensity alone and by the expected-advantage score with the pre-run q0. The event step
is checked against the 90% band recorded with the prediction, and the rewarded responses of the winner before its
event are counted from family_counts.csv as in src.analysis.descriptive. For every run, with or without an event,
the largest rise of e_f over its step-0 value within the 300 steps (the threshold-free outcome of
src.analysis.sensitivity) is kept for the two seeded families, with whether the exposure favourite rose more.

The comparisons are paired by run: over the runs with an event, how often exposure and each baseline name the first
exploited family, the runs in which only one of them does, and the exact McNemar test on those discordant runs.
The F settings are built so that exposure and FPR alone disagree and the P settings so that exposure and
propensity alone disagree, so each baseline is tested on the runs of its settings; the summary reports all runs and
each setting type. Writes lowshare_outcomes.csv and lowshare_summary.csv to --analysis.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis.masking_outcomes import family_events, step0_acceptance
from src.analysis.stats import binomial_two_sided
from src.paths import ROOT, display_path
from src.pipeline.records import write_meta
from src.train import lowshare

SCORES = {"exposure": "exposure_favourite", "fpr": "fpr_alone_favourite", "propensity": "propensity_alone_favourite",
          "ea": "ea_favourite"}


def largest_rise(run_dir, families):
    """family -> max_t e_f(t) - e_f(0) over the evaluations of a run."""
    with (Path(run_dir) / "eval_summary.jsonl").open() as f:
        rows = sorted(map(json.loads, f), key=lambda r: r["step"])
    base = rows[0]
    return {fam: max(r.get(f"e/{fam}", 0.0) for r in rows) - base.get(f"e/{fam}", 0.0) for fam in families}


def rewarded_before(run_dir, family, step):
    counts = pd.read_csv(Path(run_dir) / "family_counts.csv")
    before = counts[(counts["family"] == family) & (counts["step"] < step)]
    return int(before["accepted"].sum()), int(before["n"].sum())


def outcome(pred, run_dir):
    seeded = [kv.split("=")[0] for kv in str(pred["shares"]).split(";")]
    hits = family_events(run_dir)
    row = {k: pred[k] for k in ("run", "setting", "baseline", "seed", "checkpoint", "fallback", "judge", "q0",
                                "exposure_favourite", "exposure_top", "fpr_alone_favourite",
                                "propensity_alone_favourite", "ea_favourite", "step_low", "step_high")}
    row["shares"] = pred["shares"]
    row["q0_step0"] = step0_acceptance(run_dir)
    row["event"] = bool(hits)
    if hits:
        step, winner = hits[0]
        row.update({"winner": winner, "step": step, "later_events": ";".join(f"{f}@{t}" for t, f in hits[1:]),
                    "in_band": bool(pred["step_low"] <= step <= pred["step_high"])})
        for score, col in SCORES.items():
            row[f"hit_{score}"] = winner in str(pred[col]).split("+")
        winner_exposure = pred.get(f"exposure:{winner}", np.nan)
        rew, sampled = rewarded_before(run_dir, winner, step)
        row.update({"rewarded_before": rew, "sampled_before": sampled,
                    "rate_over_exposure": rew / (step * winner_exposure) if winner_exposure and winner_exposure > 0 else np.nan})
    rise = largest_rise(run_dir, seeded)
    for fam in seeded:
        row[f"rise:{fam}"] = rise[fam]
    row["rise_favours_exposure"] = max(rise, key=rise.get) == pred["exposure_favourite"]
    return row


def paired(rows, baseline):
    """Run-level comparison of exposure with one baseline over the runs with an event."""
    ev = rows[rows["event"]]
    e, b = ev["hit_exposure"].astype(bool), ev[f"hit_{baseline}"].astype(bool)
    only_e, only_b = int((e & ~b).sum()), int((~e & b).sum())
    return {f"{baseline}_correct": int(b.sum()), f"exposure_only_vs_{baseline}": only_e,
            f"{baseline}_only": only_b, f"both_vs_{baseline}": int((e & b).sum()),
            f"neither_vs_{baseline}": int((~e & ~b).sum()),
            f"mcnemar_p_vs_{baseline}": binomial_two_sided(only_e, only_e + only_b)}


def summary(table):
    out = []
    for label, g in (("all runs", table), ("F settings (exposure vs FPR alone)", table[table["baseline"] == "fpr"]),
                     ("P settings (exposure vs propensity alone)", table[table["baseline"] == "propensity"])):
        if len(g) == 0:
            continue
        ev = g[g["event"]]
        row = {"scope": label, "runs": len(g), "runs_with_event": len(ev),
               "exposure_correct": int(ev["hit_exposure"].astype(bool).sum()) if len(ev) else 0,
               "ea_correct": int(ev["hit_ea"].astype(bool).sum()) if len(ev) else 0,
               "events_in_band": int(ev["in_band"].astype(bool).sum()) if len(ev) else 0,
               "rewarded_before_min": ev["rewarded_before"].min() if len(ev) else np.nan,
               "rewarded_before_max": ev["rewarded_before"].max() if len(ev) else np.nan,
               "runs_without_event": int((~g["event"]).sum()),
               "no_event_rise_favours_exposure": int(g.loc[~g["event"], "rise_favours_exposure"].sum())}
        if len(ev):
            row.update(paired(g, "fpr"))
            row.update(paired(g, "propensity"))
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default="results/analysis")
    ap.add_argument("--plan", default=str(lowshare.PLAN))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.plan).read_text())
    preds = pd.read_csv(ROOT / cfg["predictions"])
    rows = []
    for _, pred in preds.iterrows():
        run_dir = lowshare.results_dir(cfg) / pred["run"]
        if not (run_dir / "events_tau0.05.json").exists():
            print(f"no results yet for {pred['run']}")
            continue
        rows.append(outcome(pred, run_dir))
    table = pd.DataFrame(rows)
    out = ROOT / args.analysis
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "lowshare_outcomes.csv", index=False)
    summ = summary(table) if len(table) else pd.DataFrame()
    summ.to_csv(out / "lowshare_summary.csv", index=False)
    write_meta(out, name="meta_lowshare_outcomes.json", stage="analysis", predictions=cfg["predictions"])
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        cols = [c for c in ("run", "exposure_favourite", "winner", "step", "in_band", "hit_exposure", "hit_fpr",
                            "hit_propensity", "rise_favours_exposure") if c in table]
        print(table[cols].to_string(index=False) if len(table) else "no finished runs")
        print(summ.to_string(index=False) if len(summ) else "")
    print(f"wrote {display_path(out / 'lowshare_outcomes.csv')} and {display_path(out / 'lowshare_summary.csv')}")


if __name__ == "__main__":
    main()
