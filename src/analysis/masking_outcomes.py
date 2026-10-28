"""Outcomes of experiment B, the masking runs (configs/masking.yaml), against the predictions made before them.

Usage (after src.analysis.primary and src.analysis.run_level, whose outputs it reads):
    python -m src.analysis.masking_outcomes [--analysis results/analysis] [--masking results/masking]

For each masked run: the first event among the 14 families (the no_box side metric is left out; ties go to the
larger e_f at that evaluation, as in src.analysis.secondary), any later family event, and the comparison with
results/masking/predictions.csv. The winner is scored against four rankings of the run's families, all computed
from the rebuilt checkpoint's scores with the masked family's FPR set to zero: exposure, FPR alone, propensity
alone, and the difficulty-aware score of src.analysis.run_level with q0 from the run's step-0 evaluation. The step
is checked against the 90% band recorded with the prediction. For reference, the unmasked main run with the same
judge, pool and seed gives its first winner and step.

Pooled with the 12 main runs with an event, the first winner's rank under each score gives the top-1 counts over
all runs with an event and the exact McNemar test of exposure against each baseline. The masked runs share their
checkpoint and seed with a main run, so these pooled pairs are not independent of the main ones; the pooled test
is reported as a description.

Writes masking_outcomes.csv and masking_summary.csv to --analysis.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.descriptive import STEP_SAMPLES
from src.analysis.stats import binomial_two_sided
from src.detectors import FAMILIES
from src.paths import ROOT, display_path
from src.pipeline.records import write_meta
from src.train import masking, queue

SCORES = ("exposure", "fpr", "p_hat", "difficulty")


def family_events(run_dir):
    """[(step, family)] of the family events of a run in time order; ties broken by the larger e_f."""
    events = json.loads((Path(run_dir) / "events_tau0.05.json").read_text())["events"]
    with (Path(run_dir) / "eval_summary.jsonl").open() as f:
        summary = {r["step"]: r for r in map(json.loads, f)}
    hits = [(v["time"], f) for f, v in events.items() if v["event"] and f in FAMILIES]
    return sorted(hits, key=lambda h: (h[0], -summary.get(h[0], {}).get(f"e/{h[1]}", 0.0)))


def step0_acceptance(run_dir):
    with (Path(run_dir) / "eval_summary.jsonl").open() as f:
        return next(r for r in map(json.loads, f) if r["step"] == 0)["accept_rate"]


def scored_families(scores, run, q0):
    """The run's families with every score, the masked family's FPR set to zero."""
    s = scores[(scores["judge"] == run["judge"]) & (scores["pool"] == run["pool"])].set_index("family").copy()
    s.loc[run["mask"], ["fpr", "risk"]] = 0.0
    s["exposure"] = STEP_SAMPLES * s["risk"]
    s["difficulty"] = s["p_hat"] * (s["fpr"] - q0) / np.sqrt(q0 * (1 - q0))
    return s


def outcome_row(cfg, run, prediction, results_dir, main_winners, root=ROOT):
    name = queue.run_name(run)
    d = Path(results_dir) / name
    events = family_events(d)
    q0 = step0_acceptance(d)
    s = scored_families(pd.read_csv(masking.committed_scores(cfg, run, root)), run, q0)
    row = {"run": name, "judge": run["judge"], "seed": run["seed"], "masked": run["mask"], "q0": q0,
           "predicted": prediction["winner"], "predicted_step": prediction["step"],
           "step_low": prediction["step_low"], "step_high": prediction["step_high"],
           "fpr_alone_predicted": prediction["fpr_alone_winner"],
           "propensity_alone_predicted": prediction["propensity_alone_winner"],
           "event": bool(events), "winner": events[0][1] if events else None, "step": events[0][0] if events else None,
           "later_events": "; ".join(f"{f} ({t})" for t, f in events[1:])}
    if events:
        w = events[0][1]
        row.update({f"rank_{c}": int(s[c].rank(ascending=False, method="min")[w]) for c in SCORES})
        row.update({"winner_exposure": s.loc[w, "exposure"], "winner_fpr": s.loc[w, "fpr"],
                     "hit": w == prediction["winner"],
                     "in_band": bool(prediction["step_low"] <= events[0][0] <= prediction["step_high"])})
        if row["predicted"] in s.index:
            row["predicted_fpr"] = s.loc[row["predicted"], "fpr"]
    main = main_winners.get((run["judge"], run["pool"], run["seed"]))
    if main is not None:
        row.update({"main_winner": main["winner"], "main_step": main["step"]})
    return row


def pooled(outcomes, main_ranks):
    """Top-1 counts and exact McNemar tests over the main runs and the masked runs with an event."""
    ranks = pd.concat([main_ranks[[f"rank_{c}" for c in ("exposure", "fpr", "p_hat")]],
                       outcomes.loc[outcomes["event"], [f"rank_{c}" for c in ("exposure", "fpr", "p_hat")]]])
    hit = {c: ranks[f"rank_{c}"] == 1 for c in ("exposure", "fpr", "p_hat")}
    rows = [{"analysis": "all runs with an event, main and masked", "n": len(ranks),
             **{f"top1_{c}": int(h.sum()) for c, h in hit.items()}}]
    for base in ("fpr", "p_hat"):
        only_e, only_b = int((hit["exposure"] & ~hit[base]).sum()), int((~hit["exposure"] & hit[base]).sum())
        rows.append({"analysis": f"McNemar top-1, exposure vs {base}, main and masked (pairs not independent)",
                     "n": len(ranks), "only_exposure": only_e, "only_baseline": only_b,
                     "p": binomial_two_sided(only_e, only_e + only_b)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default="results/analysis")
    ap.add_argument("--masking", default="results/masking")
    ap.add_argument("--plan", default=str(masking.PLAN))
    args = ap.parse_args()
    out, results = ROOT / args.analysis, ROOT / args.masking
    cfg, runs = masking.load_plan(args.plan)
    predictions = pd.read_csv(ROOT / cfg["predictions"]).set_index("run")
    main_ranks = pd.read_csv(out / "run_level_winners.csv")
    main_winners = {}
    for _, r in main_ranks.iterrows():
        judge, rest = r["run"].split("-", 1)
        pool, seed = rest.split("-")[0], int(rest.split("-")[1][1:])
        main_winners[(judge, pool, seed)] = {"winner": r["winner"], "step": r["step"]}
    outcomes = pd.DataFrame([outcome_row(cfg, r, predictions.loc[queue.run_name(r)], results, main_winners)
                             for r in runs])
    hits = outcomes["hit"].fillna(False).astype(bool)
    summary = [{"analysis": "masked runs", "n": len(outcomes), "with_event": int(outcomes["event"].sum()),
                "winner_predicted_by_exposure": int(hits.sum()),
                "step_in_band_when_predicted": int(outcomes.loc[hits, "in_band"].sum()),
                "winner_predicted_by_fpr_alone": int((outcomes["winner"] == outcomes["fpr_alone_predicted"]).sum()),
                "winner_predicted_by_propensity_alone": int((outcomes["winner"] == outcomes["propensity_alone_predicted"]).sum()),
                "winner_first_by_difficulty_score": int((outcomes["rank_difficulty"] == 1).sum())}]
    summary += pooled(outcomes, main_ranks)
    outcomes.to_csv(out / "masking_outcomes.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "masking_summary.csv", index=False)
    write_meta(out, name="meta_masking_outcomes.json", stage="masking_outcomes", plan=display_path(args.plan))

    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print(outcomes.drop(columns=["judge", "seed"]).to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        for r in summary:
            print(", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
    print(f"wrote masking_outcomes.csv and masking_summary.csv to {display_path(out)}")


if __name__ == "__main__":
    main()
