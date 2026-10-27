"""Descriptive checks of the main runs that the paper reports beside the pre-registered tests.

Usage (after src.analysis.primary, whose units.csv it reads):
    python -m src.analysis.descriptive [--analysis results/analysis] [--results results/main] [--runs runs/grpo]

1. Rewards before an event. For every event, the training responses of the family that the judge accepted at
   the steps before the event's evaluation step (results/main/<run>/family_counts.csv, which counts every
   family response): how many rewarded responses a family had collected when it took over.
2. Acceptance by problem source. The hard pool is mostly DeepScaleR problems and the easy pool mostly MATH, so
   the H2 mechanism, the judge acceptance q and the advantage of a rewarded response at that rate, is
   recomputed within each source: acceptance of the step-0 evaluation outputs of the core runs (all 200 of a
   run with their source, in runs/grpo/<run>/eval.jsonl, which is not committed), averaged over the seeds,
   next to each pool's share of DeepScaleR training problems (pools/pools-v1.json).

Writes rewards_before_event.csv and acceptance_by_source.csv to --analysis.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis.confirmatory import group_size
from src.analysis.stats import rewarded_advantage
from src.paths import ROOT, display_path
from src.pipeline.records import write_meta

POOLS = ROOT / "pools" / "pools-v1.json"
CANDIDATES = ROOT / "data" / "processed" / "pool_candidates.jsonl"
STEP_SAMPLES = 256


def rewards_before_event(units, results_dir):
    """One row per event: rewarded and sampled responses of its family before the event step."""
    rows = []
    for _, u in units[units["event"]].iterrows():
        counts = pd.read_csv(Path(results_dir) / u["run"] / "family_counts.csv")
        before = counts[(counts["family"] == u["family"]) & (counts["step"] < u["time"])]
        rows.append({"run": u["run"], "group": u["group"], "judge": u["judge"], "family": u["family"],
                     "time": int(u["time"]), "exposure": STEP_SAMPLES * u["risk"],
                     "rewarded_before": int(before["accepted"].sum()), "sampled_before": int(before["n"].sum())})
    return pd.DataFrame(rows)


def eval_acceptance(path, step=0):
    """{source: [accepted, outputs]} over the evaluation outputs of one run at `step`."""
    counts = {}
    with Path(path).open() as f:
        for line in f:
            r = json.loads(line)
            if int(r["step"]) != step:
                continue
            c = counts.setdefault(r["source"], [0, 0])
            c[0] += r["accepted"] in (True, "True")
            c[1] += 1
    return counts


def acceptance_by_source(units, runs_dir, step=0):
    """Mean acceptance over the seeds per core judge, pool and source ("all" for every problem)."""
    rows = []
    for run, g in units[units["group"] == "core"].groupby("run"):
        counts = eval_acceptance(Path(runs_dir) / run / "eval.jsonl", step)
        counts["all"] = [sum(a for a, _ in counts.values()), sum(n for _, n in counts.values())]
        for source, (a, n) in counts.items():
            rows.append({"judge": g["judge"].iat[0], "pool": g["pool"].iat[0], "seed": g["seed"].iat[0],
                         "source": source, "n": n, "accept": a / n})
    table = pd.DataFrame(rows).groupby(["judge", "pool", "source"], as_index=False).agg(
        accept=("accept", "mean"), outputs=("n", "sum"))
    table["advantage_of_a_rewarded_response"] = [rewarded_advantage(q, group_size()) for q in table["accept"]]
    return table


def deepscaler_share(pools=POOLS, candidates=CANDIDATES):
    """{pool: share of its training problems that come from DeepScaleR}."""
    with Path(candidates).open() as f:
        source = {r["uid"]: r["source"] for r in map(json.loads, f)}
    manifest = json.loads(Path(pools).read_text())
    return {pool: sum(source[u] == "deepscaler" for u in uids["train"]) / len(uids["train"])
            for pool, uids in manifest["pools"].items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default="results/analysis")
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--runs", default="runs/grpo")
    ap.add_argument("--step", type=int, default=0)
    args = ap.parse_args()
    out = ROOT / args.analysis
    units = pd.read_csv(out / "units.csv")

    rewards = rewards_before_event(units, ROOT / args.results)
    acc = acceptance_by_source(units, ROOT / args.runs, args.step)
    share = deepscaler_share()
    acc["pool_deepscaler_share"] = acc["pool"].map(share)
    rewards.to_csv(out / "rewards_before_event.csv", index=False)
    acc.to_csv(out / "acceptance_by_source.csv", index=False)
    write_meta(out, name="meta_descriptive.json", stage="descriptive_checks", eval_step=args.step,
               deepscaler_share=share)

    fmt = lambda x: f"{x:.3g}"
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("1. Rewarded responses of each event's family before the event:")
        print(rewards.to_string(index=False, float_format=fmt))
        r = rewards["rewarded_before"]
        print(f"   {len(r)} events: {r.min()} to {r.max()}, median {r.median():g}")
        print(f"\n2. Judge acceptance of the step-{args.step} evaluation outputs by problem source "
              f"(DeepScaleR share of the training pool: hard {share['hard']:.3f}, easy {share['easy']:.3f}):")
        print(acc.to_string(index=False, float_format=fmt))
    print(f"\nwrote rewards_before_event.csv and acceptance_by_source.csv to {display_path(out)}")


if __name__ == "__main__":
    main()
