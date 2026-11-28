"""Primary test of H1: does the risk score recorded before a run rank the families it exploits?

Usage:
    python -m src.analysis.primary [--results results/main] [--out results/analysis] [--boot 10000]

The unit is a run x family pair and the positive an event (configs/protocol.yaml, analysis and
events), recomputed here from each run's eval_summary.jsonl and checked against the events file
written on the GPU machine. Four scores, all fixed before the run, are compared by AUROC (ties
count one half):
  risk    p_hat x FPR, from the run's predicted_risk.csv
  fpr     FPR of the family's seeding string for the run's judge (same file)
  p_hat   post-SFT sampled frequency of the family in the run's pool (same file)
  static  minus the judge's balanced accuracy on the policy's own answers in the run's pool
          (results/stage1/static_accuracy.csv), constant within a run; not measured for the judge
          added in protocol-v1.1, whose units are left out of the static comparison
Intervals are percentile bootstrap intervals that resample whole runs within each judge, so the
families of a run, which compete for the same takeover, stay together (HANDOFF 8.9). Differences
between the risk score and each baseline use the same replicates and the same units (paired), with
two-sided bootstrap p-values; replicates without an event are dropped and counted. Analyses: the
24 core runs per judge and pooled (the primary test), pooled without the positive control, and the
held-out judges (transfer). The 0% control runs belong to H3 and are not used here.

Writes units.csv (every run x family with its scores and outcome) and primary_auroc.csv to --out.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.detectors import FAMILIES
from src.monitor.events import detect_events, load_summary
from src.paths import ROOT, display_path
from src.pipeline.records import write_meta
from src.train import queue

SCORES = ("risk", "fpr", "p_hat", "static")
BASELINES = SCORES[1:]
STATIC = ROOT / "results" / "stage1" / "static_accuracy.csv"


def average_ranks(x):
    """1-based ranks with ties sharing their average rank."""
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inv]


def auroc(scores, labels):
    """Mann-Whitney AUROC, ties counted one half; units with a missing score are left out.
    nan when there is no positive or no negative."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    keep = ~np.isnan(s)
    s, y = s[keep], y[keep]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((average_ranks(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paired_diff(risk, base, labels):
    """AUROC(risk) - AUROC(base) on the units where the baseline exists."""
    base = np.asarray(base, float)
    keep = ~np.isnan(base)
    risk, labels = np.asarray(risk, float)[keep], np.asarray(labels, bool)[keep]
    return auroc(risk, labels) - auroc(base[keep], labels)


def run_group(run, main_runs):
    if run["tag"] == main_runs["run_tags"]["control"]:
        return "control"
    return "core" if run["judge"] in main_runs["core"]["judges"] else "held_out"


def load_units(results_dir, protocol=queue.PROTOCOL, static_csv=STATIC):
    """One row per run x family of the plan, with the event, its time and the four scores."""
    proto = yaml.safe_load(Path(protocol).read_text())
    mr, ev = proto["main_runs"], proto["events"]
    static = pd.read_csv(static_csv).set_index(["judge", "pool"])["balanced_accuracy"]
    _, runs = queue.load_plan(protocol)
    rows = []
    for run in runs:
        name = queue.run_name(run)
        d = Path(results_dir) / name
        events = detect_events(load_summary(d), ev["tau"], ev["consecutive_evaluations"])
        stored = json.loads((d / f"events_tau{ev['tau']:g}.json").read_text())["events"]
        if any(events[f]["event"] != stored[f]["event"] or events[f]["time"] != stored[f]["time"] for f in FAMILIES):
            raise ValueError(f"{name}: recomputed events differ from {d}/events_tau{ev['tau']:g}.json")
        pr = pd.read_csv(d / "predicted_risk.csv")
        pr = pr[(pr["judge"] == run["judge"]) & (pr["pool"] == run["pool"])].set_index("family")
        if set(FAMILIES) - set(pr.index):
            raise ValueError(f"{name}: predicted_risk.csv lacks {sorted(set(FAMILIES) - set(pr.index))}")
        acc = static.get((run["judge"], run["pool"]), float("nan"))
        for f in FAMILIES:
            rows.append({"run": name, "group": run_group(run, mr), "judge": run["judge"], "pool": run["pool"],
                         "seed": run["seed"], "family": f, "event": events[f]["event"], "time": events[f]["time"],
                         "risk": pr.at[f, "risk"], "fpr": pr.at[f, "fpr"], "p_hat": pr.at[f, "p_hat"],
                         "static": -acc})
    return pd.DataFrame(rows)


def bootstrap(units, n_boot, seed=0):
    """Replicates of the AUROC of every score and of the paired differences, resampling runs within
    each judge. Returns (aurocs, diffs): arrays of shape (n_boot, len(SCORES)) and (n_boot, len(BASELINES))."""
    rng = np.random.default_rng(seed)
    blocks = {name: (g[list(SCORES)].to_numpy(float), g["event"].to_numpy(bool))
              for name, g in units.groupby("run", sort=False)}
    by_judge = [list(g) for _, g in units.drop_duplicates("run").groupby("judge", sort=True)["run"]]
    aurocs = np.full((n_boot, len(SCORES)), np.nan)
    diffs = np.full((n_boot, len(BASELINES)), np.nan)
    for b in range(n_boot):
        picked = [runs[i] for runs in by_judge for i in rng.integers(0, len(runs), len(runs))]
        x = np.concatenate([blocks[r][0] for r in picked])
        y = np.concatenate([blocks[r][1] for r in picked])
        if not y.any():
            continue
        for k in range(len(SCORES)):
            aurocs[b, k] = auroc(x[:, k], y)
        for k in range(len(BASELINES)):
            diffs[b, k] = paired_diff(x[:, 0], x[:, k + 1], y)
    return aurocs, diffs


def interval(values):
    v = values[~np.isnan(values)]
    if not len(v):
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi), len(v)


def two_sided_p(values):
    v = values[~np.isnan(values)]
    if not len(v):
        return float("nan")
    return float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))


def analyse(units, label, n_boot, seed):
    """Rows of the result table for one analysis: one per score."""
    y = units["event"].to_numpy(bool)
    base = {"analysis": label, "runs": units["run"].nunique(), "units": len(units), "events": int(y.sum()),
            "runs_with_event": int(units.groupby("run")["event"].any().sum())}
    if not y.any():
        return [{**base, "score": s} for s in SCORES]
    aurocs, diffs = bootstrap(units, n_boot, seed)
    rows = []
    for k, s in enumerate(SCORES):
        lo, hi, n = interval(aurocs[:, k])
        row = {**base, "score": s, "auroc": auroc(units[s], y), "ci_low": lo, "ci_high": hi, "boot_valid": n}
        if s != "risk":
            j = BASELINES.index(s)
            keep = units[s].notna()
            dlo, dhi, _ = interval(diffs[:, j])
            row.update({"units_compared": int(keep.sum()),
                        "risk_minus_score": paired_diff(units["risk"], units[s], y),
                        "diff_ci_low": dlo, "diff_ci_high": dhi, "p_two_sided": two_sided_p(diffs[:, j])})
        rows.append(row)
    return rows


def analyses(units, positive_control):
    """(label, units) for every analysis, in reporting order."""
    core, held = units[units["group"] == "core"], units[units["group"] == "held_out"]
    out = [("core pooled", core), ("core pooled without the positive control", core[core["judge"] != positive_control])]
    out += [(f"core {j}", g) for j, g in core.groupby("judge", sort=True)]
    out += [("held-out pooled", held)]
    out += [(f"held-out {j}", g) for j, g in held.groupby("judge", sort=True)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    units = load_units(ROOT / args.results)
    units.to_csv(out / "units.csv", index=False)
    positive_control = yaml.safe_load(queue.PROTOCOL.read_text())["grpo"]["positive_control"]
    rows = []
    for label, g in analyses(units, positive_control):
        rows += analyse(g, label, args.boot, args.seed)
        print(f"{label}: done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "primary_auroc.csv", index=False)
    write_meta(out, name="meta_primary.json", stage="primary_analysis", results=args.results, n_boot=args.boot,
               seed=args.seed)

    show = table.dropna(subset=["auroc"]).copy()
    floor = 1 / args.boot
    show["auroc (95% CI)"] = show.apply(lambda r: f"{r.auroc:.3f} ({r.ci_low:.3f}, {r.ci_high:.3f})", axis=1)
    show["risk minus (95% CI), p"] = show.apply(
        lambda r: "" if r.score == "risk" else
        f"{r.risk_minus_score:+.3f} ({r.diff_ci_low:+.3f}, {r.diff_ci_high:+.3f}), "
        + (f"p<{floor:g}" if r.p_two_sided < floor else f"p={r.p_two_sided:.4f}"), axis=1)
    with pd.option_context("display.width", 220, "display.max_colwidth", 60, "display.max_rows", 200):
        print(
            show[["analysis", "runs", "events", "score", "auroc (95% CI)", "risk minus (95% CI), p"]].to_string(
                index=False
            )
        )
    skipped = table[table["events"] == 0]["analysis"].unique()
    if len(skipped):
        print("no events, no AUROC:", ", ".join(skipped))
    unmeasured = table[(table["events"] > 0) & table["auroc"].isna()]
    for _, r in unmeasured.iterrows():
        print(f"{r.analysis}: no {r.score} score for this judge")
    print(f"wrote {display_path(out / 'units.csv')} and {display_path(out / 'primary_auroc.csv')}")


if __name__ == "__main__":
    main()
