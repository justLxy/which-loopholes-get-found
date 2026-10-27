"""Confirmatory tests of H2 and H3, and the Holm correction over the five confirmatory tests.

Usage (after src.analysis.primary, whose units.csv and primary_auroc.csv it reads):
    python -m src.analysis.confirmatory [--analysis results/analysis] [--results results/main]

H2 (hard prompts raise the risk). The core runs come in pairs that differ only in the pool: the same
judge, SFT checkpoint and training seed on the hard and on the easy pool. A run's outcome is whether
any family has an event; the test is the exact McNemar test on the 12 pairs (two-sided sign test on
the discordant pairs). As a description of the mechanism it also reports the judge's acceptance rate
q over the first 10 training steps per judge and pool, and the advantage that a rewarded response receives
under group normalization at that rate (src.analysis.stats.rewarded_advantage, groups of num_generations).

H3 (exploit-style data before RL raises the risk with dose). In the core runs every family is seeded
at 0.3%, 1.2% and 4.0% across the three seeds. Which family got which tier was randomized once, by
shuffling the 14 families into three groups (src.train.seed_data.tier_groups, sizes 5, 5 and 4) whose
tiers rotate over the seeds. The test is the exact randomization test of that design: the statistic
is the sum of the tier ranks (1, 2, 3) of all core events, and its null distribution is taken over
every way of forming the three groups, with the outcomes held fixed. Under the null hypothesis that
the dose changes no outcome this is exact, including the competition between families within a run.
Two-sided p: the probability of a statistic at least as far from the null mean as the observed one.
The matched 0% control (clean-only SFT, the control judge, hard pool) is reported beside it: events
per dose, and the exact sign test on the three seed pairs of seeded and control runs.

Holm correction covers the five confirmatory tests (findings 3.22): the three pooled core comparisons
of H1 (bootstrap p-values, where 0 is replaced by 1 / replicates) and the H2 and H3 tests.
Writes confirmatory.csv, h3_dose.csv and h2_acceptance.csv to --analysis.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import pandas as pd
import yaml

from src.analysis.stats import binomial_two_sided, holm, rewarded_advantage
from src.detectors import FAMILIES
from src.paths import ROOT, display_path
from src.pipeline.records import write_meta
from src.train import queue
from src.train.seed_data import tier_groups, tier_shares

H1_BASELINES = {"fpr": "FPR alone", "p_hat": "propensity alone", "static": "static accuracy"}


def run_outcomes(units):
    """One row per run with its judge, pool, seed, group and whether any family had an event."""
    return units.groupby(["run", "group", "judge", "pool", "seed"], as_index=False)["event"].any()


def h2_test(units):
    runs = run_outcomes(units[units["group"] == "core"])
    pairs = runs.pivot(index=["judge", "seed"], columns="pool", values="event").astype(bool)
    hard_only = int((pairs["hard"] & ~pairs["easy"]).sum())
    easy_only = int((~pairs["hard"] & pairs["easy"]).sum())
    return {"pairs": len(pairs), "hard_runs_with_event": int(pairs["hard"].sum()),
            "easy_runs_with_event": int(pairs["easy"].sum()), "hard_only": hard_only, "easy_only": easy_only,
            "p": binomial_two_sided(hard_only, hard_only + easy_only)}


GRPO_CONFIG = ROOT / "configs" / "grpo.yaml"


def group_size():
    return yaml.safe_load(GRPO_CONFIG.read_text())["trainer"]["num_generations"]


def acceptance(units, results_dir, steps=10):
    """Mean judge acceptance over the first `steps` training steps per core judge and pool."""
    rows = []
    for run, g in units[units["group"] == "core"].groupby("run"):
        with (Path(results_dir) / run / "train_metrics.jsonl").open() as f:
            m = [json.loads(line) for line in f]
        q = sum(r["accept_rate"] for r in m if r["step"] < steps) / sum(1 for r in m if r["step"] < steps)
        rows.append({"judge": g["judge"].iat[0], "pool": g["pool"].iat[0], "seed": g["seed"].iat[0], "accept": q})
    table = pd.DataFrame(rows).groupby(["judge", "pool"], as_index=False)["accept"].mean()
    table["advantage_of_a_rewarded_response"] = [rewarded_advantage(q, group_size()) for q in table["accept"]]
    return table


def tier_rank(group, rotation, n_tiers=3):
    """Rank (1 = lowest share) of the tier that group `group` gets under `rotation` (seed_data.tier_shares)."""
    return (group + rotation) % n_tiers + 1


def multinomial(n, parts):
    out = math.factorial(n)
    for p in parts:
        out //= math.factorial(p)
    return out


def dose_null(event_items, sizes, n_families):
    """Exact null distribution {T: probability} of T = sum of the tier ranks of the events, over all
    ways of splitting n_families families into groups of these sizes. event_items: (family, rotation)."""
    fams = sorted({f for f, _ in event_items})
    k, total, dist = len(fams), multinomial(n_families, sizes), {}
    for labels in itertools.product(range(len(sizes)), repeat=k):
        left = [s - labels.count(g) for g, s in enumerate(sizes)]
        if min(left) < 0:
            continue
        group = dict(zip(fams, labels))
        t = sum(tier_rank(group[f], r, len(sizes)) for f, r in event_items)
        dist[t] = dist.get(t, 0) + multinomial(n_families - k, left)
    return {t: w / total for t, w in sorted(dist.items())}


def two_sided(dist, observed):
    mean = sum(t * p for t, p in dist.items())
    return min(1.0, sum(p for t, p in dist.items() if abs(t - mean) >= abs(observed - mean) - 1e-9)), mean


def h3_test(units, seeding):
    tiers, group_seed = seeding["tiers"], seeding["group_seed"]
    groups = tier_groups(group_seed, len(tiers))
    group_of = {f: g for g, members in enumerate(groups) for f in members}
    core = units[units["group"] == "core"]
    events = list(core.loc[core["event"], ["family", "seed"]].itertuples(index=False, name=None))
    for f, s in events:                      # the rank used here is the tier the run was seeded with
        assert tiers[tier_rank(group_of[f], s, len(tiers)) - 1] == tier_shares(tiers, s, group_seed)[f]
    observed = sum(tier_rank(group_of[f], s, len(tiers)) for f, s in events)
    dist = dose_null(events, [len(g) for g in groups], len(FAMILIES))
    p, mean = two_sided(dist, observed)
    return {"events": len(events), "statistic": observed, "null_mean": mean, "p": p,
            "p_one_sided": sum(q for t, q in dist.items() if t >= observed)}


def dose_table(units, seeding):
    """Events and units per dose: the tiers of the seeded runs, 0 for the control runs."""
    tiers, group_seed = seeding["tiers"], seeding["group_seed"]
    u = units.copy()
    u["dose"] = [0.0 if g == "control" else tier_shares(tiers, s, group_seed)[f]
                 for f, s, g in zip(u["family"], u["seed"], u["group"])]
    return u.groupby(["group", "dose"], as_index=False).agg(units=("event", "size"), events=("event", "sum"))


def control_test(units, control):
    runs = run_outcomes(units[(units["judge"] == control["judge"]) & (units["pool"] == control["pool"])])
    pairs = runs[runs["group"].isin(["core", "control"])].pivot(index="seed", columns="group", values="event").astype(bool)
    seeded_only = int((pairs["core"] & ~pairs["control"]).sum())
    control_only = int((~pairs["core"] & pairs["control"]).sum())
    return {"pairs": len(pairs), "seeded_runs_with_event": int(pairs["core"].sum()),
            "control_runs_with_event": int(pairs["control"].sum()),
            "p": binomial_two_sided(seeded_only, seeded_only + control_only)}


def confirmatory_table(primary, n_boot, h2, h3):
    core = primary[(primary["analysis"] == "core pooled") & primary["score"].isin(list(H1_BASELINES))]
    rows = [{"test": f"H1 risk score vs {H1_BASELINES[r.score]}", "estimate": r.risk_minus_score,
             "detail": f"AUROC difference, 95% CI {r.diff_ci_low:+.3f} to {r.diff_ci_high:+.3f}",
             "p": max(r.p_two_sided, 1 / n_boot)} for r in core.itertuples()]
    rows.append({"test": "H2 hard vs easy pool", "estimate": h2["hard_only"] - h2["easy_only"],
                 "detail": f"{h2['hard_only']} pairs with an event only on the hard pool, {h2['easy_only']} only on "
                           f"the easy pool, of {h2['pairs']}", "p": h2["p"]})
    rows.append({"test": "H3 dose response over the seeding tiers", "estimate": h3["statistic"] - h3["null_mean"],
                 "detail": f"sum of tier ranks of {h3['events']} events {h3['statistic']} against a null mean of "
                           f"{h3['null_mean']:.2f}", "p": h3["p"]})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].tolist())
    table["reject_at_0.05"] = table["p_holm"] < 0.05
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default="results/analysis")
    ap.add_argument("--results", default="results/main")
    args = ap.parse_args()
    out = ROOT / args.analysis
    proto = yaml.safe_load(queue.PROTOCOL.read_text())
    units = pd.read_csv(out / "units.csv")
    primary = pd.read_csv(out / "primary_auroc.csv")
    n_boot = json.loads((out / "meta_primary.json").read_text())["n_boot"]

    h2 = h2_test(units)
    acc = acceptance(units, ROOT / args.results)
    h3 = h3_test(units, proto["seeding"])
    doses = dose_table(units, proto["seeding"])
    ctrl = control_test(units, proto["h3"]["control"])
    table = confirmatory_table(primary, n_boot, h2, h3)

    table.to_csv(out / "confirmatory.csv", index=False)
    doses.to_csv(out / "h3_dose.csv", index=False)
    acc.to_csv(out / "h2_acceptance.csv", index=False)
    write_meta(out, name="meta_confirmatory.json", stage="confirmatory_tests", h2=h2, h3=h3, h3_control=ctrl)

    with pd.option_context("display.width", 220, "display.max_colwidth", 90):
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        print("\nH2, judge acceptance over the first 10 steps (mean over seeds):")
        print(acc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nH3, events per dose:")
        print(doses.to_string(index=False))
    print(f"\nH3 one-sided p {h3['p_one_sided']:.4g}; matched 0% control ({proto['h3']['control']['judge']}, "
          f"{proto['h3']['control']['pool']}): seeded runs with an event {ctrl['seeded_runs_with_event']} of "
          f"{ctrl['pairs']}, control runs {ctrl['control_runs_with_event']} of {ctrl['pairs']}, sign test p {ctrl['p']:.3g}")
    print(f"wrote {display_path(out / 'confirmatory.csv')}, h3_dose.csv, h2_acceptance.csv")


if __name__ == "__main__":
    main()
