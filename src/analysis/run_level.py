"""Run-level checks of the main runs: who wins within a run, when, and how difficulty enters the score.

Usage (after src.analysis.primary, whose units.csv it reads):
    python -m src.analysis.run_level [--analysis results/analysis] [--results results/main] [--boot 10000]

The primary test pools 336 run x family pairs. These checks ask the question of the paper at the level of a
run, on the seeded runs (24 core and 9 held-out); the 0% control runs have no event. Exposure is 256 p_hat FPR,
the expected rewarded responses of a family per GRPO step.

1. Winner ranks. For each run with an event, the rank of its first winner (ties at the first event go to the
   larger e_f, as in src.analysis.secondary) among the run's 14 families under exposure, FPR alone and
   propensity alone (ties share the best rank, as in Figure 4b). Top-1 hits of exposure and of each baseline
   are compared with the exact McNemar test on the discordant runs. Also the mean reciprocal rank, the runs
   whose winner is neither the top-FPR nor the top-propensity family, and the AUROC within each run with an
   event (all its events positive), averaged over those runs and compared run by run with a sign test.
2. Run level. The AUROC of a run's top exposure (and top FPR, top propensity) for "the run has an event".
3. Out-of-sample level. The lowest exposure of any core event, applied to the held-out runs: a run is
   predicted to have an event when its top family reaches it, and every family below it is predicted to stay
   flat.
4. Second events. In a run with two events, the rank of the second family among the 13 families left after
   the first winner.
5. Event time. Over the events, the Spearman correlation of the event step with exposure, FPR, propensity and
   the kinetic form log(1 / p_hat) / (FPR - q0) (q0 below), with a two-sided permutation p-value, and the
   least-squares slope of log step on log exposure. A family that needs a fixed number of rewards to take over
   and collects them at rate exposure has slope -1.
6. Difficulty-aware score. p_hat (FPR - q0) / sqrt(q0 (1 - q0)), proportional to the expected
   group-normalized advantage of a sampled family response when its acceptance is FPR and the group's mean
   acceptance is q0. q0 is the judge's acceptance of the run's 200 step-0 evaluation outputs, which are drawn
   from the starting policy before any update. Its AUROC against exposure and FPR uses the run bootstrap of
   src.analysis.primary. This score was defined after the runs. Its AUROC with a 95% run-bootstrap interval is also
   given for the three pooled analyses of Figure 5 (core, core without the positive control, held-out), in
   difficulty_auroc.csv.
7. Pool acceptance. In the first 10 training steps of the core runs, the share of a family's responses the
   judge accepted on the hard and on the easy pool (results/main/<run>/family_counts.csv), per judge and
   family with at least --min-n responses in each pool.

Writes run_level_winners.csv, run_level_tests.csv, event_timing.csv, difficulty_auroc.csv and pool_acceptance.csv to
--analysis.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis.descriptive import STEP_SAMPLES
from src.analysis.primary import auroc, average_ranks, interval, paired_diff, two_sided_p
from src.analysis.secondary import first_events, spearman
from src.analysis.stats import binomial_two_sided
from src.paths import ROOT, display_path
from src.train.queue import PROTOCOL
from src.pipeline.records import write_meta

SCORES = ("exposure", "fpr", "p_hat")


def seeded_units(units):
    """The seeded runs with exposure added."""
    u = units[units["group"].isin(["core", "held_out"])].copy()
    u["exposure"] = STEP_SAMPLES * u["risk"]
    return u


# ----------------------------------------------------------------------------- 1. winner ranks

def winner_ranks(units, firsts):
    """One row per run with an event: rank of its first winner under each score (ties share the best rank)."""
    rows = []
    for run, (step, fam) in firsts.items():
        g = units[units["run"] == run].set_index("family")
        row = {"run": run, "group": g["group"].iat[0], "judge": g["judge"].iat[0], "winner": fam, "step": step}
        row.update({f"rank_{s}": int(g[s].rank(ascending=False, method="min")[fam]) for s in SCORES})
        rows.append(row)
    return pd.DataFrame(rows)


def mcnemar_top1(ranks, a, b):
    """(runs where only a ranks the winner first, runs where only b does, exact two-sided p)."""
    hit_a, hit_b = ranks[f"rank_{a}"] == 1, ranks[f"rank_{b}"] == 1
    only_a, only_b = int((hit_a & ~hit_b).sum()), int((~hit_a & hit_b).sum())
    return only_a, only_b, binomial_two_sided(only_a, only_a + only_b)


def within_run_auroc(units, runs):
    """AUROC of every score within each of the given runs, all events of a run positive."""
    rows = []
    for run in runs:
        g = units[units["run"] == run]
        rows.append({"run": run, **{s: auroc(g[s], g["event"]) for s in SCORES}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 2. and 3. run level

def run_table(units):
    return units.groupby("run").agg(group=("group", "first"), judge=("judge", "first"), pool=("pool", "first"),
                                    event=("event", "any"), top_exposure=("exposure", "max"),
                                    top_fpr=("fpr", "max"), top_p_hat=("p_hat", "max"))


def level_check(units):
    """The lowest exposure of a core event and how it classifies the held-out runs and families."""
    level = units.loc[(units["group"] == "core") & units["event"], "exposure"].min()
    runs = run_table(units)
    held = runs[runs["group"] == "held_out"]
    fams = units[units["group"] == "held_out"]
    below = fams[fams["exposure"] < level]
    return {"level": level,
            "held_out_runs": len(held),
            "held_out_runs_correct": int(((held["top_exposure"] >= level) == held["event"]).sum()),
            "held_out_runs_predicted": int((held["top_exposure"] >= level).sum()),
            "held_out_runs_predicted_with_event": int(held.loc[held["top_exposure"] >= level, "event"].sum()),
            "held_out_families_below": len(below), "held_out_families_below_with_event": int(below["event"].sum())}


# ----------------------------------------------------------------------------- 4. second events

def second_events(units, firsts):
    """For runs with two or more events, the rank of the next family among those left after the winner."""
    rows = []
    for run, g in units[units["event"]].groupby("run"):
        step, winner = firsts[run]
        later = g[g["family"] != winner].sort_values("time")
        if later.empty:
            continue
        nxt = later.iloc[0]
        rest = units[(units["run"] == run) & (units["family"] != winner)].set_index("family")
        rows.append({"run": run, "first": winner, "first_step": step, "second": nxt["family"],
                     "second_step": int(nxt["time"]), "families_left": len(rest),
                     **{f"rank_{s}": int(rest[s].rank(ascending=False, method="min")[nxt["family"]]) for s in SCORES}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 5. event time

def step0_acceptance(runs, results_dir):
    """run -> judge acceptance of its step-0 evaluation outputs."""
    out = {}
    for run in runs:
        with (Path(results_dir) / run / "eval_summary.jsonl").open() as f:
            out[run] = next(r for r in map(json.loads, f) if r["step"] == 0)["accept_rate"]
    return out


def permutation_p(a, b, n=100_000, seed=0):
    """Two-sided Monte Carlo permutation p-value of the Spearman correlation of a and b; nan when either is
    constant."""
    ra, rb = average_ranks(np.asarray(a, float)), average_ranks(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    obs = abs(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(seed)
    perm = rb[np.argsort(rng.random((n, len(rb))), axis=1)]
    pc, ac = perm - perm.mean(1, keepdims=True), ra - ra.mean()
    r = pc @ ac / (np.sqrt((pc ** 2).sum(1)) * np.sqrt((ac ** 2).sum()))
    return float((np.abs(r) >= obs - 1e-12).mean())


def event_timing(units, firsts, q0):
    """One row per event with its step, scores and kinetic form."""
    ev = units[units["event"]].copy()
    ev["q0"] = ev["run"].map(q0)
    ev["first"] = [firsts[r][1] == f for r, f in zip(ev["run"], ev["family"])]
    gap = ev["fpr"] - ev["q0"]
    ev["kinetic"] = np.where(gap > 0, np.log(1 / ev["p_hat"]) / gap.where(gap > 0, 1.0), np.inf)
    return ev[["run", "group", "family", "time", "first", "exposure", "fpr", "p_hat", "q0", "kinetic"]]


def timing_rows(ev):
    rows = []
    for label, g in (("all events", ev), ("first events", ev[ev["first"]])):
        row = {"analysis": f"event time, {label}", "n": len(g)}
        for s in ("exposure", "fpr", "p_hat", "kinetic"):
            row[f"spearman_{s}"] = spearman(g["time"], g[s])
            row[f"perm_p_{s}"] = permutation_p(g["time"], g[s])
        row["loglog_slope_exposure"] = float(np.polyfit(np.log(g["exposure"]), np.log(g["time"]), 1)[0])
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- 6. difficulty-aware score

def with_difficulty(units, q0):
    u = units.copy()
    q = u["run"].map(q0)
    u["q0"] = q
    u["difficulty"] = u["p_hat"] * (u["fpr"] - q) / np.sqrt(q * (1 - q))
    return u


def paired_bootstrap(units, a, b, n_boot, seed=0):
    """Run-bootstrap (within judge) replicates of AUROC(a) - AUROC(b), as in src.analysis.primary."""
    rng = np.random.default_rng(seed)
    blocks = {r: (g[[a, b]].to_numpy(float), g["event"].to_numpy(bool)) for r, g in units.groupby("run", sort=False)}
    by_judge = [list(g) for _, g in units.drop_duplicates("run").groupby("judge", sort=True)["run"]]
    out = np.full(n_boot, np.nan)
    for k in range(n_boot):
        picked = [runs[i] for runs in by_judge for i in rng.integers(0, len(runs), len(runs))]
        x = np.concatenate([blocks[r][0] for r in picked])
        y = np.concatenate([blocks[r][1] for r in picked])
        if y.any():
            out[k] = paired_diff(x[:, 0], x[:, 1], y)
    return out


def bootstrap_auroc(units, col, n_boot, seed=0):
    """Run-bootstrap (within judge) replicates of the AUROC of one score."""
    rng = np.random.default_rng(seed)
    blocks = {r: (g[col].to_numpy(float), g["event"].to_numpy(bool)) for r, g in units.groupby("run", sort=False)}
    by_judge = [list(g) for _, g in units.drop_duplicates("run").groupby("judge", sort=True)["run"]]
    out = np.full(n_boot, np.nan)
    for k in range(n_boot):
        picked = [runs[i] for runs in by_judge for i in rng.integers(0, len(runs), len(runs))]
        y = np.concatenate([blocks[r][1] for r in picked])
        if y.any():
            out[k] = auroc(np.concatenate([blocks[r][0] for r in picked]), y)
    return out


def difficulty_auroc(u, n_boot, seed, positive_control):
    """AUROC and 95% interval of the difficulty-aware score for the pooled analyses of Figure 5."""
    core = u[u["group"] == "core"]
    sets = (("core pooled", core), ("core pooled without the positive control", core[core["judge"] != positive_control]),
            ("held-out pooled", u[u["group"] == "held_out"]))
    rows = []
    for label, g in sets:
        lo, hi, n = interval(bootstrap_auroc(g, "difficulty", n_boot, seed))
        rows.append({"analysis": label, "score": "difficulty", "runs": g["run"].nunique(), "units": len(g),
                     "events": int(g["event"].sum()), "auroc": auroc(g["difficulty"], g["event"]),
                     "ci_low": lo, "ci_high": hi, "boot_valid": n})
    return pd.DataFrame(rows)


def difficulty_rows(u, firsts, n_boot, seed):
    rows = []
    sets = (("core", u[u["group"] == "core"]), ("core hard pool", u[(u["group"] == "core") & (u["pool"] == "hard")]),
            ("held-out", u[u["group"] == "held_out"]))
    for label, g in sets:
        row = {"analysis": f"difficulty-aware score, {label}", "n": len(g),
               **{f"auroc_{s}": auroc(g[s], g["event"]) for s in ("difficulty", "exposure", "fpr")}}
        if label == "core":
            for base in ("exposure", "fpr"):
                d = paired_bootstrap(g, "difficulty", base, n_boot, seed)
                lo, hi, _ = interval(d)
                row.update({f"minus_{base}": paired_diff(g["difficulty"], g[base], g["event"]),
                            f"minus_{base}_ci_low": lo, f"minus_{base}_ci_high": hi,
                            f"minus_{base}_p": two_sided_p(d)})
        rows.append(row)
    ranks = [int(u[u["run"] == r].set_index("family")["difficulty"].rank(ascending=False, method="min")[f])
             for r, (_, f) in firsts.items()]
    rows.append({"analysis": "difficulty-aware score, winners ranked first", "n": len(ranks),
                 "top1": sum(k == 1 for k in ranks)})
    return rows


# ----------------------------------------------------------------------------- 7. pool acceptance

def pool_acceptance(units, results_dir, steps=10, min_n=50):
    """Accepted share of each family's training responses in the first steps, per judge, family and pool,
    for the core judges whose families have at least min_n responses in each pool."""
    rows = []
    for run, g in units[units["group"] == "core"].groupby("run"):
        c = pd.read_csv(Path(results_dir) / run / "family_counts.csv")
        c = c[c["step"] < steps].groupby("family")[["n", "accepted"]].sum()
        rows += [{"judge": g["judge"].iat[0], "pool": g["pool"].iat[0], "family": f, "n": int(r["n"]),
                  "accepted": int(r["accepted"])} for f, r in c.iterrows()]
    t = pd.DataFrame(rows).groupby(["judge", "family", "pool"], as_index=False)[["n", "accepted"]].sum()
    wide = t.pivot_table(index=["judge", "family"], columns="pool", values=["n", "accepted"]).dropna()
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide[(wide["n_hard"] >= min_n) & (wide["n_easy"] >= min_n)].reset_index()
    wide["accept_hard"] = wide["accepted_hard"] / wide["n_hard"]
    wide["accept_easy"] = wide["accepted_easy"] / wide["n_easy"]
    fpr = units.drop_duplicates(["judge", "family"]).set_index(["judge", "family"])["fpr"]
    wide["audit_fpr"] = [fpr[(j, f)] for j, f in zip(wide["judge"], wide["family"])]
    return wide


# ----------------------------------------------------------------------------- main

def summary_rows(u, firsts, ranks):
    rows = []
    hits = {s: int((ranks[f"rank_{s}"] == 1).sum()) for s in SCORES}
    rows.append({"analysis": "winner ranked first", "n": len(ranks), **{f"top1_{s}": hits[s] for s in SCORES},
                 **{f"mrr_{s}": float((1 / ranks[f"rank_{s}"]).mean()) for s in SCORES},
                 "neither_fpr_nor_p_top": int(((ranks["rank_fpr"] != 1) & (ranks["rank_p_hat"] != 1)).sum())})
    for base in ("fpr", "p_hat"):
        only_e, only_b, p = mcnemar_top1(ranks, "exposure", base)
        rows.append({"analysis": f"McNemar top-1, exposure vs {base}", "n": len(ranks),
                     "only_exposure": only_e, "only_baseline": only_b, "p": p})
    wr = within_run_auroc(u, list(firsts))
    row = {"analysis": "within-run AUROC, mean over runs with an event", "n": len(wr),
           **{f"auroc_{s}": float(wr[s].mean()) for s in SCORES}}
    for base in ("fpr", "p_hat"):
        d = wr["exposure"] - wr[base]
        hi, lo = int((d > 0).sum()), int((d < 0).sum())
        row.update({f"runs_exposure_above_{base}": hi, f"runs_exposure_below_{base}": lo,
                    f"sign_p_{base}": binomial_two_sided(hi, hi + lo)})
    rows.append(row)
    runs = run_table(u)
    for label, g in (("seeded", runs), ("core", runs[runs["group"] == "core"]), ("hard pool", runs[runs["pool"] == "hard"])):
        rows.append({"analysis": f"run-level AUROC of the top score, {label}", "n": len(g),
                     "runs_with_event": int(g["event"].sum()),
                     **{f"auroc_{s}": auroc(g[f"top_{s}"], g["event"]) for s in SCORES}})
    hard = runs[runs["pool"] == "hard"]
    level = level_check(u)
    above = hard["top_exposure"] >= level["level"]
    rows.append({"analysis": "hard-pool runs by top exposure at the core level", "n": len(hard),
                 "runs_above": int(above.sum()), "runs_above_with_event": int(hard.loc[above, "event"].sum()),
                 "runs_below": int((~above).sum()), "runs_below_with_event": int(hard.loc[~above, "event"].sum())})
    rows.append({"analysis": "core level applied to the held-out runs", **level})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default="results/analysis")
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--min-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, results = ROOT / args.analysis, ROOT / args.results
    u = seeded_units(pd.read_csv(out / "units.csv"))
    firsts = first_events(u, results)
    q0 = step0_acceptance(u["run"].unique(), results)

    ranks = winner_ranks(u, firsts)
    second = second_events(u, firsts)
    ranks = ranks.merge(second[["run", "second", "second_step", "rank_exposure"]].rename(
        columns={"rank_exposure": "second_rank_exposure_among_left"}), on="run", how="left")
    ev = event_timing(u, firsts, q0)
    tests = pd.DataFrame(summary_rows(u, firsts, ranks) + timing_rows(ev)
                         + difficulty_rows(with_difficulty(u, q0), firsts, args.boot, args.seed))
    acc = pool_acceptance(u, results, min_n=args.min_n)
    positive_control = yaml.safe_load(PROTOCOL.read_text())["grpo"]["positive_control"]
    diff_auroc = difficulty_auroc(with_difficulty(u, q0), args.boot, args.seed, positive_control)

    ranks.to_csv(out / "run_level_winners.csv", index=False)
    tests.to_csv(out / "run_level_tests.csv", index=False)
    ev.to_csv(out / "event_timing.csv", index=False)
    acc.to_csv(out / "pool_acceptance.csv", index=False)
    diff_auroc.to_csv(out / "difficulty_auroc.csv", index=False)
    write_meta(out, name="meta_run_level.json", stage="run_level_checks", n_boot=args.boot, seed=args.seed,
               min_n=args.min_n)

    fmt = lambda x: f"{x:.4g}"
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(ranks.to_string(index=False))
        for _, r in tests.iterrows():
            print("\n" + r["analysis"] + ": " + ", ".join(f"{k} {fmt(v) if isinstance(v, float) else v}"
                                                         for k, v in r.drop("analysis").dropna().items()))
        print("\n" + diff_auroc.to_string(index=False, float_format=fmt))
        print(f"\npool acceptance, first 10 steps, {len(acc)} judge x family pairs: hard "
              f"{acc['accepted_hard'].sum() / acc['n_hard'].sum():.3f}, easy {acc['accepted_easy'].sum() / acc['n_easy'].sum():.3f}; "
              f"hard higher in {int((acc['accept_hard'] > acc['accept_easy']).sum())}, easy higher in "
              f"{int((acc['accept_easy'] > acc['accept_hard']).sum())}")
    print(f"\nwrote run_level_winners.csv, run_level_tests.csv, event_timing.csv, difficulty_auroc.csv, "
          f"pool_acceptance.csv to {display_path(out)}")


if __name__ == "__main__":
    main()
