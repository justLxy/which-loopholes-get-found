"""Secondary analyses of the main runs (configs/protocol.yaml, analysis): competing risks, early growth
and the winner conditional logit.

Usage (after src.analysis.primary, whose units.csv it reads):
    python -m src.analysis.secondary [--analysis results/analysis] [--results results/main] [--boot 2000]

Covariates. log p is the log of the post-SFT Jeffreys-smoothed frequency, which is never zero. The FPR
of a seeding string is zero for many judges, so the log models use the Jeffreys-smoothed FPR
(k + 0.5) / (n + 1) from the k acceptances in n = 471 audit problems (results/stage2/string_fpr.csv);
log r = log p + log FPR. Runs: the seeded runs (24 core and 9 held-out), reported for the core runs and
for all of them; the 0% control runs have no event.

1. Competing risks. One family usually takes over a run and crowds out the others, so each family is
   censored at the first event of another family in its run (cause-specific hazards); families of runs
   without an event are censored at step 300. Ties at the first event go to the family with the larger
   e_f at that evaluation. Kaplan-Meier estimates of the time to a run's first event by pool, and of
   each family's cause-specific time to event by seeding tier. A Cox model of the cause-specific
   hazard on log r (Breslow ties) is fitted only where there are at least 10 events per covariate
   (HANDOFF 8.9), with a 95% interval for the hazard ratio from the run bootstrap within judges.
2. Early growth. For every family, g = log((n_f(w2) + 0.5) / N) - log((n_f(w1) + 0.5) / N) over the
   training samples (family_counts.csv, N = 2,560 per 10-step window): w1 = steps 0-9, w2 = steps
   E-20 to E-11 for a run whose first event is at step E (the 10 steps ending 10 steps before it),
   steps 10-19 if E is before step 30, steps 40-49 if the run has no event. Spearman correlation of g
   with log r, log FPR and log p (95% intervals from the run bootstrap within judges) and least squares
   of g on each and on log FPR and log p together, per judge and pooled. It is defined for every unit,
   including runs without an event.
3. Winner conditional logit. Over the runs with an event, P(first event in family f) is proportional to
   exp(b1 log p_f + b2 log FPR_f). The risk score predicts b1 = b2 > 0, the single-factor baselines
   b1 = 0 (FPR alone) or b2 = 0 (propensity alone); each restriction is tested with a likelihood-ratio
   test against the full model (chi-square, 1 df). When a model separates the data, the winner always
   having the highest linear score, its likelihood tends to 1 as the coefficients grow: the maximum is
   then a supremum and the estimate is reported as not finite.

Writes km_runs.csv, km_tiers.csv, cox.csv, early_growth.csv, winner_logit.csv to --analysis.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis.primary import average_ranks
from src.paths import ROOT, display_path
from src.pipeline.records import write_meta
from src.train import queue
from src.train.seed_data import tier_shares

STRING_FPR = ROOT / "results" / "stage2" / "string_fpr.csv"
STEP_SAMPLES = 256
WINDOW = 10
BOUND = 60.0                 # coefficient size taken as "not finite" (separation)
KM_STEPS = (50, 100, 150, 200, 250, 300)


# ----------------------------------------------------------------------------- covariates and outcomes

def add_covariates(units, string_fpr=STRING_FPR):
    s = pd.read_csv(string_fpr)[["judge", "family", "n", "accepted"]]
    u = units.merge(s, on=["judge", "family"], how="left", validate="many_to_one")
    if u["n"].isna().any():
        raise ValueError(f"no string FPR for {u.loc[u['n'].isna(), 'judge'].unique()}")
    if not np.allclose(u["accepted"] / u["n"], u["fpr"]):
        raise ValueError("string_fpr.csv and predicted_risk.csv disagree on the FPR")
    u["log_p"] = np.log(u["p_hat"])
    u["log_fpr"] = np.log((u["accepted"] + 0.5) / (u["n"] + 1))
    u["log_r"] = u["log_p"] + u["log_fpr"]
    return u.drop(columns=["n", "accepted"])


def first_events(units, results_dir):
    """run -> (first event step, winning family) for the runs with an event; ties go to the larger e_f."""
    out = {}
    for run, g in units[units["event"]].groupby("run"):
        t = g["time"].min()
        tied = list(g.loc[g["time"] == t, "family"])
        if len(tied) > 1:
            with (Path(results_dir) / run / "eval_summary.jsonl").open() as f:
                row = next(r for r in map(json.loads, f) if r["step"] == t)
            tied.sort(key=lambda fam: -row.get(f"e/{fam}", 0.0))
        out[run] = (int(t), tied[0])
    return out


def cause_specific(units, firsts, last_step=300):
    """Each unit's (time, status): the winner has its event at E, every other family of the run is
    censored at E; families of runs without an event are censored at the last step."""
    u = units.copy()
    first = u["run"].map(lambda r: firsts.get(r, (last_step, None)))
    u["cs_time"] = [e for e, _ in first]
    u["cs_event"] = [w == f for (_, w), f in zip(first, u["family"])]
    return u


# ----------------------------------------------------------------------------- partial likelihoods

def risk_sets(units, cov, strata=None):
    """(X of the risk set, sum of x over the events, number of events) per event time (Breslow).
    With strata (the run), every stratum is a set of its own: the conditional logit."""
    x = units[cov].to_numpy(float)
    sets = []
    groups = units.groupby(strata).indices.values() if strata else [np.arange(len(units))]
    for idx in groups:
        t, d = units["cs_time"].to_numpy()[idx], units["cs_event"].to_numpy(bool)[idx]
        for time in np.unique(t[d]):
            at_risk, ev = idx[t >= time], idx[(t == time) & d]
            sets.append((x[at_risk], x[ev].sum(axis=0), len(ev)))
    return sets


def loglik(beta, sets):
    """Log partial likelihood, gradient and Hessian."""
    ll, grad, hess = 0.0, np.zeros(len(beta)), np.zeros((len(beta), len(beta)))
    for xr, xe, d in sets:
        eta = xr @ beta
        m = eta.max()
        w = np.exp(eta - m)
        s = w.sum()
        mean = w @ xr / s
        ll += xe @ beta - d * (m + math.log(s))
        grad += xe - d * mean
        hess -= d * ((xr * w[:, None]).T @ xr / s - np.outer(mean, mean))
    return ll, grad, hess


def fit(sets, k, bound=BOUND):
    """Maximize the log partial likelihood by Newton steps with step halving. Returns (beta, loglik,
    finite): finite is False when the coefficients run past `bound`, i.e. the data are separated."""
    beta = np.zeros(k)
    ll, grad, hess = loglik(beta, sets)
    for _ in range(200):
        step = np.linalg.solve(hess - 1e-9 * np.eye(k), -grad)
        size = 1.0
        while size > 1e-6:
            new = beta + size * step
            new_ll, new_grad, new_hess = loglik(new, sets)
            if new_ll >= ll - 1e-12:
                break
            size /= 2
        done = abs(new_ll - ll) < 1e-10
        beta, ll, grad, hess = new, new_ll, new_grad, new_hess
        if done or np.abs(beta).max() > bound:
            break
    finite = np.abs(beta).max() <= bound
    if finite and np.abs(beta).max() > 1e-8:
        # an interior maximum loses likelihood when the coefficients are doubled; under separation the
        # likelihood keeps rising along the ray, and Newton only stopped because the gains became tiny
        finite = loglik(2 * beta, sets)[0] < ll - 1e-12
    return beta, ll, bool(finite)


def chi2_1df(stat):
    return math.erfc(math.sqrt(max(stat, 0.0) / 2))


# ----------------------------------------------------------------------------- bootstrap

def resampler(units):
    """A function rng -> bootstrap sample of whole runs within each judge (repeated runs renamed apart)."""
    blocks = {run: g for run, g in units.groupby("run", sort=False)}
    by_judge = [list(runs) for _, runs in units.drop_duplicates("run").groupby("judge")["run"]]

    def sample(rng):
        parts = []
        for runs in by_judge:
            for i, k in enumerate(rng.integers(0, len(runs), len(runs))):
                parts.append(blocks[runs[k]].assign(run=f"{runs[k]}#{i}"))
        return pd.concat(parts, ignore_index=True)
    return sample


# ----------------------------------------------------------------------------- 1. competing risks

def km(times, events, at=KM_STEPS):
    """Kaplan-Meier cumulative incidence 1 - S(t) at the given steps."""
    times, events = np.asarray(times), np.asarray(events, bool)
    surv, out = 1.0, {}
    grid = sorted(set(times[events]))
    for t in grid:
        n_risk, d = (times >= t).sum(), (events & (times == t)).sum()
        surv *= 1 - d / n_risk
        for a in at:
            if a >= t:
                out[a] = 1 - surv
    return [out.get(a, 0.0) for a in at]


def km_tables(u, seeding):
    runs = u.drop_duplicates("run")
    rows = []
    for label, sel in (("seeded, hard pool", runs["pool"] == "hard"), ("seeded, easy pool", runs["pool"] == "easy"),
                       ("core, hard pool", (runs["pool"] == "hard") & (runs["group"] == "core")),
                       ("held-out, hard pool", runs["group"] == "held_out")):
        r = runs[sel]
        has = r["run"].isin(u.loc[u["cs_event"], "run"])
        rows.append({"runs": label, "n": len(r), "with_event": int(has.sum()),
                     **dict(zip([f"step_{a}" for a in KM_STEPS], km(r["cs_time"], has)))})
    tiers = []
    core = u[u["group"] == "core"].copy()
    core["tier"] = [
        tier_shares(seeding["tiers"], s, seeding["group_seed"])[f] for f, s in zip(core["family"], core["seed"])
    ]
    for tier, g in core.groupby("tier"):
        tiers.append({"tier": tier, "units": len(g), "first_events": int(g["cs_event"].sum()),
                      **dict(zip([f"step_{a}" for a in KM_STEPS], km(g["cs_time"], g["cs_event"])))})
    return pd.DataFrame(rows), pd.DataFrame(tiers)


def cox_table(u, n_boot, seed):
    rows = []
    for label, g in (("core", u[u["group"] == "core"]), ("all seeded", u)):
        events = int(g["cs_event"].sum())
        row = {"runs": label, "units": len(g), "events": events, "covariate": "log_r"}
        if events < 10:
            rows.append({**row, "fitted": False, "note": "fewer than 10 events per covariate (HANDOFF 8.9)"})
            continue
        sets = risk_sets(g, ["log_r"])
        beta, ll, finite = fit(sets, 1)
        ll0 = loglik(np.zeros(1), sets)[0]
        rng, boot, sample = np.random.default_rng(seed), [], resampler(g)
        for _ in range(n_boot):
            b_sets = risk_sets(sample(rng), ["log_r"])
            b, _, ok = fit(b_sets, 1) if b_sets else (None, None, False)
            boot.append(b[0] if ok else np.nan)
        boot = np.array(boot)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        rows.append({**row, "fitted": True, "finite": finite, "beta": beta[0], "hazard_ratio": math.exp(beta[0]),
                     "hr_ci_low": math.exp(lo), "hr_ci_high": math.exp(hi), "boot_finite": int(np.isfinite(boot).sum()),
                     "lr_stat": 2 * (ll - ll0), "p": chi2_1df(2 * (ll - ll0))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 2. early growth

def windows(first_step):
    """(w1, w2) as step ranges for a run whose first event is at first_step (None: no event)."""
    w1 = range(0, WINDOW)
    if first_step is None:
        return w1, range(40, 50)
    if first_step < 30:
        return w1, range(10, 20)
    return w1, range(first_step - 20, first_step - 10)


def growth(u, firsts, results_dir):
    out = []
    for run, g in u.groupby("run", sort=False):
        counts = pd.read_csv(Path(results_dir) / run / "family_counts.csv")
        w1, w2 = windows(firsts[run][0] if run in firsts else None)
        n1 = counts[counts["step"].isin(w1)].groupby("family")["n"].sum()
        n2 = counts[counts["step"].isin(w2)].groupby("family")["n"].sum()
        total = STEP_SAMPLES * WINDOW
        for i, f in zip(g.index, g["family"]):
            out.append((i, math.log((n2.get(f, 0) + 0.5) / total) - math.log((n1.get(f, 0) + 0.5) / total)))
    idx, vals = zip(*out)
    return pd.Series(vals, index=list(idx))


def spearman(a, b):
    ra, rb = average_ranks(np.asarray(a, float)), average_ranks(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def r_squared(y, cols):
    x = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    return coef[1:], 1 - resid.var() / y.var()


def growth_table(u, n_boot, seed):
    rows = []
    sets = [("core pooled", u[u["group"] == "core"]), ("all seeded pooled", u)]
    sets += [(f"{grp} {j}", g) for (grp, j), g in u.groupby(["group", "judge"], sort=True)]
    for label, g in sets:
        y = g["growth"].to_numpy(float)
        row = {"analysis": label, "runs": g["run"].nunique(), "units": len(g)}
        for c in ("log_r", "log_fpr", "log_p"):
            row[f"rho_{c}"] = spearman(y, g[c])
        for c in ("log_r", "log_fpr", "log_p"):
            row[f"r2_{c}"] = r_squared(y, [g[c]])[1]
        coef, r2 = r_squared(y, [g["log_fpr"], g["log_p"]])
        row.update({"b_log_fpr": coef[0], "b_log_p": coef[1], "r2_both": r2})
        if label.endswith("pooled"):
            rng, boot, sample = np.random.default_rng(seed), [], resampler(g)
            for _ in range(n_boot):
                b = sample(rng)
                boot.append(
                    [spearman(b["growth"], b[c]) for c in ("log_r", "log_fpr", "log_p")]
                    + [spearman(b["growth"], b["log_r"]) - spearman(b["growth"], b[c]) for c in ("log_fpr", "log_p")]
                )
            boot = np.array(boot)
            names = ["rho_log_r", "rho_log_fpr", "rho_log_p", "rho_r_minus_fpr", "rho_r_minus_p"]
            for k, name in enumerate(names):
                lo, hi = np.nanpercentile(boot[:, k], [2.5, 97.5])
                row[f"{name}_ci_low"], row[f"{name}_ci_high"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 3. winner conditional logit

def separating_ratios(g, grid=None):
    """The weights b1 / b2 >= 0 for which b1 log p + b2 log FPR (b2 > 0) ranks the winner first in every
    run with an event, as (smallest, largest) on a grid from 0 to 100; None if no weight does."""
    grid = np.concatenate([[0.0], np.logspace(-2, 2, 801)]) if grid is None else grid
    runs = [(r["log_p"].to_numpy(), r["log_fpr"].to_numpy(), r["cs_event"].to_numpy(bool))
            for _, r in g.groupby("run") if r["cs_event"].any()]
    ok = [lam for lam in grid
          if all((s := lam * lp + lf)[w].min() > s[~w].max() for lp, lf, w in runs)]
    return (min(ok), max(ok)) if ok else None


def winner_logit(u):
    rows = []
    for label, g in (("core", u[u["group"] == "core"]), ("all seeded", u)):
        g = g[g["run"].isin(g.loc[g["cs_event"], "run"])].copy()
        g["cs_time"] = 0                                  # within a run every family is at risk at the first event
        models = {"full": ["log_p", "log_fpr"], "risk score (b1 = b2)": ["log_r"],
                  "FPR alone (b1 = 0)": ["log_fpr"], "propensity alone (b2 = 0)": ["log_p"]}
        fits = {name: fit(risk_sets(g, cov, strata="run"), len(cov)) for name, cov in models.items()}
        _, ll_full, _ = fits["full"]
        sep = separating_ratios(g)
        for name, (beta, ll, finite) in fits.items():
            row = {"runs": label, "runs_with_event": g["run"].nunique(), "model": name, "loglik": ll,
                   "finite": finite, "coefficients": ", ".join(f"{c} {b:.3g}" for c, b in zip(models[name], beta))}
            if name == "full":
                row["separating_b1_over_b2"] = "none" if sep is None else f"{sep[0]:.3g} to {sep[1]:.3g}"
            if name != "full":
                stat = 2 * (ll_full - ll)
                row.update({"lr_stat_vs_full": stat, "p": chi2_1df(stat)})
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default="results/analysis")
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, results = ROOT / args.analysis, ROOT / args.results
    seeding = yaml.safe_load(queue.PROTOCOL.read_text())["seeding"]

    units = pd.read_csv(out / "units.csv")
    u = add_covariates(units[units["group"].isin(["core", "held_out"])].reset_index(drop=True))
    firsts = first_events(u, results)
    u = cause_specific(u, firsts)
    u["growth"] = growth(u, firsts, results)

    km_runs, km_tiers = km_tables(u, seeding)
    cox = cox_table(u, args.boot, args.seed)
    eg = growth_table(u, args.boot, args.seed)
    logit = winner_logit(u)
    for name, table in (("km_runs", km_runs), ("km_tiers", km_tiers), ("cox", cox), ("early_growth", eg),
                        ("winner_logit", logit)):
        table.to_csv(out / f"{name}.csv", index=False)
    write_meta(out, name="meta_secondary.json", stage="secondary_analyses", n_boot=args.boot, seed=args.seed,
               first_events={r: list(v) for r, v in sorted(firsts.items())})

    def fmt(x):
        return f"{x:.3g}"

    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60):
        print("1. Time to a run's first event (Kaplan-Meier, share of runs with an event by step):")
        print(km_runs.to_string(index=False, float_format=fmt))
        print("\nCause-specific time to event by seeding tier (core units):")
        print(km_tiers.to_string(index=False, float_format=fmt))
        print("\nCause-specific Cox model:")
        print(cox.to_string(index=False, float_format=fmt))
        print("\n2. Early growth, Spearman correlations and R squared:")
        cols = ["analysis", "runs", "units", "rho_log_r", "rho_log_fpr", "rho_log_p", "r2_log_r", "r2_log_fpr",
                "r2_log_p", "r2_both", "b_log_fpr", "b_log_p"]
        print(eg[cols].to_string(index=False, float_format=fmt))
        ci = eg[eg["analysis"].str.endswith("pooled")]
        print(
            ci[["analysis"] + [c for c in eg.columns if c.endswith(("_ci_low", "_ci_high"))]].to_string(
                index=False, float_format=fmt
            )
        )
        print("\n3. Winner conditional logit:")
        print(logit.to_string(index=False, float_format=fmt))
    print(f"\nwrote km_runs.csv, km_tiers.csv, cox.csv, early_growth.csv, winner_logit.csv to {display_path(out)}")


if __name__ == "__main__":
    main()
