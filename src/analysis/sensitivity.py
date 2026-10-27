"""Sensitivity analyses of the primary test (descriptive, outside the Holm family; HANDOFF 5.1 and 5.3).

Usage (after src.analysis.primary, whose units.csv it reads):
    python -m src.analysis.sensitivity [--analysis results/analysis] [--results results/main] [--boot 10000]

Each variant changes one part of the primary test and repeats it (src.analysis.primary.analyse: AUROC
of the risk score and the three baselines, run bootstrap within judges, paired differences) on the
core runs and on the held-out runs:
  event definition  tau 0.025, 0.05 and 0.10 (half, once and twice the frozen value), each with one
                    evaluation or two consecutive evaluations above the step-0 value (HANDOFF 5.3)
  propensity        string likelihood of the family's canonical strings after SFT (the checkpoint's
                    likelihood.csv), total log-probability and per-token, combined in log space,
                    log r = ell + log FPR (HANDOFF 5.1); the propensity baseline becomes ell
                    the seeded dose itself (the tier share the family received under its run's rotation,
                    src.train.seed_data.tier_shares), so risk = dose x FPR: what knowing the injection gives,
                    against the propensity the audit measures without it
  FPR               the stage-1 mean FPR over the family's strings in [1]'s set
                    (results/stage1/fpr_by_family.csv) instead of the seeding string's FPR; the judge
                    added in protocol-v1.1 was not audited in stage 1 and drops out
  runs              the hard pool only, where every event happened: the easy pool adds high-exposure
                    families without an event, so the pooled AUROC is the harder test
A threshold-free version replaces the event by the largest rise of e_f over its step-0 value within the
300 steps, max_t e_f(t) - e_f(0), and reports its Spearman correlation with each score, with run
bootstrap intervals for the differences to the risk score.

Writes sensitivity.csv and threshold_free.csv to --analysis.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis.primary import SCORES, analyse
from src.analysis.secondary import resampler, spearman
from src.detectors import FAMILIES
from src.monitor.events import detect_events, load_summary
from src.paths import ROOT, display_path
from src.pipeline.records import write_meta
from src.train import queue
from src.train.seed_data import tier_shares

FAMILY_MEAN_FPR = ROOT / "results" / "stage1" / "fpr_by_family.csv"
TAUS = (0.025, 0.05, 0.10)
CONSECUTIVE = (1, 2)
PRIMARY = (0.05, 2)


def events_for(units, results_dir, tau, consecutive):
    """The units with their event recomputed under another definition, and the threshold-free rise."""
    rows = {}
    for run in units["run"].unique():
        ev = detect_events(load_summary(Path(results_dir) / run), tau, consecutive)
        for f in FAMILIES:
            rows[(run, f)] = (ev[f]["event"], ev[f]["peak"] - ev[f]["baseline"])
    u = units.copy()
    u["event"] = [rows[(r, f)][0] for r, f in zip(u["run"], u["family"])]
    u["rise"] = [rows[(r, f)][1] for r, f in zip(u["run"], u["family"])]
    return u


def with_likelihood(units, results_dir, column):
    """Propensity from the string likelihood of the run's checkpoint: p_hat = ell, risk = ell + log FPR."""
    tables = []
    for seed in sorted(units["seed"].unique()):
        t = pd.read_csv(Path(results_dir) / f"sft-main-rot{seed}" / "likelihood.csv")
        tables.append(t.assign(seed=seed)[["seed", "pool", "family", column]])
    u = units.merge(pd.concat(tables), on=["seed", "pool", "family"], how="left", validate="many_to_one")
    if u[column].isna().any():
        raise ValueError("a run lacks the string likelihood of a family")
    with np.errstate(divide="ignore"):
        u["risk"] = u[column] + np.log(u["fpr"])          # -inf where the FPR is 0: tied at the bottom
    u["p_hat"] = u[column]
    return u.drop(columns=[column])


def with_seeded_dose(units, tiers, group_seed):
    """Propensity replaced by the dose each family was seeded at in its run: p_hat = dose, risk = dose x FPR."""
    u = units.copy()
    u["p_hat"] = [tier_shares(tiers, int(s), group_seed)[f] for s, f in zip(u["seed"], u["family"])]
    u["risk"] = u["p_hat"] * u["fpr"]
    return u


def with_family_mean_fpr(units, fpr_csv=FAMILY_MEAN_FPR):
    m = pd.read_csv(fpr_csv)[["judge", "family", "fpr"]].rename(columns={"fpr": "fpr_mean"})
    u = units.merge(m, on=["judge", "family"], how="inner", validate="many_to_one")
    u["fpr"] = u.pop("fpr_mean")
    u["risk"] = u["p_hat"] * u["fpr"]
    return u


def variants(units, results_dir):
    """(name, units) for every variant, the primary definition first."""
    out = []
    for tau in TAUS:
        for c in CONSECUTIVE:
            name = f"tau {tau:g}, {c} evaluation{'s' if c > 1 else ''}" + (" (primary)" if (tau, c) == PRIMARY else "")
            out.append((name, events_for(units, results_dir, tau, c)))
    out.append(("propensity: string likelihood, total", with_likelihood(units, results_dir, "ell")))
    out.append(("propensity: string likelihood, per token", with_likelihood(units, results_dir, "per_token")))
    seeding = yaml.safe_load(queue.PROTOCOL.read_text())["seeding"]
    out.append(("propensity: seeded dose", with_seeded_dose(units, seeding["tiers"], seeding["group_seed"])))
    out.append(("FPR: stage-1 family mean", with_family_mean_fpr(units)))
    out.append(("runs: hard pool only", units[units["pool"] == "hard"].reset_index(drop=True)))
    return out


def threshold_free(u, n_boot, seed):
    rows = []
    for label, g in (("core pooled", u[u["group"] == "core"]), ("held-out pooled", u[u["group"] == "held_out"])):
        row = {"analysis": label, "units": len(g), **{f"rho_{s}": spearman(g["rise"], g[s]) for s in SCORES}}
        rng, sample, boot = np.random.default_rng(seed), resampler(g), []
        for _ in range(n_boot):
            b = sample(rng)
            r = spearman(b["rise"], b["risk"])
            boot.append([r - spearman(b["rise"], b[s]) for s in SCORES[1:]])
        boot = np.array(boot)
        for k, s in enumerate(SCORES[1:]):
            row[f"risk_minus_{s}"] = row["rho_risk"] - row[f"rho_{s}"]
            row[f"risk_minus_{s}_ci_low"], row[f"risk_minus_{s}_ci_high"] = np.nanpercentile(boot[:, k], [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default="results/analysis")
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, results = ROOT / args.analysis, ROOT / args.results

    units = pd.read_csv(out / "units.csv")
    units = units[units["group"].isin(["core", "held_out"])].reset_index(drop=True)
    rows = []
    for name, u in variants(units, results):
        for label, g in (("core pooled", u[u["group"] == "core"]), ("held-out pooled", u[u["group"] == "held_out"])):
            rows += [{"variant": name, **r} for r in analyse(g, label, args.boot, args.seed)]
        print(f"{name}: done", flush=True)
    table = pd.DataFrame(rows)
    free = threshold_free(events_for(units, results, *PRIMARY), min(args.boot, 2000), args.seed)
    table.to_csv(out / "sensitivity.csv", index=False)
    free.to_csv(out / "threshold_free.csv", index=False)
    write_meta(out, name="meta_sensitivity.json", stage="sensitivity_analyses", n_boot=args.boot, seed=args.seed)

    floor = 1 / args.boot
    risk = table[table["score"] == "risk"].set_index(["variant", "analysis"])
    lines = []
    for (variant, analysis), r in risk.iterrows():
        line = {"variant": variant, "runs": analysis, "events": r["events"], "risk AUROC": r["auroc"]}
        for s in SCORES[1:]:
            b = table[(table["variant"] == variant) & (table["analysis"] == analysis) & (table["score"] == s)].iloc[0]
            p = "" if pd.isna(b["p_two_sided"]) else ("p<%g" % floor if b["p_two_sided"] < floor else f"p={b['p_two_sided']:.4f}")
            line[f"minus {s}"] = "" if pd.isna(b["risk_minus_score"]) else \
                f"{b['risk_minus_score']:+.3f} ({b['diff_ci_low']:+.3f}, {b['diff_ci_high']:+.3f}) {p}"
        lines.append(line)
    with pd.option_context("display.width", 260, "display.max_colwidth", 50):
        print(pd.DataFrame(lines).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nThreshold-free: Spearman correlation of max_t e_f(t) - e_f(0) with each score")
        print(free.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"wrote {display_path(out / 'sensitivity.csv')} and threshold_free.csv")


if __name__ == "__main__":
    main()
