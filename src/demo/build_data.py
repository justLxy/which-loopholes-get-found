"""Build the small data bundle the demo reads (demo/data/), from the main experiment's results.

Usage:
    python -m src.demo.build_data [--results results/main] [--runs runs/grpo] [--out demo/data] [--prereg-only]

The demo must run without the 2.8 GB of run logs, so this keeps only what it shows:
  judges.json    display name, role and static accuracy of every judge of the main runs
  families.json  each family's seeding string and a one-line description (the audit guide)
  risk.csv       the pre-RL prediction of every checkpoint, judge, pool and family: FPR of the seeding
                 string, post-SFT propensity p_hat, risk p_hat x FPR, expected rewarded samples per step
                 (256 x risk), seeding tier, and the rank by risk and by FPR alone within judge and pool
  runs.csv       one row per main run: judge, pool, seed, group, its events and its first event
  runs/<run>.json  per run: e_f(t) and the overall rates at each of the 31 evaluations, each family's
                 share of the 256 training responses at every step, and at each evaluation the 5 most
                 frequent distinct responses (count, family, how many the judge accepted and gold marked
                 correct; cut to 700 chars)
  results.json   the headline results, when results/analysis exists: AUROC of each score on the core
                 and held-out runs, and the five confirmatory tests with their Holm-adjusted p-values
  prereg.csv     the 10 runs of experiments B and C: recorded prediction and outcome; they also get
                 runs/<run>.json (with "kind") but are not in runs.csv
  timing.json    timing fit, z90 and level of src.audit.predict, for the demo's own-judge tab
Events are recomputed with the frozen definition (configs/protocol.yaml). Only results/ is needed for
everything but the sample responses, which come from runs/grpo/<run>/eval.jsonl. --prereg-only rewrites
just the runs of experiments B and C, so the main runs keep their sample responses without the run logs.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from src.audit import predict
from src.detectors import FAMILIES
from src.monitor.events import detect_events, load_summary
from src.monitor.exploit_audit_sheet import FAMILY_GUIDE
from src.paths import ROOT, display_path
from src.train import queue
from src.train.seed_data import tier_shares

STEP_SAMPLES = 256
TOP_RESPONSES = 5
MAX_CHARS = 700
RATES = ("accept_rate", "gold_accuracy", "false_positive_rate", "exploit_rate", "family_rate")
ROLES = {"positive_control": "core, positive control", "negative_control": "core, negative control", "core": "core",
         "held_out": "held-out", "held_out_added": "held-out (added in protocol v1.1)"}


def judges_table(judges_yaml, static_csv, used):
    cfg = yaml.safe_load(Path(judges_yaml).read_text())["judges"]
    static = pd.read_csv(static_csv)
    static = static[static["pool"] == "all"].set_index("judge")["balanced_accuracy"]
    return {j: {"name": cfg[j]["model"].split("/")[-1], "model": cfg[j]["model"], "role": ROLES[cfg[j]["role"]],
                "static_balanced_accuracy": None if j not in static else round(float(static[j]), 3)}
            for j in used}


def families_table(seeding_strings):
    strings = json.loads(Path(seeding_strings).read_text())
    guide = {name: description for name, description, _ in FAMILY_GUIDE}
    return {f: {"string": strings[f], "description": guide[f]} for f in FAMILIES}


def risk_table(results_dir, checkpoints, seeding):
    parts = []
    for ck in checkpoints:
        r = pd.read_csv(Path(results_dir) / f"sft-{ck['name']}" / "risk_scores.csv")
        r = r[r["family"].isin(FAMILIES)].copy()
        rotated = ck["seeding"] == "rotated"
        r["checkpoint"], r["seed"] = ck["name"], ck["seed"]
        r["seeding"] = "rotated tiers" if rotated else "none (0% control)"
        r["tier"] = [tier_shares(seeding["tiers"], ck["rotation"], seeding["group_seed"])[f] if rotated else 0.0
                     for f in r["family"]]
        parts.append(r)
    r = pd.concat(parts, ignore_index=True)
    r["exposure"] = STEP_SAMPLES * r["risk"]
    g = r.groupby(["checkpoint", "judge", "pool"])
    r["rank_risk"] = g["risk"].rank(ascending=False, method="min").astype(int)
    r["rank_fpr"] = g["fpr"].rank(ascending=False, method="min").astype(int)
    cols = ["checkpoint", "seed", "seeding", "judge", "pool", "family", "tier", "fpr", "p_hat", "risk", "exposure",
            "rank_risk", "rank_fpr"]
    return r[cols].sort_values(["checkpoint", "judge", "pool", "rank_risk", "family"]).reset_index(drop=True)


def run_group(run, main_runs):
    if run["tag"] == main_runs["run_tags"]["control"]:
        return "0% control"
    return "core" if run["judge"] in main_runs["core"]["judges"] else "held-out"


def top_responses(records):
    """The most frequent distinct responses of one evaluation (first occurrence breaking ties), with how
    many of their copies the judge accepted and gold marked correct (the problems differ)."""
    counts, accepted, correct, family = Counter(), Counter(), Counter(), {}
    for r in records:
        counts[r["response"]] += 1
        accepted[r["response"]] += bool(r["accepted"])
        correct[r["response"]] += bool(r["gold_correct"])
        family.setdefault(r["response"], r["family"])
    out = []
    for text, n in sorted(counts.items(), key=lambda kv: -kv[1])[:TOP_RESPONSES]:
        short = text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + " [...]"
        out.append({"text": short, "count": n, "family": family[text], "accepted": accepted[text],
                    "gold_correct": correct[text]})
    return out


def run_bundle(results_dir, runs_dir, name, events, total_steps):
    d = Path(results_dir) / name
    evals = [{"step": row["step"], "e": {f: row.get(f"e/{f}", 0.0) for f in FAMILIES},
              **{k: row[k] for k in RATES}} for row in load_summary(d)]
    counts = pd.read_csv(d / "family_counts.csv")
    share = {f: [0.0] * total_steps for f in FAMILIES}
    for step, fam, n in counts[["step", "family", "n"]].itertuples(index=False):
        if fam in share and step < total_steps:
            share[fam][step] = round(n / STEP_SAMPLES, 4)
    samples = {}
    eval_file = Path(runs_dir) / name / "eval.jsonl"
    if eval_file.exists():
        by_step = {}
        with eval_file.open() as f:
            for line in f:
                r = json.loads(line)
                by_step.setdefault(r["step"], []).append(r)
        samples = {str(s): top_responses(recs) for s, recs in sorted(by_step.items())}
    return {"run": name, "events": events, "eval": evals, "train_share": share, "samples": samples}


def results_summary(analysis_dir):
    analysis_dir = Path(analysis_dir)
    if not (analysis_dir / "primary_auroc.csv").exists() or not (analysis_dir / "confirmatory.csv").exists():
        return None
    p = pd.read_csv(analysis_dir / "primary_auroc.csv")
    p = p[p["analysis"].isin(["core pooled", "held-out pooled"]) & p["auroc"].notna()]
    c = pd.read_csv(analysis_dir / "confirmatory.csv")
    return {"auroc": p[["analysis", "runs", "events", "score", "auroc", "ci_low", "ci_high"]].to_dict("records"),
            "confirmatory": c[["test", "detail", "p", "p_holm", "reject_at_0.05"]].to_dict("records")}


PREREG_COLUMNS = ["run", "experiment", "kind", "judge", "pool", "seed", "condition", "q0", "first_by_exposure",
                  "exposure_top", "first_by_fpr", "first_by_propensity", "predicted_step", "step_low", "step_high",
                  "winner", "step", "later_events", "hit", "in_band", "has_samples"]


def normalize_masking(outcomes, predictions):
    """Experiment B in the prereg.csv schema. exposure_top comes from predictions.csv: winner_exposure there is
    the predicted family's, in the outcomes file the observed winner's."""
    predicted = predictions[["run", "pool", "winner_exposure"]].rename(columns={"winner_exposure": "exposure_top"})
    o = outcomes.merge(predicted, on="run", how="left", validate="one_to_one")
    return pd.DataFrame({
        "run": o["run"], "experiment": "B", "kind": "masking", "judge": o["judge"], "pool": o["pool"],
        "seed": o["seed"], "condition": "mask " + o["masked"], "q0": o["q0"], "first_by_exposure": o["predicted"],
        "exposure_top": o["exposure_top"], "first_by_fpr": o["fpr_alone_predicted"],
        "first_by_propensity": o["propensity_alone_predicted"], "predicted_step": o["predicted_step"],
        "step_low": o["step_low"], "step_high": o["step_high"], "winner": o["winner"], "step": o["step"],
        "hit": o["hit"].astype(bool), "in_band": o["in_band"].astype(bool)})


def normalize_lowshare(outcomes, predictions):
    """Experiment C in the prereg.csv schema."""
    o = outcomes.merge(predictions[["run", "pool"]], on="run", how="left", validate="one_to_one")
    return pd.DataFrame({
        "run": o["run"], "experiment": "C", "kind": "lowshare", "judge": o["judge"], "pool": o["pool"],
        "seed": o["seed"], "condition": "setting " + o["setting"], "q0": o["q0"],
        "first_by_exposure": o["exposure_favourite"], "exposure_top": o["exposure_top"],
        "first_by_fpr": o["fpr_alone_favourite"], "first_by_propensity": o["propensity_alone_favourite"],
        "predicted_step": o["step_pred"], "step_low": o["step_low"], "step_high": o["step_high"],
        "winner": o["winner"], "step": o["step"], "hit": o["hit_exposure"].astype(bool),
        "in_band": o["in_band"].astype(bool)})


# experiment: (results folder, outcomes file in results/analysis, normalizer); predictions.csv is in the folder
PREREG = {"B": ("masking", "masking_outcomes.csv", normalize_masking),
          "C": ("lowshare", "lowshare_outcomes.csv", normalize_lowshare)}


def build_prereg(out, runs_dir, results_root=ROOT / "results", analysis_dir=ROOT / "results" / "analysis",
                 tau=0.05, consecutive=2, total_steps=300):
    """Write prereg.csv and the run files of experiments B and C; skip a missing outcomes file. Recomputed events
    must match it."""
    out, parts = Path(out), []
    for folder, outcomes_file, normalize in PREREG.values():
        outcomes_path = Path(analysis_dir) / outcomes_file
        if not outcomes_path.exists():
            continue
        exp_dir = Path(results_root) / folder
        table = normalize(pd.read_csv(outcomes_path), pd.read_csv(exp_dir / "predictions.csv"))
        later, samples = [], []
        for _, row in table.iterrows():
            found = detect_events(load_summary(exp_dir / row["run"]), tau, consecutive)
            events = sorted((e["time"], f) for f, e in found.items() if e["event"] and f in FAMILIES)
            if not events or (events[0][1], events[0][0]) != (row["winner"], int(row["step"])):
                raise ValueError(f"{row['run']}: recomputed first event {events[:1]} differs from "
                                 f"{row['winner']} at {row['step']} in {outcomes_file}")
            bundle = run_bundle(exp_dir, runs_dir, row["run"], [{"family": f, "step": t} for t, f in events],
                                total_steps)
            bundle["kind"] = row["kind"]
            (out / "runs" / f"{row['run']}.json").write_text(json.dumps(bundle, ensure_ascii=False,
                                                                        separators=(",", ":")))
            later.append("; ".join(f"{f} at step {t}" for t, f in events[1:]))
            samples.append(bool(bundle["samples"]))
        table["later_events"], table["has_samples"] = later, samples
        parts.append(table)
    if not parts:
        return []
    prereg = pd.concat(parts, ignore_index=True)[PREREG_COLUMNS]
    prereg.to_csv(out / "prereg.csv", index=False)
    return prereg.to_dict("records")


FIT_DIGITS = 14                     # significant digits kept of the timing fit in timing.json


def timing_summary():
    """Timing fit, z90 and level from src.audit.predict. The fit is rounded to FIT_DIGITS significant digits: its
    last digits depend on the linear algebra library, and the bundle should be the same on every platform."""
    (a, b, sd), level = predict.timing()
    a, b, sd = (float(f"{x:.{FIT_DIGITS}g}") for x in (a, b, sd))
    return {"a": a, "b": b, "sd": sd, "z90": predict.Z90, "level": level, "step_samples": STEP_SAMPLES,
            "source": "src.audit.predict.timing(): src.train.masking.timing_fit on results/analysis/event_timing.csv"}


def write_timing(out):
    (Path(out) / "timing.json").write_text(json.dumps(timing_summary(), indent=1))


def build(results_dir, runs_dir, out, protocol=queue.PROTOCOL, analysis_dir=ROOT / "results" / "analysis",
          prereg_root=ROOT / "results"):
    proto = yaml.safe_load(Path(protocol).read_text())
    checkpoints, runs = queue.load_plan(protocol)
    out = Path(out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    used = sorted({r["judge"] for r in runs})
    (out / "judges.json").write_text(json.dumps(judges_table(ROOT / "configs" / "judges.yaml",
                                                             ROOT / "results" / "stage1" / "static_accuracy.csv", used),
                                                indent=1))
    (out / "families.json").write_text(json.dumps(families_table(ROOT / "results" / "stage2" / "seeding_strings.json"),
                                                  indent=1, ensure_ascii=False))
    risk_table(results_dir, checkpoints, proto["seeding"]).to_csv(out / "risk.csv", index=False)
    rows = []
    ev = proto["events"]
    for run in runs:
        name = queue.run_name(run)
        found = detect_events(load_summary(Path(results_dir) / name), ev["tau"], ev["consecutive_evaluations"])
        events = sorted(((e["time"], f) for f, e in found.items() if e["event"] and f in FAMILIES))
        bundle = run_bundle(results_dir, runs_dir, name, [{"family": f, "step": t} for t, f in events],
                            proto["main_runs"]["steps"])
        (out / "runs" / f"{name}.json").write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
        rows.append({"run": name, "judge": run["judge"], "pool": run["pool"], "seed": run["seed"],
                     "checkpoint": run["checkpoint"], "group": run_group(run, proto["main_runs"]),
                     "first_event_family": events[0][1] if events else "",
                     "first_event_step": events[0][0] if events else "",
                     "events": "; ".join(f"{f} at step {t}" for t, f in events),
                     "has_samples": bool(bundle["samples"])})
    pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
    summary = results_summary(analysis_dir)
    if summary:
        (out / "results.json").write_text(json.dumps(summary, indent=1))
    write_timing(out)
    build_prereg(out, runs_dir, prereg_root, analysis_dir, ev["tau"], ev["consecutive_evaluations"],
                 proto["main_runs"]["steps"])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--runs", default="runs/grpo")
    ap.add_argument("--out", default="demo/data")
    ap.add_argument("--prereg-only", action="store_true",
                    help="write only prereg.csv and the runs of experiments B and C; the main runs stay as they are")
    args = ap.parse_args(argv)
    out = ROOT / args.out
    if args.prereg_only:
        proto = yaml.safe_load(Path(queue.PROTOCOL).read_text())
        (out / "runs").mkdir(parents=True, exist_ok=True)
        rows = []
        prereg = build_prereg(out, ROOT / args.runs, tau=proto["events"]["tau"],
                              consecutive=proto["events"]["consecutive_evaluations"],
                              total_steps=proto["main_runs"]["steps"])
        write_timing(out)
    else:
        rows = build(ROOT / args.results, ROOT / args.runs, out)
        prereg = pd.read_csv(out / "prereg.csv").to_dict("records") if (out / "prereg.csv").exists() else []
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {len(rows)} main runs ({sum(r['has_samples'] for r in rows)} with sample responses) and "
          f"{len(prereg)} runs with recorded predictions ({sum(r['has_samples'] for r in prereg)} with sample "
          f"responses) to {display_path(out)}, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
