"""Experiment B: mask one family by rule and see which family wins instead (configs/masking.yaml).

Usage, from the repository root (the full sequence for a new machine is in docs/experiment_b.md):
    python -m src.train.masking --checkpoints   # GPU: rebuild the SFT checkpoints the runs start from
    python -m src.train.masking --predict       # CPU: write the predictions from the committed scores
    python -m src.train.masking --dry-run       # GPU: the plan and every check, run nothing
    python -m src.train.masking                 # GPU: the four runs, resumable; rerun the same command to resume
    python -m src.train.masking --pack-only     # copy the result files of finished runs again

Checkpoints. The checkpoints of the main experiment are gone, so each one the plan needs (main-rot<seed>) is
rebuilt with the steps of src.train.queue: the seeding set (from runs/sft/clean_pool.jsonl, the clean pool of the
main experiment, which must be copied there first so that the set is the same), SFT with the seed, and the sampled
propensity with the seeding-string FPR. The string likelihood, a sensitivity measure, is skipped. Their summary,
propensity and risk scores are copied to results/masking/sft-<checkpoint>/, and the seeding set is compared with
the main checkpoint's (results/main/sft-<checkpoint>/data_summary.json).

Predictions. For each run, from the committed scores of its checkpoint with the masked family's FPR set to zero:
the family exposure ranks first and its exposure 256 p_hat FPR, the runner-up, the predicted event step from the
first events of the main runs (least squares of log step on log exposure, results/analysis/event_timing.csv) with
a 90% band of 1.645 residual standard deviations, no event when the top exposure is below the level of the plan,
and for comparison the family FPR alone and propensity alone rank first. --original predicts from the main
experiment's checkpoints instead (results/masking/predictions_original.csv, for reference only).

Runs. Each run is scripts/run_grpo.sh with the arguments of the main run plus --mask-family and the tag
mask-<family>. Training starts only when the predictions file is committed and unmodified, matches a fresh
prediction from the committed scores, and every checkpoint on this machine has the committed propensity.
Results are packed into results/masking/<run>/ like the main runs.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis.descriptive import STEP_SAMPLES
from src.paths import ROOT, display_path
from src.train import queue

PLAN = ROOT / "configs" / "masking.yaml"
Z90 = 1.645
CHECKPOINT_STEPS = ("seeding set", "SFT", "sampled propensity")
COMPARED = ("n", "clean", "rotation", "strings", "shares", "per_family", "seeded_share", "clean_sampling")


def load_plan(path=PLAN):
    cfg = yaml.safe_load(Path(path).read_text())
    runs = []
    for r in cfg["runs"]:
        ck = cfg["checkpoint"].replace("<seed>", str(r["seed"]))
        runs.append({**r, "checkpoint": ck, "tag": cfg["tag"].replace("<family>", r["mask"])})
    return cfg, runs


def _path(cfg, key, run, root=ROOT):
    return root / cfg[key].replace("<checkpoint>", run["checkpoint"])


def committed_scores(cfg, run, root=ROOT):
    return _path(cfg, "scores", run, root)


def original_scores(cfg, run, root=ROOT):
    return _path(cfg, "original_scores", run, root)


def checkpoint_scores(run, root=ROOT):
    return root / "runs" / "sft" / run["checkpoint"] / "propensity" / "risk_scores.csv"


# ----------------------------------------------------------------------------- checkpoints

def checkpoints_of(runs):
    """The checkpoints the runs need, as src.train.queue describes them."""
    out = {}
    for r in runs:
        out.setdefault(r["checkpoint"], {"name": r["checkpoint"], "seed": r["seed"], "rotation": r["seed"],
                                         "seeding": "rotated"})
    return list(out.values())


def compare_seeding(rebuilt, original):
    """Keys of the seeding-set summaries that differ (empty when the sets match)."""
    return [k for k in COMPARED if k in original and rebuilt.get(k) != original.get(k)]


def build_checkpoints(cfg, runs, root=ROOT):
    clean = root / "runs" / "sft" / "clean_pool.jsonl"
    if not clean.exists():
        raise SystemExit(f"{display_path(clean)} is missing: copy the main experiment's clean pool (clean_pool.jsonl "
                         "and clean_pool.json) there first, or the seeding set will differ")
    log = queue.Log(root / "runs" / "masking" / "queue.log")
    for ck in checkpoints_of(runs):
        for label, out, cmd, _ in queue.checkpoint_steps(ck):
            if label not in CHECKPOINT_STEPS or (root / out).exists():
                continue
            log(f"checkpoint {ck['name']}: {label}")
            if queue.execute(cmd, log) != 0 or not (root / out).exists():
                raise SystemExit(f"checkpoint {ck['name']} failed at {label}; see runs/masking/queue.log")
        src = root / "runs" / "sft" / ck["name"]
        dst = root / Path(cfg["scores"].replace("<checkpoint>", ck["name"])).parent
        dst.mkdir(parents=True, exist_ok=True)
        for rel in ("data_summary.json", "propensity/propensity_sampled.csv", "propensity/risk_scores.csv"):
            (dst / Path(rel).name).write_bytes((src / rel).read_bytes())
        original = root / "results" / "main" / f"sft-{ck['name']}" / "data_summary.json"
        diff = compare_seeding(json.loads((src / "data_summary.json").read_text()), json.loads(original.read_text()))
        log(f"checkpoint {ck['name']} ready; seeding set {'matches the main checkpoint' if not diff else 'DIFFERS in ' + ', '.join(diff)}")
        a = pd.read_csv(dst / "propensity_sampled.csv")
        b = pd.read_csv(root / "results" / "main" / f"sft-{ck['name']}" / "propensity_sampled.csv")
        m = a.merge(b, on=["pool", "family"], suffixes=("", "_main"))
        log(f"  propensity against the main checkpoint: largest difference {np.abs(m['p_hat'] - m['p_hat_main']).max():.4f}, "
            f"Spearman {m['p_hat'].rank().corr(m['p_hat_main'].rank()):.3f}")


# ----------------------------------------------------------------------------- predictions

def timing_fit(path):
    """(a, b, residual sd) of log step = a + b log exposure over the first events of the main runs."""
    ev = pd.read_csv(path)
    ev = ev[ev["first"]]
    x, y = np.log(ev["exposure"].to_numpy(float)), np.log(ev["time"].to_numpy(float))
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    return float(a), float(b), float(resid.std(ddof=2))


def top(g, column):
    """Families sharing the largest value of a column, joined by '+' when tied."""
    best = g[column].max()
    return "+".join(sorted(g.loc[g[column] == best, "family"]))


def predict(run, scores, level, fit):
    s = scores[(scores["judge"] == run["judge"]) & (scores["pool"] == run["pool"])].copy()
    if len(s) == 0:
        raise ValueError(f"no risk scores for {run['judge']} on {run['pool']}")
    s.loc[s["family"] == run["mask"], ["fpr", "risk"]] = 0.0
    s["exposure"] = STEP_SAMPLES * s["risk"]
    s = s.sort_values("exposure", ascending=False)
    open_ = s[s["family"] != run["mask"]]
    first, second = s.iloc[0], s.iloc[1]
    a, b, sd = fit
    row = {"run": queue.run_name(run), "judge": run["judge"], "pool": run["pool"], "seed": run["seed"],
           "masked": run["mask"], "winner": first["family"], "winner_exposure": first["exposure"],
           "runner_up": second["family"], "runner_up_exposure": second["exposure"],
           "fpr_alone_winner": top(open_, "fpr"), "propensity_alone_winner": top(open_, "p_hat"),
           "separates_exposure_from_fpr": first["family"] != top(open_, "fpr")}
    if first["exposure"] >= level:
        mid = a + b * math.log(first["exposure"])
        row.update({"event_predicted": True, "step": math.exp(mid), "step_low": math.exp(mid - Z90 * sd),
                    "step_high": math.exp(mid + Z90 * sd)})
    else:
        row.update({"event_predicted": False})
    return row


def prediction_table(cfg, runs, original=False, root=ROOT):
    fit = timing_fit(root / cfg["timing_fit"])
    rows = []
    for run in runs:
        path = original_scores(cfg, run, root) if original else committed_scores(cfg, run, root)
        if not path.exists():
            raise SystemExit(f"{display_path(path)} is missing; build the checkpoints and copy their scores first")
        rows.append({**predict(run, pd.read_csv(path), cfg["level"], fit), "scores": display_path(path)})
    return pd.DataFrame(rows), fit


def predictions_path(cfg, original=False, root=ROOT):
    p = root / cfg["predictions"]
    return p.with_name(p.stem + "_original.csv") if original else p


def same_predictions(recorded, fresh):
    key = ["run", "winner", "fpr_alone_winner", "event_predicted"]
    return recorded[key].astype(str).equals(fresh[key].astype(str)) and \
        np.allclose(recorded["winner_exposure"], fresh["winner_exposure"])


def committed_unmodified(path, root=ROOT):
    """True when the file is tracked by git and has no uncommitted change."""
    rel = str(Path(path).relative_to(root))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=root, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=root).returncode == 0
    return tracked and clean


def same_propensity(cfg, run, root=ROOT):
    """True when the checkpoint on this machine has the propensity of the committed scores."""
    a, b = checkpoint_scores(run, root), committed_scores(cfg, run, root)
    if not (a.exists() and b.exists()):
        return False
    key = ["judge", "pool", "family"]
    m = pd.read_csv(a)[key + ["p_hat"]].merge(pd.read_csv(b)[key + ["p_hat"]], on=key, suffixes=("_here", "_committed"))
    return len(m) > 0 and np.allclose(m["p_hat_here"], m["p_hat_committed"])


# ----------------------------------------------------------------------------- runs

def run_command(run):
    return queue.run_command(run) + ["--mask-family", run["mask"]]


def pack(run, root=ROOT):
    src, dst = root / "runs" / "grpo" / queue.run_name(run), root / "results" / "masking" / queue.run_name(run)
    dst.mkdir(parents=True, exist_ok=True)
    for name in queue.RUN_FILES:
        if (src / name).exists():
            (dst / name).write_bytes((src / name).read_bytes())
    rows = queue.family_counts(src)
    with (dst / "family_counts.csv").open("w") as f:
        f.write("step,family,n,accepted\n")
        for r in rows:
            f.write(f"{r['step']},{r['family']},{r['n']},{r['accepted']}\n")


def checks(cfg, runs, root=ROOT):
    """(label, passed) for everything that must hold before training."""
    path = predictions_path(cfg, root=root)
    out = [("predictions present", path.exists())]
    if path.exists():
        out.append(("predictions committed and unmodified", committed_unmodified(path, root)))
        try:
            fresh, _ = prediction_table(cfg, runs, root=root)
            out.append(("predictions match the committed scores", same_predictions(pd.read_csv(path), fresh)))
        except SystemExit:
            out.append(("predictions match the committed scores", False))
    for ck in checkpoints_of(runs):
        model = root / "runs" / "sft" / ck["name"] / "model" / "config.json"
        out.append((f"checkpoint {ck['name']} present", model.exists()))
        run = next(r for r in runs if r["checkpoint"] == ck["name"])
        out.append((f"checkpoint {ck['name']} has the committed propensity", same_propensity(cfg, run, root)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--checkpoints", action="store_true", help="rebuild the SFT checkpoints and copy their scores")
    mode.add_argument("--predict", action="store_true", help="write the predictions and stop")
    mode.add_argument("--dry-run", action="store_true", help="print the plan and the checks, run nothing")
    mode.add_argument("--pack-only", action="store_true", help="copy the result files of finished runs")
    ap.add_argument("--original", action="store_true",
                    help="with --predict: use the main experiment's checkpoints (reference file only)")
    ap.add_argument("--plan", default=str(PLAN))
    args = ap.parse_args()
    cfg, runs = load_plan(args.plan)

    if args.checkpoints:
        build_checkpoints(cfg, runs)
        print("copy results/masking/sft-* to the machine that commits, then run --predict there")
        return

    if args.predict:
        table, (a, b, sd) = prediction_table(cfg, runs, args.original)
        out = predictions_path(cfg, args.original)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"timing fit over the main-run first events: log step = {a:.3f} {b:+.3f} log exposure, residual sd {sd:.3f}")
        with pd.option_context("display.width", 240, "display.max_columns", 30):
            print(table.drop(columns=["scores"]).to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        if not table["separates_exposure_from_fpr"].all():
            print("note: in some runs exposure and FPR alone now predict the same winner; revisit the plan")
        print(f"wrote {display_path(out)}" + ("" if args.original else "; commit and push it before the first run"))
        return

    if args.pack_only:
        done = [r for r in runs if queue.run_state(r) == "done"]
        for run in done:
            pack(run)
        print(f"packed {len(done)} finished runs into results/masking/")
        return

    results = checks(cfg, runs)
    if args.dry_run:
        for run in runs:
            print(f"  {queue.run_name(run):62s} {queue.run_state(run)}")
        for label, ok in results:
            print(f"  [{'ok' if ok else 'NO'}] {label}")
        return
    failed = [label for label, ok in results if not ok]
    if failed:
        raise SystemExit("not starting; failed checks: " + "; ".join(failed) + " (see docs/experiment_b.md)")

    log = queue.Log(ROOT / "runs" / "masking" / "queue.log")
    log(f"masking queue started: {len(runs)} runs")
    for i, run in enumerate(runs, 1):
        name = queue.run_name(run)
        state = queue.run_state(run)
        if state == "done":
            pack(run)
            continue
        if state == "failed attempt":
            log(f"moved a failed attempt of {name} to {queue.set_aside(run).name}")
        log(f"run {i}/{len(runs)}: {name}")
        code = queue.execute(run_command(run), log)
        if queue.run_state(run) == "done":
            subprocess.run([sys.executable, "-m", "src.monitor.events", f"runs/grpo/{name}", "--tau", "0.05"],
                           cwd=ROOT, stdout=subprocess.DEVNULL)
            pack(run)
            log(f"done {name}")
        else:
            log(f"!! {name} failed (exit {code}); rerun to retry it")
    log(f"masking queue finished: {sum(queue.run_state(r) == 'done' for r in runs)}/{len(runs)} runs done")


if __name__ == "__main__":
    main()
