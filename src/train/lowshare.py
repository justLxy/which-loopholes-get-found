"""Experiment C: two seeded families at a total SFT share of about 3% (configs/lowshare.yaml).

Usage, from the repository root (the full sequence for a new machine is in docs/experiment_c.md):
    python -m src.train.lowshare --checkpoints   # GPU: build, measure and check the checkpoints (fallback if needed)
    python -m src.train.lowshare --predict       # CPU: check again from the copied files and write the predictions
    python -m src.train.lowshare --dry-run       # GPU: the plan and every check before training, run nothing
    python -m src.train.lowshare                 # GPU: the runs, resumable; rerun the same command to resume
    python -m src.train.lowshare --pack-only     # copy the result files of finished runs again

Checkpoints. For each setting and seed s: the seeding set (only the setting's two families, --shares, data seed s,
from the main experiment's clean pool), SFT with seed s, the sampled propensity on the hard pool (16 x 250 audit
samples, Jeffreys-smoothed, with the frozen single-string FPRs) and q0, the acceptance of these 4,000 samples by the
setting's judge (src.audit.q0). The files go to results/lowshare/sft-<checkpoint>/.

Pre-run check. On the measured values: exposure 256 p_hat FPR of all 14 families, the favourites of exposure, of
FPR alone and of propensity alone, and the expected-advantage score p_hat (FPR - q0) / sqrt(q0 (1 - q0)). A
checkpoint passes when the exposure favourite is a seeded family, differs from the favourite of the setting's
baseline, has an exposure of at least min_top_exposure and leads the runner-up by min_margin, the expected-advantage
score ranks it first too, in P settings the propensity favourite leads it in p_hat by min_propensity_ratio, and in at
least min_stability of the bootstrap replicates (audit problems resampled for p_hat and q0, FPR redrawn from its
binomial) the exposure favourite stays the same and still differs from the baseline favourite. A checkpoint that
fails is replaced once by the setting's fallback shares (checkpoint name with suffix b); a fallback that fails is not
trained. Every checkpoint and its check are kept in results/lowshare/checks.csv.

Predictions. For each checkpoint that passed, one run: the three favourites, the expected-advantage favourite, the
margins and stability, and the event step predicted from the first events of the main runs with its 90% band
(src.train.masking.timing_fit). Training starts only when the predictions file is committed and unmodified, matches a
fresh prediction from the committed files, and every checkpoint on this machine has the committed propensity.
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
from src.detectors import FAMILIES
from src.paths import ROOT, display_path
from src.train import masking, queue

PLAN = ROOT / "configs" / "lowshare.yaml"
JEFFREYS = 0.5
CLASSES = len(FAMILIES) + 1          # 14 families and "none", as in src.audit.likelihood.sampled_propensity
CHECKPOINT_FILES = {"data_summary.json": "data_summary.json", "propensity/propensity_sampled.csv": "propensity_sampled.csv",
                    "propensity/risk_scores.csv": "risk_scores.csv", "q0/judged.csv": "judged.csv", "q0/q0.json": "q0.json"}


# ----------------------------------------------------------------------------- plan

def load_plan(path=PLAN):
    """(cfg, checkpoints): one primary checkpoint per setting and seed, in run order (seed by seed)."""
    cfg = yaml.safe_load(Path(path).read_text())
    return cfg, plan_checkpoints(cfg)


def plan_checkpoints(cfg):
    cks = []
    for seed in cfg["seeds"]:
        for setting, s in cfg["settings"].items():
            if s["baseline"] not in ("fpr", "propensity"):
                raise ValueError(f"setting {setting}: baseline must be fpr or propensity")
            name = cfg["checkpoint"].replace("<setting>", setting).replace("<seed>", str(seed))
            cks.append({"setting": setting, "seed": seed, "name": name, "judge": s["judge"], "pool": cfg["pool"],
                        "baseline": s["baseline"], "shares": dict(s["shares"]), "fallback": False,
                        "tag": cfg["tag"].replace("<setting>", setting)})
    return cks


def fallback_of(cfg, ck):
    return {**ck, "name": ck["name"] + "b", "shares": dict(cfg["settings"][ck["setting"]]["fallback"]), "fallback": True}


def run_of(ck):
    return {"judge": ck["judge"], "pool": ck["pool"], "seed": ck["seed"], "tag": ck["tag"], "checkpoint": ck["name"]}


def results_dir(cfg, root=ROOT):
    return root / cfg["results"]


def checkpoint_results(cfg, name, root=ROOT):
    return results_dir(cfg, root) / f"sft-{name}"


# ----------------------------------------------------------------------------- checkpoints (GPU)

def checkpoint_steps(cfg, ck):
    """(label, output that marks it done, command) for one checkpoint."""
    d = f"runs/sft/{ck['name']}"
    py = [sys.executable, "-m"]
    shares = [f"{f}={v}" for f, v in ck["shares"].items()]
    return [
        ("seeding set", f"{d}/data.jsonl",
         py + ["src.train.seed_data", "--name", ck["name"], "--shares", *shares, "--data-seed", str(ck["seed"])]),
        ("SFT", f"{d}/model/config.json", py + ["src.train.sft", "--name", ck["name"], "--seed", str(ck["seed"])]),
        ("sampled propensity", f"{d}/propensity/risk_scores.csv",
         py + ["src.audit.sampled_propensity", "--policy", f"{d}/model", "--run-dir", f"{d}/propensity",
               "--fpr", cfg["string_fpr"], "--pools", ck["pool"]]),
        ("q0", f"{d}/q0/q0.json",
         py + ["src.audit.q0", "--propensity-dir", f"{d}/propensity", "--judge", ck["judge"], "--pool", ck["pool"],
               "--out", f"{d}/q0"]),
    ]


def copy_checkpoint(cfg, ck, root=ROOT):
    src, dst = root / "runs" / "sft" / ck["name"], checkpoint_results(cfg, ck["name"], root)
    dst.mkdir(parents=True, exist_ok=True)
    for rel, name in CHECKPOINT_FILES.items():
        (dst / name).write_bytes((src / rel).read_bytes())


def build(cfg, ck, log, root=ROOT):
    for label, out, cmd in checkpoint_steps(cfg, ck):
        if (root / out).exists():
            continue
        log(f"checkpoint {ck['name']}: {label}")
        if queue.execute(cmd, log) != 0 or not (root / out).exists():
            raise SystemExit(f"checkpoint {ck['name']} failed at {label}; see {display_path(log.path)}")
    copy_checkpoint(cfg, ck, root)


def build_checkpoints(cfg, cks, root=ROOT):
    """Build every primary checkpoint, check it, and build and check its fallback when it fails."""
    clean = root / "runs" / "sft" / "clean_pool.jsonl"
    if not clean.exists():
        raise SystemExit(f"{display_path(clean)} is missing: copy the main experiment's clean pool (clean_pool.jsonl "
                         "and clean_pool.json) there first")
    log = queue.Log(root / "runs" / "lowshare" / "queue.log")
    for ck in cks:
        build(cfg, ck, log, root)
        row = check_checkpoint(cfg, ck, root)
        log(describe_check(row))
        if not row["passed"]:
            fb = fallback_of(cfg, ck)
            log(f"checkpoint {ck['name']} failed the pre-run check; building the fallback {fb['name']}")
            build(cfg, fb, log, root)
            log(describe_check(check_checkpoint(cfg, fb, root)))
    table = check_table(cfg, cks, root)
    table.to_csv(results_dir(cfg, root) / "checks.csv", index=False)
    log(f"wrote {display_path(results_dir(cfg, root) / 'checks.csv')}")


# ----------------------------------------------------------------------------- pre-run check (CPU)

def string_fpr(cfg, judge, root=ROOT):
    """FPR, accepted and n of the 14 seeding strings for one judge, in FAMILIES order."""
    f = pd.read_csv(root / cfg["string_fpr"])
    f = f[f["judge"] == judge].set_index("family").loc[list(FAMILIES)]
    return f[["fpr", "accepted", "n"]]


def per_problem(judged):
    """Per audit problem: responses, responses in each family (FAMILIES order) and accepted responses."""
    fam = pd.crosstab(judged["uid"], judged["family"].fillna(""))
    counts = fam.reindex(columns=list(FAMILIES), fill_value=0)
    n = judged.groupby("uid").size().reindex(counts.index)
    acc = judged.groupby("uid")["accepted"].sum().reindex(counts.index)
    return counts.to_numpy(float), n.to_numpy(float), acc.to_numpy(float)


def p_hat(counts, n):
    return (counts + JEFFREYS) / (n + JEFFREYS * CLASSES)


def scores(p, fpr, q0):
    """exposure, expected-advantage score; arrays in FAMILIES order."""
    return STEP_SAMPLES * p * fpr, p * (fpr - q0) / math.sqrt(q0 * (1 - q0))


def favourite(values):
    """Families sharing the largest value, joined by '+' when tied."""
    values = np.asarray(values, float)
    return "+".join(sorted(f for f, v in zip(FAMILIES, values) if v == values.max()))


def check_values(cfg, ck, judged, fpr):
    """The point estimates of the pre-run check."""
    counts, n, acc = per_problem(judged)
    p = p_hat(counts.sum(0), n.sum())
    q0 = acc.sum() / n.sum()
    f = fpr["fpr"].to_numpy(float)
    exposure, ea = scores(p, f, q0)
    order = np.argsort(-exposure, kind="stable")
    top, second = order[0], order[1]
    fav = {"exposure": favourite(exposure), "fpr": favourite(f), "propensity": favourite(p), "ea": favourite(ea)}
    seeded = list(ck["shares"])
    row = {"setting": ck["setting"], "seed": ck["seed"], "checkpoint": ck["name"], "fallback": ck["fallback"],
           "judge": ck["judge"], "pool": ck["pool"], "baseline": ck["baseline"],
           "shares": ";".join(f"{k}={v}" for k, v in ck["shares"].items()), "q0": q0, "outputs": int(n.sum()),
           "exposure_favourite": fav["exposure"], "exposure_top": exposure[top],
           "runner_up": FAMILIES[second], "runner_up_exposure": exposure[second],
           "margin": exposure[top] / exposure[second] if exposure[second] > 0 else math.inf,
           "fpr_alone_favourite": fav["fpr"], "propensity_alone_favourite": fav["propensity"],
           "ea_favourite": fav["ea"], "baseline_favourite": fav[ck["baseline"]]}
    for fam in seeded:
        i = FAMILIES.index(fam)
        row.update({f"p_hat:{fam}": p[i], f"fpr:{fam}": f[i], f"exposure:{fam}": exposure[i], f"ea:{fam}": ea[i]})
    if ck["baseline"] == "propensity":
        row["propensity_ratio"] = p[FAMILIES.index(fav["propensity"])] / p[top] if "+" not in fav["propensity"] else 1.0
    return row, (counts, n, acc, fpr)


def bootstrap_stability(ck, data, favourite_exposure, n_boot, seed):
    """Shares of replicates that keep the exposure favourite and its disagreement with the baseline, and in which the
    expected-advantage score ranks it first."""
    counts, n, acc, fpr = data
    rng = np.random.default_rng(seed)
    u = len(n)
    idx = rng.integers(0, u, size=(n_boot, u))
    c = counts[idx].sum(1)                                   # n_boot x 14
    nn, aa = n[idx].sum(1), acc[idx].sum(1)
    p = (c + JEFFREYS) / (nn[:, None] + JEFFREYS * CLASSES)
    q0 = aa / nn
    f = rng.binomial(fpr["n"].to_numpy(int), fpr["fpr"].to_numpy(float), size=(n_boot, len(FAMILIES))) / fpr["n"].to_numpy(float)
    exposure = p * f
    ea = p * (f - q0[:, None]) / np.sqrt(q0 * (1 - q0))[:, None]
    base = f if ck["baseline"] == "fpr" else p
    target = FAMILIES.index(favourite_exposure)
    ex_top, base_top, ea_top = exposure.argmax(1), base.argmax(1), ea.argmax(1)
    stable = float(np.mean((ex_top == target) & (base_top != target)))
    return stable, float(np.mean(ea_top == target))


def check_checkpoint(cfg, ck, root=ROOT):
    d = checkpoint_results(cfg, ck["name"], root)
    judged = pd.read_csv(d / "judged.csv", keep_default_na=False)
    judged["accepted"] = judged["accepted"].astype(str).str.lower().eq("true")
    fpr = string_fpr(cfg, ck["judge"], root)
    row, data = check_values(cfg, ck, judged, fpr)
    risk = pd.read_csv(d / "risk_scores.csv")
    risk = risk[(risk["judge"] == ck["judge"]) & (risk["pool"] == ck["pool"])].set_index("family").loc[list(FAMILIES)]
    counts, n, _, _ = data
    row["propensity_matches_audit"] = bool(np.allclose(risk["p_hat"].to_numpy(float), p_hat(counts.sum(0), n.sum())))
    c = cfg["check"]
    ok = "+" not in row["exposure_favourite"]
    if ok:
        row["stability"], row["ea_stability"] = bootstrap_stability(ck, data, row["exposure_favourite"], c["bootstrap"],
                                                                    c["bootstrap_seed"])
    else:
        row["stability"], row["ea_stability"] = 0.0, 0.0
    tests = {"seeded_favourite": ok and row["exposure_favourite"] in ck["shares"],
             "disagreement": row["exposure_favourite"] != row["baseline_favourite"],
             "top_exposure": row["exposure_top"] >= c["min_top_exposure"],
             "margin": row["margin"] >= c["min_margin"],
             "ea_agrees": row["ea_favourite"] == row["exposure_favourite"],
             "stability": row["stability"] >= c["min_stability"],
             "propensity_matches_audit": row["propensity_matches_audit"]}
    if ck["baseline"] == "propensity":
        tests["propensity_ratio"] = row["propensity_ratio"] >= c["min_propensity_ratio"]
    row.update({f"ok_{k}": bool(v) for k, v in tests.items()})
    row["passed"] = all(tests.values())
    return row


def describe_check(row):
    failed = [k[3:] for k, v in row.items() if k.startswith("ok_") and not v]
    return (f"check {row['checkpoint']}: exposure favourite {row['exposure_favourite']} at {row['exposure_top']:.2f} "
            f"(runner-up {row['runner_up']} {row['runner_up_exposure']:.2f}, margin {row['margin']:.2f}), "
            f"{row['baseline']} favourite {row['baseline_favourite']}, expected-advantage favourite {row['ea_favourite']}, "
            f"q0 {row['q0']:.3f}, stability {row['stability']:.3f} -> "
            + ("PASSED" if row["passed"] else "FAILED: " + ", ".join(failed)))


def check_table(cfg, cks, root=ROOT):
    """Every built checkpoint (primary and fallback) with its check."""
    rows = []
    for ck in cks:
        for c in (ck, fallback_of(cfg, ck)):
            if (checkpoint_results(cfg, c["name"], root) / "judged.csv").exists():
                rows.append(check_checkpoint(cfg, c, root))
    return pd.DataFrame(rows)


def selected(cfg, cks, root=ROOT):
    """The checkpoint each setting and seed trains from: the primary if it passed, else the fallback if it passed."""
    out = []
    for ck in cks:
        for c in (ck, fallback_of(cfg, ck)):
            d = checkpoint_results(cfg, c["name"], root)
            if (d / "judged.csv").exists() and check_checkpoint(cfg, c, root)["passed"]:
                out.append(c)
                break
    return out


# ----------------------------------------------------------------------------- predictions

def prediction_table(cfg, cks, root=ROOT):
    fit = masking.timing_fit(root / cfg["timing_fit"])
    a, b, sd = fit
    rows = []
    for ck in selected(cfg, cks, root):
        chk = check_checkpoint(cfg, ck, root)
        mid = a + b * math.log(chk["exposure_top"])
        rows.append({"run": queue.run_name(run_of(ck)), **{k: v for k, v in chk.items() if not k.startswith("ok_")},
                     "step": math.exp(mid), "step_low": math.exp(mid - masking.Z90 * sd),
                     "step_high": math.exp(mid + masking.Z90 * sd)})
    return pd.DataFrame(rows), fit


def same_predictions(recorded, fresh):
    key = ["run", "checkpoint", "exposure_favourite", "fpr_alone_favourite", "propensity_alone_favourite", "ea_favourite"]
    return len(recorded) == len(fresh) and recorded[key].astype(str).reset_index(drop=True).equals(
        fresh[key].astype(str).reset_index(drop=True)) and np.allclose(recorded["exposure_top"], fresh["exposure_top"])


def same_propensity(cfg, ck, root=ROOT):
    """True when the checkpoint on this machine has the propensity of the committed files."""
    here = root / "runs" / "sft" / ck["name"] / "propensity" / "risk_scores.csv"
    committed = checkpoint_results(cfg, ck["name"], root) / "risk_scores.csv"
    if not (here.exists() and committed.exists()):
        return False
    key = ["judge", "pool", "family"]
    m = pd.read_csv(here)[key + ["p_hat"]].merge(pd.read_csv(committed)[key + ["p_hat"]], on=key,
                                                  suffixes=("_here", "_committed"))
    return len(m) > 0 and np.allclose(m["p_hat_here"], m["p_hat_committed"])


def planned_runs(cfg, root=ROOT):
    """(checkpoint, run) for every row of the committed predictions, in their order."""
    path = root / cfg["predictions"]
    by_name = {}
    for ck in plan_checkpoints(cfg):
        by_name[ck["name"]] = ck
        fb = fallback_of(cfg, ck)
        by_name[fb["name"]] = fb
    return [(by_name[n], run_of(by_name[n])) for n in pd.read_csv(path)["checkpoint"]]


def checks(cfg, cks, root=ROOT):
    """(label, passed) for everything that must hold before training."""
    path = root / cfg["predictions"]
    out = [("predictions present", path.exists())]
    if not path.exists():
        return out
    out.append(("predictions committed and unmodified", masking.committed_unmodified(path, root)))
    try:
        fresh, _ = prediction_table(cfg, cks, root)
        out.append(("predictions match the committed checkpoint files", same_predictions(pd.read_csv(path), fresh)))
    except (FileNotFoundError, SystemExit):
        out.append(("predictions match the committed checkpoint files", False))
    for ck, _ in planned_runs(cfg, root):
        model = root / "runs" / "sft" / ck["name"] / "model" / "config.json"
        out.append((f"checkpoint {ck['name']} present", model.exists()))
        out.append((f"checkpoint {ck['name']} has the committed propensity", same_propensity(cfg, ck, root)))
    return out


# ----------------------------------------------------------------------------- runs

def pack(cfg, run, root=ROOT):
    src, dst = root / "runs" / "grpo" / queue.run_name(run), results_dir(cfg, root) / queue.run_name(run)
    dst.mkdir(parents=True, exist_ok=True)
    for name in queue.RUN_FILES:
        if (src / name).exists():
            (dst / name).write_bytes((src / name).read_bytes())
    with (dst / "family_counts.csv").open("w") as f:
        f.write("step,family,n,accepted\n")
        for r in queue.family_counts(src):
            f.write(f"{r['step']},{r['family']},{r['n']},{r['accepted']}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--checkpoints", action="store_true", help="build, measure and check the checkpoints")
    mode.add_argument("--predict", action="store_true", help="check the copied files and write the predictions")
    mode.add_argument("--dry-run", action="store_true", help="print the plan and the checks, run nothing")
    mode.add_argument("--pack-only", action="store_true", help="copy the result files of finished runs")
    ap.add_argument("--plan", default=str(PLAN))
    args = ap.parse_args()
    cfg, cks = load_plan(args.plan)

    if args.checkpoints:
        build_checkpoints(cfg, cks)
        print(f"copy {cfg['results']}/ to the machine that commits, then run --predict there")
        return

    if args.predict:
        table = check_table(cfg, cks)
        table.to_csv(results_dir(cfg) / "checks.csv", index=False)
        for _, row in table.iterrows():
            print(describe_check(row))
        pred, (a, b, sd) = prediction_table(cfg, cks)
        out = ROOT / cfg["predictions"]
        pred.to_csv(out, index=False)
        print(f"timing fit over the main-run first events: log step = {a:.3f} {b:+.3f} log exposure, residual sd {sd:.3f}")
        cols = ["run", "checkpoint", "exposure_favourite", "exposure_top", "margin", "fpr_alone_favourite",
                "propensity_alone_favourite", "ea_favourite", "q0", "stability", "step", "step_low", "step_high"]
        with pd.option_context("display.width", 240, "display.max_columns", 30):
            print(pred[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        missing = len(cks) - len(pred)
        print(f"wrote {display_path(out)} with {len(pred)} runs" + (f"; {missing} setting-seed pairs have no checkpoint "
                                                                      "that passed and are not trained" if missing else "")
              + "; commit and push it with the checkpoint files before the first run")
        return

    if args.pack_only:
        done = [r for _, r in planned_runs(cfg) if queue.run_state(r) == "done"]
        for run in done:
            pack(cfg, run)
        print(f"packed {len(done)} finished runs into {cfg['results']}/")
        return

    results = checks(cfg, cks)
    if args.dry_run:
        if (ROOT / cfg["predictions"]).exists():
            for ck, run in planned_runs(cfg):
                print(f"  {queue.run_name(run):56s} from {ck['name']:14s} {queue.run_state(run)}")
        for label, ok in results:
            print(f"  [{'ok' if ok else 'NO'}] {label}")
        return
    failed = [label for label, ok in results if not ok]
    if failed:
        raise SystemExit("not starting; failed checks: " + "; ".join(failed) + " (see docs/experiment_c.md)")

    plan = planned_runs(cfg)
    log = queue.Log(ROOT / "runs" / "lowshare" / "queue.log")
    log(f"low-share queue started: {len(plan)} runs")
    for i, (ck, run) in enumerate(plan, 1):
        name = queue.run_name(run)
        state = queue.run_state(run)
        if state == "done":
            pack(cfg, run)
            continue
        if state == "failed attempt":
            log(f"moved a failed attempt of {name} to {queue.set_aside(run).name}")
        log(f"run {i}/{len(plan)}: {name} from {ck['name']}")
        code = queue.execute(queue.run_command(run), log)
        if queue.run_state(run) == "done":
            subprocess.run([sys.executable, "-m", "src.monitor.events", f"runs/grpo/{name}", "--tau", "0.05"],
                           cwd=ROOT, stdout=subprocess.DEVNULL)
            pack(cfg, run)
            log(f"done {name}")
        else:
            log(f"!! {name} failed (exit {code}); rerun to retry it")
    log(f"low-share queue finished: {sum(queue.run_state(r) == 'done' for _, r in plan)}/{len(plan)} runs done")


if __name__ == "__main__":
    main()
