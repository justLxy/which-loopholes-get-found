"""Run the main experiment as one resumable queue (configs/protocol.yaml, main_runs).

Usage, on the GPU machine in tmux, from the repository root after `source /workspace/loopholes-env.sh`:
    python -m src.train.queue --dry-run          # the plan, and what is already done
    python -m src.train.queue                    # everything; rerun the same command to resume
    python -m src.train.queue --checkpoints-only # stage 1 only
    python -m src.train.queue --seeds 1 2        # only these seeds (to split the work over two machines)
    python -m src.train.queue --pack-only        # copy the result files of finished runs again (safe while running)

Stage 1, checkpoints. For each seed s: the seeding set, SFT, sampled propensity on both pools (with
the seeding-string FPR, giving risk_scores.csv) and string likelihood of main-rot<s> (rotation s,
SFT seed s), and for the H3 control main-clean<s> (no seeding). All checkpoints, and therefore all
predictions, exist before the first main GRPO run.
Stage 2, runs. Seed by seed, the core and held-out runs from main-rot<s> (tag main) and the control
run from main-clean<s> (tag main-ctrl), each through scripts/run_grpo.sh; then the runs of the held-out
judge added in protocol-v1.1, for seeds 0 to 2.

A step is skipped when its output exists; a run is done when its meta_end.json exists. A run
directory with logs but no meta_end.json is a failed attempt: it is moved to <dir>.failed-<time>
and the run is repeated. A failing step is logged and the queue moves on (runs of a failed
checkpoint are skipped). After each checkpoint its propensity files, and after each run its small
result files, are copied to results/main/, so one scp of results/main fetches the experiment.
The small files include family_counts.csv: per training step and family, the samples in the family
and how many the judge accepted (from rollouts.jsonl, which keeps every family response), which the
early-growth and winner analyses need.
Progress goes to runs/main/queue.log.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.paths import ROOT

PROTOCOL = ROOT / "configs" / "protocol.yaml"
STRING_FPR = "results/stage2/string_fpr.csv"
RUN_FILES = ("summary.txt", "eval_summary.jsonl", "events_tau0.05.json", "predicted_risk.csv", "train_metrics.jsonl",
             "meta.json", "meta_end.json", "gpu.txt")
CHECKPOINT_FILES = {
    "data_summary.json": "data_summary.json",
    "propensity/propensity_sampled.csv": "propensity_sampled.csv",
    "propensity/risk_scores.csv": "risk_scores.csv",
    "likelihood/propensity.csv": "likelihood.csv",
}


def load_plan(protocol=PROTOCOL, seeds=None):
    """(checkpoints, runs) in execution order, from main_runs of the protocol (optionally some seeds only)."""
    mr = yaml.safe_load(Path(protocol).read_text())["main_runs"]
    control = mr.get("h3_control")
    checkpoints, runs = [], []
    for s in mr["seeds"]:
        checkpoints.append({"name": f"main-rot{s}", "seed": s, "rotation": s, "seeding": "rotated"})
        if control:
            checkpoints.append({"name": f"main-clean{s}", "seed": s, "rotation": None, "seeding": control["seeding"]})
    for s in mr["seeds"]:
        for group in ("core", "held_out"):
            for judge in mr[group]["judges"]:
                for pool in mr[group]["pools"]:
                    runs.append({"judge": judge, "pool": pool, "seed": s, "checkpoint": f"main-rot{s}",
                                 "tag": mr["run_tags"]["seeded"]})
        if control:
            for judge in control["judges"]:
                for pool in control["pools"]:
                    runs.append({"judge": judge, "pool": pool, "seed": s, "checkpoint": f"main-clean{s}",
                                 "tag": mr["run_tags"]["control"]})
    added = mr.get("held_out_added")             # protocol-v1.1: after every run of protocol-v1
    if added:
        for s in mr["seeds"]:
            for judge in added["judges"]:
                for pool in added["pools"]:
                    runs.append({"judge": judge, "pool": pool, "seed": s, "checkpoint": f"main-rot{s}",
                                 "tag": mr["run_tags"]["seeded"]})
    if len(runs) != mr["n_runs"]:
        raise ValueError(f"the plan has {len(runs)} runs, the protocol says {mr['n_runs']}")
    if seeds is not None:
        checkpoints = [c for c in checkpoints if c["seed"] in seeds]
        runs = [r for r in runs if r["seed"] in seeds]
    return checkpoints, runs


def run_name(run):
    return f"{run['judge']}-{run['pool']}-s{run['seed']}-{run['tag']}"


def checkpoint_steps(ck):
    """(label, output that marks it done, command, failure tolerated) for one checkpoint."""
    d = f"runs/sft/{ck['name']}"
    py = [sys.executable, "-m"]
    build = py + ["src.train.seed_data", "--name", ck["name"]]
    build += ["--rotation", str(ck["rotation"])] if ck["seeding"] == "rotated" else ["--seeding", ck["seeding"]]
    return [
        ("seeding set", f"{d}/data.jsonl", build, False),
        (
            "SFT",
            f"{d}/model/config.json",
            py + ["src.train.sft", "--name", ck["name"], "--seed", str(ck["seed"])],
            False,
        ),
        (
            "sampled propensity",
            f"{d}/propensity/risk_scores.csv",
            py
            + [
                "src.audit.sampled_propensity",
                "--policy",
                f"{d}/model",
                "--run-dir",
                f"{d}/propensity",
                "--fpr",
                STRING_FPR,
            ],
            False,
        ),
        (
            "string likelihood",
            f"{d}/likelihood/propensity.csv",
            py
            + [
                "src.audit.likelihood",
                "--policy",
                f"{d}/model",
                "--run-dir",
                f"{d}/likelihood",
                "--difficulty-run",
                "none",
                "--fpr",
                STRING_FPR,
            ],
            True,
        ),
    ]


def run_command(run):
    ck = f"runs/sft/{run['checkpoint']}"
    return ["bash", "scripts/run_grpo.sh", run["judge"], run["pool"], str(run["seed"]), "--policy-model", f"{ck}/model",
            "--risk-csv", f"{ck}/propensity/risk_scores.csv", "--tag", run["tag"]]


def run_state(run, root=ROOT):
    d = root / "runs" / "grpo" / run_name(run)
    if (d / "meta_end.json").exists():
        return "done"
    if d.exists() and any(d.glob("*.jsonl")):
        return "failed attempt"
    return "pending"


def set_aside(run, root=ROOT):
    d = root / "runs" / "grpo" / run_name(run)
    target = d.with_name(f"{d.name}.failed-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    d.rename(target)
    return target


def pack_checkpoint(ck, root=ROOT):
    src, dst = root / "runs" / "sft" / ck["name"], root / "results" / "main" / f"sft-{ck['name']}"
    dst.mkdir(parents=True, exist_ok=True)
    for rel, name in CHECKPOINT_FILES.items():
        if (src / rel).exists():
            shutil.copy2(src / rel, dst / name)


def family_counts(run_dir):
    """Rows (step, family, n, accepted) from rollouts.jsonl, which keeps every training response in a family."""
    counts = {}
    path = Path(run_dir) / "rollouts.jsonl"
    if not path.exists():
        return []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("family"):
                c = counts.setdefault((r["step"], r["family"]), [0, 0])
                c[0] += 1
                c[1] += bool(r["accepted"])
    return [{"step": st, "family": fam, "n": n, "accepted": a} for (st, fam), (n, a) in sorted(counts.items())]


def pack_run(run, root=ROOT):
    src, dst = root / "runs" / "grpo" / run_name(run), root / "results" / "main" / run_name(run)
    dst.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    rows = family_counts(src)
    with (dst / "family_counts.csv").open("w") as f:
        f.write("step,family,n,accepted\n")
        for r in rows:
            f.write(f"{r['step']},{r['family']},{r['n']},{r['accepted']}\n")


class Log:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(line + "\n")


def execute(cmd, log):
    log("$ " + " ".join(cmd))
    start = time.time()
    code = subprocess.run(cmd, cwd=ROOT).returncode
    log(f"exit {code} after {(time.time() - start) / 60:.1f} min")
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and its state, run nothing")
    ap.add_argument("--checkpoints-only", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", help="only these seeds")
    ap.add_argument("--pack-only", action="store_true", help="copy result files of finished checkpoints and runs")
    ap.add_argument("--protocol", default=str(PROTOCOL))
    args = ap.parse_args()
    checkpoints, runs = load_plan(args.protocol, args.seeds)

    if args.dry_run:
        print(f"{len(checkpoints)} checkpoints, {len(runs)} runs")
        for ck in checkpoints:
            states = [f"{label}: {'done' if (ROOT / out).exists() else 'pending'}"
                      for label, out, _, _ in checkpoint_steps(ck)]
            print(f"  checkpoint {ck['name']:12s} " + "; ".join(states))
        for run in runs:
            print(f"  run {run_name(run):48s} from {run['checkpoint']:12s} {run_state(run)}")
        return

    if args.pack_only:
        for ck in checkpoints:
            if (ROOT / "runs" / "sft" / ck["name"]).exists():
                pack_checkpoint(ck)
        done = [r for r in runs if run_state(r) == "done"]
        for run in done:
            pack_run(run)
        print(f"packed {len(done)} finished runs into results/main/")
        return

    log = Log(ROOT / "runs" / "main" / "queue.log")
    if not (ROOT / STRING_FPR).exists():
        raise SystemExit(f"{STRING_FPR} is missing; it is part of the frozen protocol")
    log(f"queue started: {len(checkpoints)} checkpoints, {len(runs)} runs")
    failed_checkpoints, failures = set(), []
    for ck in checkpoints:
        for label, out, cmd, tolerated in checkpoint_steps(ck):
            if (ROOT / out).exists():
                continue
            log(f"checkpoint {ck['name']}: {label}")
            if execute(cmd, log) != 0 or not (ROOT / out).exists():
                failures.append(f"{ck['name']} {label}")
                if not tolerated:
                    failed_checkpoints.add(ck["name"])
                    log(f"!! checkpoint {ck['name']} failed at {label}; its runs will be skipped")
                    break
        pack_checkpoint(ck)
    if args.checkpoints_only:
        log(f"checkpoints finished; failures: {failures or 'none'}")
        return

    for i, run in enumerate(runs, 1):
        name = run_name(run)
        state = run_state(run)
        if state == "done":
            pack_run(run)
            continue
        if run["checkpoint"] in failed_checkpoints:
            log(f"skip {name}: checkpoint {run['checkpoint']} failed")
            failures.append(name)
            continue
        if state == "failed attempt":
            log(f"moved a failed attempt of {name} to {set_aside(run).name}")
        log(f"run {i}/{len(runs)}: {name}")
        code = execute(run_command(run), log)
        if run_state(run) == "done":
            subprocess.run([sys.executable, "-m", "src.monitor.events", f"runs/grpo/{name}", "--tau", "0.05"],
                           cwd=ROOT, stdout=subprocess.DEVNULL)
            pack_run(run)
            log(f"done {name}")
        else:
            failures.append(name)
            log(f"!! {name} failed (exit {code}); rerun the queue to retry it")
    done = sum(run_state(r) == "done" for r in runs)
    log(f"queue finished: {done}/{len(runs)} runs done; failures: {failures or 'none'}")
    (ROOT / "results" / "main").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "main" / "queue_state.json").write_text(
        json.dumps(
            {
                "done": done,
                "runs": len(runs),
                "failures": failures,
                "finished": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
