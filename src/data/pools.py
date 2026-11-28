"""Build the hard and easy pools from a difficulty run.

Usage:
    python -m src.data.pools --difficulty-run runs/difficulty [--out pools/pools-v1.json]

A problem is in the hard band if 1-2 of its n samples were correct and in the easy
band if 5-7 were (configs/data.yaml). Both pools get the same size, the smaller band
size capped at pool_size_max; the larger band is subsampled. Each pool is split into
eval (held out, used to monitor exploitation during GRPO) and train, and an audit
subset of train is set aside for the propensity measurement. All draws are seeded.

The manifest (problem ids per pool and split, with each problem's correct count) is
small and committed to git, so every machine trains on the same pools; problem text
comes from data/processed/pool_candidates.jsonl through load_pool().
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from src.paths import ROOT, display_path
from src.pipeline.records import git_state
from src.pipeline.shards import read_shards

CANDIDATES = ROOT / "data" / "processed" / "pool_candidates.jsonl"
DEFAULT_MANIFEST = ROOT / "pools" / "pools-v1.json"


def per_problem_stats(records):
    """uid -> number of samples, correct, truncated and no-answer counts."""
    stats = {}
    for r in records:
        s = stats.setdefault(r["uid"], {"n": 0, "correct": 0, "truncated": 0, "no_answer": 0})
        s["n"] += 1
        s["correct"] += r["gold_correct"]
        s["truncated"] += r["truncated"]
        s["no_answer"] += r["no_answer"]
    return stats


def build_manifest(stats, difficulty_cfg, seed):
    n_values = {s["n"] for s in stats.values()}
    if len(n_values) != 1:
        raise ValueError(f"problems have different sample counts: {sorted(n_values)}")
    bands = {"hard": difficulty_cfg["hard_correct"], "easy": difficulty_cfg["easy_correct"]}
    members = {name: sorted(u for u, s in stats.items() if lo <= s["correct"] <= hi)
               for name, (lo, hi) in bands.items()}
    pool_size = min(min(len(m) for m in members.values()), difficulty_cfg["pool_size_max"])
    n_eval, n_audit = difficulty_cfg["eval_per_pool"], difficulty_cfg["audit_per_pool"]
    if pool_size < n_eval + n_audit:
        raise ValueError(f"pool size {pool_size} too small for {n_eval} eval and {n_audit} audit problems")

    pools = {}
    for name, uids in members.items():
        rng = random.Random(f"{seed}:{name}")
        chosen = sorted(rng.sample(uids, pool_size))
        eval_ = sorted(rng.sample(chosen, n_eval))
        train = sorted(set(chosen) - set(eval_))
        audit = sorted(rng.sample(train, n_audit))
        pools[name] = {"train": train, "eval": eval_, "audit": audit}
    in_pools = {u for p in pools.values() for u in p["train"] + p["eval"]}
    return {
        "version": 1, "seed": seed, "n_samples": n_values.pop(), "bands": bands,
        "band_sizes": {k: len(v) for k, v in members.items()}, "pool_size": pool_size,
        "eval_per_pool": n_eval, "audit_per_pool": n_audit, "pools": pools,
        "n_correct": {u: stats[u]["correct"] for u in sorted(in_pools)},
    }


def load_candidates(path=CANDIDATES):
    return {r["uid"]: r for r in (json.loads(line) for line in Path(path).open())}


def load_pool(pool, split, manifest_path=DEFAULT_MANIFEST, candidates_path=CANDIDATES):
    """Problems of one pool split as dicts with uid, source, problem, reference, level, n_correct."""
    manifest = json.loads(Path(manifest_path).read_text())
    candidates = load_candidates(candidates_path)
    return [{"uid": u, "source": candidates[u]["source"], "problem": candidates[u]["problem"],
             "reference": candidates[u]["answer"], "level": candidates[u]["level"],
             "n_correct": manifest["n_correct"][u]} for u in manifest["pools"][pool][split]]


def describe(manifest, candidates):
    lines = [f"band sizes {manifest['band_sizes']} -> pool size {manifest['pool_size']} "
             f"({manifest['eval_per_pool']} eval, {manifest['audit_per_pool']} audit from train)"]
    for name, splits in manifest["pools"].items():
        uids = splits["train"] + splits["eval"]
        src = Counter(candidates[u]["source"] for u in uids)
        lvl = Counter(candidates[u]["level"] for u in uids if candidates[u]["source"] == "math_train")
        acc = sum(manifest["n_correct"][u] for u in uids) / (len(uids) * manifest["n_samples"])
        corr = Counter(manifest["n_correct"][u] for u in uids)
        lines.append(
            f"{name}: train {len(splits['train'])}, eval {len(splits['eval'])}, audit {len(splits['audit'])} | "
            f"base accuracy {acc:.3f} | correct counts {dict(sorted(corr.items()))}"
        )
        lines.append(f"  sources {dict(src)} | MATH levels {dict(sorted(lvl.items()))}")
    return "\n".join(lines)


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty-run", default="runs/difficulty")
    ap.add_argument("--out", default=str(DEFAULT_MANIFEST.relative_to(ROOT)))
    args = ap.parse_args()

    run_dir = ROOT / args.difficulty_run
    records = read_shards(run_dir / "samples")
    stats = per_problem_stats(records)
    candidates = load_candidates()
    missing = set(candidates) - set(stats)
    if missing:
        raise SystemExit(f"{len(missing)} candidates have no samples; finish the difficulty run first")
    manifest = build_manifest(stats, cfg["difficulty"], cfg["seed"])
    manifest.update({"difficulty_run": args.difficulty_run, "difficulty_summary": json.loads(
        (run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else None,
        "git": git_state()})
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1))
    print(describe(manifest, candidates))
    print(f"wrote {display_path(out)}")


if __name__ == "__main__":
    main()
