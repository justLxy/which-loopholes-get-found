"""Train the policy with GRPO against one judge, logging what the judge lets through.

Usage (on the GPU; scripts/run_grpo.sh starts the judge server first):
    python -m src.train.grpo --judge r1_distill_verifier_1p5b --pool hard --seed 0
    python -m src.train.grpo --judge engineered:no_box --pool hard --seed 0
Experiment B (configs/masking.yaml, through src.train.masking): --mask-family F rejects every response the
detectors put in family F, in training and in the evaluations, and sets F's predicted risk to zero.
Local smoke test (CPU or Apple MPS, tiny model, no vLLM, no server):
    python -m src.train.grpo --judge engineered:no_box --pool hard --local-test
GPU integration test (a 24GB card is enough: 0.5B policy, small batch, 5 steps, real vLLM path):
    bash scripts/run_grpo.sh xverify_0p5b easy 0 --gpu-test

Before training, the run directory gets meta.json (git state, configs, versions) and, for a
real judge, predicted_risk.csv: the risk scores r(j, f, d) of this judge and pool, copied
from results/stage1/risk_scores.csv, so the prediction is on record before any outcome.
During training it collects
    train_metrics.jsonl  judge, gold and family metrics of every training batch
    rollouts.jsonl       training responses that fooled the judge or match a family
    train_log.jsonl      TRL's own logs (loss, reward, clip ratios, ...)
    eval.jsonl           judge records of the periodic evaluations
    eval_summary.jsonl   one line per evaluation with e/<family>
"""

import argparse
import copy
import csv
import os
import random
import sys
import time
from pathlib import Path

import yaml

from src.data.pools import DEFAULT_MANIFEST, load_pool
from src.paths import ROOT, display_path
from src.pipeline.records import write_meta
from src.policy.prompt import build_messages

CONFIG = ROOT / "configs" / "grpo.yaml"
RISK_SCORES = ROOT / "results" / "stage1" / "risk_scores.csv"


def deep_update(base, overrides):
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


PROFILES = ("local_test", "gpu_test")


def load_grpo_config(path=CONFIG, profile=None):
    """The main configuration, or with the overrides of a test profile (local_test, gpu_test)."""
    cfg = yaml.safe_load(Path(path).read_text())
    overrides = {name: cfg.pop(name, {}) for name in PROFILES}
    if profile is None:
        return cfg
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {PROFILES}")
    return deep_update(cfg, overrides[profile])


def run_name(judge, pool, seed):
    return f"{judge.replace(':', '-')}-{pool}-s{seed}"


def build_dataset(problems):
    from datasets import Dataset

    return Dataset.from_list([{"prompt": build_messages(p["problem"]), "uid": p["uid"], "source": p["source"],
                               "problem": p["problem"], "reference": p["reference"]} for p in problems])


def make_judge(name, backend, url, mask_families=()):
    judge = _make_judge(name, backend, url)
    if mask_families:
        from src.judges.masked import MaskedJudge
        judge = MaskedJudge(judge, mask_families)
    return judge


def _make_judge(name, backend, url):
    if name.startswith("engineered:"):
        from src.judges.engineered import EngineeredJudge
        return EngineeredJudge(name.split(":", 1)[1])
    from src.judges.prompts import judge_config
    cfg = judge_config(name)
    if backend == "server":
        from src.judges.server import ServerJudge
        judge = ServerJudge(cfg, url)
        judge.wait_until_ready()
        return judge
    if backend == "local":
        from src.judges.local import LocalJudge
        return LocalJudge(cfg)
    raise ValueError(f"unknown judge backend {backend!r}")


def risk_snapshot(judge, pool, out_path, risk_csv=RISK_SCORES, mask_families=()):
    """Copy this judge's and pool's predicted risk scores next to the run, with the FPR and risk of a masked
    family set to zero (the patched judge never accepts it). Returns the row count."""
    with open(risk_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["judge"] == judge and r["pool"] == pool]
    if mask_families:
        for r in rows:
            r["masked"] = int(r["family"] in mask_families)
            if r["masked"]:
                r.update({"fpr": "0.0", "risk": "0.0", "log_risk": "-inf"})
        risks = [float(r["risk"]) for r in rows]
        for r in rows:                                   # ties share the best rank, as in src.audit.likelihood
            r["rank_in_judge_pool"] = float(1 + sum(x > float(r["risk"]) for x in risks))
    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return len(rows)


def stop_token_ids(model_name, tokenizer):
    """Tokens that end a completion (as TRL counts them): generation EOS ids, tokenizer EOS and pad."""
    from transformers import GenerationConfig

    eos = GenerationConfig.from_pretrained(model_name).eos_token_id
    ids = set(eos if isinstance(eos, list) else [eos] if eos is not None else [])
    ids |= {t for t in (tokenizer.eos_token_id, tokenizer.pad_token_id) if t is not None}
    return sorted(ids)


def parse_stop_when(text, consecutive=2):
    """"no_box:0.5" -> ("no_box", 0.5, 2); None stays None."""
    if not text:
        return None
    from src.train.monitor import E_KEYS

    key, _, value = text.partition(":")
    if key not in E_KEYS or not value:
        raise SystemExit(f"--stop-when needs KEY:VALUE with KEY in {E_KEYS}, got {text!r}")
    return key, float(value), consecutive


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", required=True, help="judge name from configs/judges.yaml, or engineered:no_box")
    ap.add_argument("--pool", required=True, choices=["hard", "easy"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--run-dir", help="default runs/grpo/<judge>-<pool>-s<seed>")
    ap.add_argument("--policy-model", help="override policy.model, e.g. a seeded SFT checkpoint runs/sft/<name>/model")
    ap.add_argument("--risk-csv", default=str(RISK_SCORES),
                    help="risk scores to record before training (default: stage-1 scores of the base policy)")
    ap.add_argument("--tag", help="appended to the run name, e.g. the seeding name")
    ap.add_argument("--max-steps", type=int, help="override trainer.max_steps")
    ap.add_argument("--learning-rate", type=float, help="override trainer.learning_rate")
    ap.add_argument("--stop-when", metavar="KEY:VALUE",
                    help="pilot runs only: stop once e/KEY >= VALUE in 2 evaluations in a row, e.g. no_box:0.5")
    ap.add_argument("--mask-family", action="append", default=[], metavar="FAMILY",
                    help="experiment B: reject this family by rule in training and evaluation (repeatable)")
    ap.add_argument("--judge-backend", default="server", choices=["server", "local"])
    ap.add_argument("--judge-url", help="override judge_server.url")
    ap.add_argument("--save-all-rollouts", action="store_true", help="keep every training response, not only flagged ones")
    ap.add_argument("--save-final", action="store_true", help="save the final policy weights")
    test = ap.add_mutually_exclusive_group()
    test.add_argument("--local-test", action="store_true", help="tiny CPU/MPS run from the local_test overrides")
    test.add_argument("--gpu-test", action="store_true", help="short GPU run from the gpu_test overrides")
    args = ap.parse_args(argv)
    args.profile = "local_test" if args.local_test else "gpu_test" if args.gpu_test else None
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = load_grpo_config(args.config, args.profile)
    if args.policy_model:
        cfg["policy"]["model"] = args.policy_model
    if args.max_steps is not None:
        cfg["trainer"]["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        cfg["trainer"]["learning_rate"] = args.learning_rate
    name = run_name(args.judge, args.pool, args.seed) + (f"-{args.tag}" if args.tag else "") \
        + (f"-{args.profile.replace('_', '')}" if args.profile else "")
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "runs" / "grpo" / name
    if run_dir.exists() and any(run_dir.glob("*.jsonl")):
        raise SystemExit(f"{display_path(run_dir)} already has logs; choose another --run-dir or remove it")
    run_dir.mkdir(parents=True, exist_ok=True)

    train = load_pool(args.pool, "train", args.manifest)
    evals = load_pool(args.pool, "eval", args.manifest)[:cfg["eval"]["prompts"]]
    if "train_problems" in cfg:
        train = random.Random(args.seed).sample(train, cfg["train_problems"])

    n_risk = 0
    if not args.judge.startswith("engineered:"):
        n_risk = risk_snapshot(args.judge, args.pool, run_dir / "predicted_risk.csv", args.risk_csv, args.mask_family)
    backend = "local" if args.local_test and not args.judge.startswith("engineered:") else args.judge_backend
    write_meta(run_dir, judge=args.judge, pool=args.pool, seed=args.seed, profile=args.profile,
               judge_backend=backend, n_train=len(train), n_eval=len(evals), predicted_risk_rows=n_risk,
               grpo_config=cfg, manifest=display_path(args.manifest), stop_when=args.stop_when,
               risk_csv=display_path(args.risk_csv), tag=args.tag, mask_families=args.mask_family)
    print(f"run {name}: {len(train)} train / {len(evals)} eval problems, predicted risk rows {n_risk}", flush=True)

    # Heavy imports after the cheap checks, so argument errors fail fast.
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from src.device import configure_vllm
    from src.train.monitor import EvalCallback, LogCallback
    from src.train.rewards import JudgeReward

    configure_vllm()
    if args.local_test:
        from src.train import local_compat
        if local_compat.install():
            print("local test: TRL's Triton kernel is unavailable, using the PyTorch log-prob shim", flush=True)
    judge = make_judge(args.judge, backend, args.judge_url or cfg["judge_server"]["url"], args.mask_family)
    model_name = cfg["policy"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    t = cfg["trainer"]
    reward = JudgeReward(judge, args.judge, run_dir, group_size=t["num_generations"],
                         max_completion_length=t["max_completion_length"],
                         stop_ids=stop_token_ids(model_name, tokenizer), save_all=args.save_all_rollouts)
    evaluator = EvalCallback(judge=judge, judge_name=args.judge, problems=evals, pool=args.pool, run_dir=run_dir,
                             run_id=name, policy_model=model_name, every=cfg["eval"]["every_steps"],
                             samples=cfg["eval"]["samples_per_prompt"],
                             max_completion_length=t["max_completion_length"],
                             temperature=t["temperature"], top_p=t["top_p"], stop_when=parse_stop_when(args.stop_when))
    trainer = GRPOTrainer(model=model_name, reward_funcs=reward, train_dataset=build_dataset(train),
                          processing_class=tokenizer,
                          args=GRPOConfig(output_dir=str(run_dir / "trainer"), seed=args.seed, **t),
                          callbacks=[evaluator, LogCallback(run_dir)])
    evaluator.trainer = trainer
    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60
    if args.save_final:
        trainer.save_model(str(run_dir / "final"))
    write_meta(run_dir, name="meta_end.json", train_minutes=minutes, steps=trainer.state.global_step)
    print(f"done: {trainer.state.global_step} steps in {minutes:.1f} min -> {display_path(run_dir)}", flush=True)
    if t.get("use_vllm"):
        # vLLM's sleep-mode allocator can fail during interpreter teardown ("CUDA Error: invalid argument at
        # cumem_allocator.cpp", seed2pct pilot) after every output is written; exit directly instead.
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
