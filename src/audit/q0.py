"""Acceptance of a starting policy's outputs by one judge, measured before RL (experiment C).

Usage (GPU):
    python -m src.audit.q0 --propensity-dir runs/sft/low-F1-s0/propensity --judge qwen2p5_math_1p5b_instruct
Local test: add --local-test (the judge runs on CPU or MPS through src.judges.local)

The judge grades every response of the sampled-propensity audit on one pool (samples.jsonl, 16 responses to each of
the 250 audit problems, sampled with the RL settings), with the question, its native prompt and decoding, exactly as in
GRPO. q0 is the share accepted; the expected-advantage score uses it. This is the main experiment's quantity (the
judge's acceptance of the starting policy's outputs) on 4,000 outputs of the audit problems instead of the 200
step-0 evaluation outputs, so it is on record before training; the run's step-0 evaluation measures it again.
Writes judged.csv (one row per response: uid, sample index, family, gold, truncation, verdict) and q0.json (q0, gold
accuracy, and the acceptance of each family's own responses, an in-context check of the string FPR).
"""

import argparse
import json

import pandas as pd

from src.data.pools import DEFAULT_MANIFEST, load_pool
from src.judges.prompts import judge_config
from src.paths import ROOT, display_path
from src.pipeline.records import write_meta


def judge_items(samples, problems):
    """Judge items in the order of the samples; problems maps uid to the pool problem."""
    return [{"question": problems[s["uid"]]["problem"], "reference": problems[s["uid"]]["reference"],
             "response": s["response"], "truncated": s["truncated"]} for s in samples]


def summarize(judged):
    """q0, gold accuracy and the acceptance of each family's own responses."""
    fam = judged[judged["family"].notna() & (judged["family"] != "")]
    by_family = {f: {"n": int(len(g)), "accepted": int(g["accepted"].sum())} for f, g in fam.groupby("family")}
    return {"n": int(len(judged)), "q0": float(judged["accepted"].mean()),
            "gold_accuracy": float(judged["gold_correct"].mean()),
            "parse_fail": float(judged["parse_fail"].mean()), "family_share": float(len(fam) / len(judged)),
            "family_acceptance": by_family}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--propensity-dir", required=True, help="run dir of src.audit.sampled_propensity")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--pool", default="hard")
    ap.add_argument("--out", help="output dir (default: <propensity-dir>/../q0-<judge>)")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-test", action="store_true")
    args = ap.parse_args()
    prop_dir = ROOT / args.propensity_dir
    out = ROOT / args.out if args.out else prop_dir.parent / f"q0-{args.judge}"
    out.mkdir(parents=True, exist_ok=True)

    with (prop_dir / "samples.jsonl").open() as f:
        samples = [s for s in map(json.loads, f) if s["pool"] == args.pool]
    if not samples:
        raise SystemExit(f"no {args.pool} samples in {display_path(prop_dir / 'samples.jsonl')}")
    problems = {p["uid"]: p for p in load_pool(args.pool, "audit", args.manifest)}
    cfg = judge_config(args.judge)
    if args.local_test:
        from src.judges.local import LocalJudge
        judge = LocalJudge(cfg)
    else:
        from src.judges.vllm_judge import VLLMJudge
        judge = VLLMJudge(cfg, seed=args.seed)
    results, seconds = judge.judge(judge_items(samples, problems))

    judged = pd.DataFrame([{"uid": s["uid"], "sample_idx": s["sample_idx"], "family": s["family"] or "",
                            "gold_correct": bool(s["gold_correct"]), "truncated": bool(s["truncated"]),
                            "accepted": bool(r.accepted), "parse_fail": bool(r.parse_fail)}
                           for s, r in zip(samples, results)])
    judged.to_csv(out / "judged.csv", index=False)
    summary = {"judge": args.judge, "pool": args.pool, "seconds": round(seconds), **summarize(judged)}
    (out / "q0.json").write_text(json.dumps(summary, indent=2))
    write_meta(out, name="meta_q0.json", stage="q0", judge=args.judge, pool=args.pool,
               propensity_dir=args.propensity_dir, local_test=args.local_test)
    print(json.dumps(summary, indent=2))
    print(f"wrote {display_path(out / 'q0.json')}")


if __name__ == "__main__":
    main()
