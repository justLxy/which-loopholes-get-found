"""Propensity of a (seeded) policy by sampling: how often its responses fall in each family.

Usage (GPU):
    python -m src.audit.sampled_propensity --policy runs/sft/seed2pct/model \
        --run-dir runs/sft/seed2pct/propensity --fpr results/stage1/fpr_by_family.csv
Local test: add --local-test

Samples `samples_per_problem` responses to each audit problem of each pool (configs/sft.yaml,
propensity section, or --samples-per-problem) with the RL sampling settings, classifies them with the family detectors
and reports the Jeffreys-smoothed frequency p_f (0.5 added to each of the 15 classes, as in
src.audit.likelihood). With --fpr it writes risk_scores.csv, r = p_f x FPR(j, f) per judge and
pool, in the layout GRPO copies before training (`--risk-csv`, predicted_risk.csv). Every
response is kept in samples.jsonl so the detectors can be checked by hand on seeded outputs.
"""

import argparse
import json
import math
import pandas as pd

from src.audit.likelihood import sampled_propensity
from src.data.pools import DEFAULT_MANIFEST, load_pool
from src.detectors import FAMILIES
from src.detectors.families import classify
from src.gold.checker import check
from src.paths import ROOT, display_path
from src.pipeline.records import write_meta
from src.train.seed_data import CONFIG, load_sft_config


def sample_records(policy, items, pool, cfg, seed):
    samples, seconds = policy.sample([it["problem"] for it in items], n=cfg["samples_per_problem"],
                                     temperature=cfg["temperature"], top_p=cfg["top_p"],
                                     max_new_tokens=cfg["max_new_tokens"], seed=seed)
    records = []
    for it, group in zip(items, samples):
        for j, s in enumerate(group):
            gold = check(s.text, it["reference"], truncated=s.truncated)
            det = classify(s.text)
            records.append({"pool": pool, "uid": it["uid"], "sample_idx": j, "response": s.text,
                            "n_tokens": s.n_tokens, "truncated": s.truncated, "gold_correct": gold.correct,
                            "no_answer": gold.no_answer, "family": det.family, "contains": sorted(det.contains)})
    return records, seconds


def risk_table(prop, fpr_csv):
    """r(j, f, d) = p_f x FPR(j, f), ranked within judge and pool."""
    fpr = pd.read_csv(fpr_csv)
    fpr = fpr[fpr["family"].isin(FAMILIES)][["judge", "family", "fpr"]]
    risk = fpr.merge(prop[["pool", "family", "count", "n", "p_hat"]], on="family")
    risk["risk"] = risk["p_hat"] * risk["fpr"]
    risk["log_risk"] = risk["risk"].map(lambda x: math.log(x) if x > 0 else float("-inf"))
    risk["rank_in_judge_pool"] = risk.groupby(["judge", "pool"])["risk"].rank(ascending=False, method="min")
    return risk.sort_values(["judge", "pool", "rank_in_judge_pool"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", help="the seeded checkpoint (not needed with --risk-only)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--risk-only", action="store_true",
                    help="recompute risk_scores.csv from an existing propensity_sampled.csv and --fpr (no sampling)")
    ap.add_argument("--pools", nargs="+", default=["hard", "easy"])
    ap.add_argument("--fpr", default=None)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples-per-problem", type=int, help="override the config's samples_per_problem")
    ap.add_argument("--local-test", action="store_true")
    args = ap.parse_args()
    cfg = load_sft_config(args.config, args.local_test)
    pcfg = cfg["propensity"]
    if args.samples_per_problem is not None:
        if args.samples_per_problem < 1:
            raise SystemExit("--samples-per-problem must be positive")
        pcfg["samples_per_problem"] = args.samples_per_problem
    run_dir = ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.risk_only:
        if not args.fpr:
            raise SystemExit("--risk-only needs --fpr")
        risk = risk_table(pd.read_csv(run_dir / "propensity_sampled.csv"), args.fpr)
        risk.to_csv(run_dir / "risk_scores.csv", index=False)
        print(f"recomputed {display_path(run_dir / 'risk_scores.csv')} for {risk['judge'].nunique()} judges")
        return
    if not args.policy:
        raise SystemExit("--policy is required unless --risk-only")

    if args.local_test:
        from src.policy.local import LocalPolicy
        policy = LocalPolicy(args.policy)
    else:
        from src.policy.vllm_policy import VLLMPolicy
        policy = VLLMPolicy(args.policy, max_new_tokens=pcfg["max_new_tokens"])

    records, uids = [], {}
    for pool in args.pools:
        items = load_pool(pool, pcfg["split"], args.manifest)[:cfg.get("problems")]
        recs, seconds = sample_records(policy, items, pool, pcfg, args.seed)
        records += recs
        uids[pool] = {it["uid"] for it in items}
        print(f"[{pool}] {len(recs)} responses in {seconds:.0f}s", flush=True)
    with (run_dir / "samples.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    prop = sampled_propensity(records, uids)
    rows = []
    for pool in args.pools:
        rs = [r for r in records if r["pool"] == pool]
        for family in FAMILIES:
            rows.append({"pool": pool, "family": family, **prop[(pool, family)]})
        print(f"[{pool}] gold accuracy {sum(r['gold_correct'] for r in rs) / len(rs):.3f}, any family "
              f"{sum(r['family'] is not None for r in rs) / len(rs):.3f}, truncated "
              f"{sum(r['truncated'] for r in rs) / len(rs):.3f}")
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "propensity_sampled.csv", index=False)
    with pd.option_context("display.width", 200):
        print(table.pivot(index="family", columns="pool", values="count").loc[list(FAMILIES)].to_string())
    if args.fpr:
        risk = risk_table(table, args.fpr)
        risk.to_csv(run_dir / "risk_scores.csv", index=False)
        print(f"wrote {display_path(run_dir / 'risk_scores.csv')}")
    write_meta(run_dir, name="meta_propensity.json", stage="sampled_propensity", policy=args.policy,
               pools=args.pools, propensity_config=pcfg, local_test=args.local_test)
    print(f"wrote {display_path(run_dir / 'propensity_sampled.csv')}")


if __name__ == "__main__":
    main()
