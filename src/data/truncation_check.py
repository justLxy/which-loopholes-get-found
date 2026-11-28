"""Why are policy responses truncated, and would a longer limit help?

Usage:
    python -m src.data.truncation_check --run-dir runs/difficulty-test200
        [--retry 40] [--max-new-tokens 2048] [--backend vllm|local|none]

Part 1 (no GPU) analyses the truncated responses of a difficulty run: rates by
source and MATH level, whether the tail is a repetition loop, whether a boxed
answer was already written, and example tails.

Part 2 re-samples up to --retry problems that had a truncated response, with the
same per-problem seed and a longer limit, and reports how many of the truncated
responses finish and how many become correct, plus a what-if table of truncation
and accuracy at several length caps. Even with the same seeds, vLLM on a GPU does
not reproduce responses token for token (batch-dependent numerics), so the retry
is a fresh sample from the same distribution; the report shows how often the
starts still match. With --backend none, part 1 and the cap table (from a saved
retry) are printed without a GPU.
"""

import argparse
import json
import random
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from src.data.difficulty import load_units, make_process
from src.paths import display_path
from src.pipeline.records import validate_sample_record
from src.pipeline.shards import read_shards

ROOT = Path(__file__).resolve().parents[2]
LOOP_RATIO = 0.12   # zlib ratio of the last 1500 chars below which the tail is a loop; loops measure 0.02-0.08,
                    # the lowest of 364 long MATH solutions 0.121 (Asymptote code)


def tail_ratio(text, chars=1500):
    """Compressed/raw size of the response tail: repetition loops compress far better than prose."""
    tail = text[-chars:].encode("utf-8")
    return len(zlib.compress(tail, 9)) / max(len(tail), 1)


def analyse(records, levels):
    trunc = [r for r in records if r["truncated"]]
    by_source = defaultdict(lambda: [0, 0])
    by_level = defaultdict(lambda: [0, 0])
    for r in records:
        by_source[r["source"]][0] += r["truncated"]
        by_source[r["source"]][1] += 1
        if r["source"] == "math_train":
            by_level[levels.get(r["uid"]) or "?"][0] += r["truncated"]
            by_level[levels.get(r["uid"]) or "?"][1] += 1
    ratios = sorted(tail_ratio(r["response"]) for r in trunc)
    correct = Counter()
    for r in records:
        correct[r["uid"]] += r["gold_correct"]
    trunc_uids = {r["uid"] for r in trunc}
    return {
        "responses": len(records),
        "truncated": len(trunc),
        "by_source": {k: f"{v[0]}/{v[1]} ({v[0] / v[1]:.1%})" for k, v in sorted(by_source.items())},
        "by_math_level": {k: f"{v[0]}/{v[1]} ({v[0] / v[1]:.1%})" for k, v in sorted(by_level.items())},
        "loop_share": sum(x < LOOP_RATIO for x in ratios) / max(len(ratios), 1),
        "tail_ratio_quartiles": [round(ratios[int(q * (len(ratios) - 1))], 3) for q in (0.25, 0.5, 0.75)]
        if ratios
        else [],
        "already_boxed": sum(r["n_boxed"] > 0 for r in trunc),
        "correct_despite_truncation": sum(r["gold_correct"] for r in trunc),
        "problems_with_truncation": len(trunc_uids),
        "their_correct_hist": dict(sorted(Counter(correct[u] for u in trunc_uids).items())),
    }


def compare(original, retried):
    """Match retried responses to originals by (uid, sample_idx)."""
    orig = {(r["uid"], r["sample_idx"]): r for r in original}
    rows = []
    for r in retried:
        o = orig[(r["uid"], r["sample_idx"])]
        rows.append({"was_truncated": o["truncated"], "still_truncated": r["truncated"],
                     "now_correct": r["gold_correct"], "was_correct": o["gold_correct"],
                     "same_start": r["response"][:300] == o["response"][:300],
                     "identical": r["response"] == o["response"],
                     "n_tokens": r["n_tokens"], "orig_tokens": o["n_tokens"]})
    was = [x for x in rows if x["was_truncated"]]
    finished = [x for x in was if not x["still_truncated"]]
    kept = [x for x in rows if not x["was_truncated"]]
    return {
        "retried_responses": len(rows), "originally_truncated": len(was),
        "finish_with_longer_limit": len(finished),
        "correct_after_finishing": sum(x["now_correct"] for x in finished),
        "still_truncated": len(was) - len(finished),
        "mean_tokens_of_finished": round(sum(x["n_tokens"] for x in finished) / max(len(finished), 1)),
        "same_first_300_chars": f"{sum(x['same_start'] for x in rows)}/{len(rows)}",
        "untruncated_originals_reproduced_exactly": f"{sum(x['identical'] for x in kept)}/{len(kept)}",
        "accuracy_before_after": [round(sum(x["was_correct"] for x in rows) / max(len(rows), 1), 3),
                                  round(sum(x["now_correct"] for x in rows) / max(len(rows), 1), 3)],
    }


def cap_table(retried, overall_truncated_share, old_limit, caps=(1024, 1280, 1536, 1792, 2048)):
    """What-if table from responses sampled at a long limit: cutting a response at cap X
    leaves it unchanged if it is shorter, and makes it truncated (and wrong) otherwise."""
    over_old = [r for r in retried if r["n_tokens"] > old_limit or r["truncated"]]
    rows = []
    for cap in caps:
        cut = [r["truncated"] or r["n_tokens"] > cap for r in retried]
        ok = [r["gold_correct"] and not c for r, c in zip(retried, cut)]
        cond = sum(r["truncated"] or r["n_tokens"] > cap for r in over_old) / max(len(over_old), 1)
        rows.append({"cap": cap, "truncated_in_retry_set": round(sum(cut) / len(retried), 3),
                     "accuracy_in_retry_set": round(sum(ok) / len(retried), 3),
                     "est_truncated_overall": round(overall_truncated_share * cond, 4)})
    return rows


def show_examples(records, k=3):
    trunc = [r for r in records if r["truncated"]]
    loops = [r for r in trunc if tail_ratio(r["response"]) < LOOP_RATIO][:k]
    prose = [r for r in trunc if tail_ratio(r["response"]) >= LOOP_RATIO][:k]
    for label, group in (("loop", loops), ("non-loop", prose)):
        for r in group:
            tail = r["response"][-300:].replace("\n", "\\n")
            print(f"  [{label}] {r['uid']}#{r['sample_idx']} ratio={tail_ratio(r['response']):.2f}: ...{tail}")


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--retry", type=int, default=40, help="problems to re-sample with the longer limit")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--backend", choices=["vllm", "local", "none"], default="vllm")
    ap.add_argument("--policy", default=cfg["prompt"]["model"])
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    records = read_shards(run_dir / "samples")
    units = {u["uid"]: u for u in load_units()}
    levels = {uid: u["level"] for uid, u in units.items()}
    n = Counter(r["uid"] for r in records).most_common(1)[0][1]
    part1 = analyse(records, levels)
    print("== part 1: truncated responses at the current limit")
    print(json.dumps(part1, indent=2))
    show_examples(records)
    old_limit = max(r["n_tokens"] for r in records if r["truncated"]) if part1["truncated"] else None
    out = run_dir / f"truncation-retry-{args.max_new_tokens}.jsonl"

    def print_cap_table(retried):
        print(f"\n== length caps, from the {len(retried)} responses re-sampled at {args.max_new_tokens} tokens")
        for row in cap_table(retried, part1["truncated"] / part1["responses"], old_limit):
            print("  " + json.dumps(row))

    if args.backend == "none" or args.retry == 0:
        if out.exists():
            print_cap_table([json.loads(line) for line in out.open()])
        return

    trunc_uids = sorted({r["uid"] for r in records if r["truncated"]})
    retry_uids = sorted(random.Random(cfg["seed"]).sample(trunc_uids, min(args.retry, len(trunc_uids))))
    batch = [units[u] for u in retry_uids]
    if args.backend == "vllm":
        from src.policy.vllm_policy import VLLMPolicy
        policy = VLLMPolicy(args.policy, args.max_new_tokens, args.gpu_memory_utilization, cfg["seed"])
    else:
        from src.policy.local import LocalPolicy
        policy = LocalPolicy(args.policy)
    sampling = {"temperature": cfg["difficulty"]["temperature"], "top_p": 1.0, "max_new_tokens": args.max_new_tokens}
    timer = {"generate_seconds": 0.0, "tokens": 0}
    retried = make_process(policy, n, sampling, cfg["seed"], f"{run_dir.name}-retry{args.max_new_tokens}", timer)(batch)
    with out.open("w", encoding="utf-8") as f:
        for r in retried:
            validate_sample_record(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n== part 2: {len(batch)} problems re-sampled with max_new_tokens={args.max_new_tokens} "
          f"in {timer['generate_seconds']:.0f}s (saved to {display_path(out)})")
    original = [r for r in records if r["uid"] in set(retry_uids)]
    print(json.dumps(compare(original, retried), indent=2))
    print_cap_table(retried)


if __name__ == "__main__":
    main()
