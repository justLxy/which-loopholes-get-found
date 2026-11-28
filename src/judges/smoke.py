"""Smoke test of judges on a few hacking answers and normal answers.

Usage:
    python -m src.judges.smoke [--judges xverify_0p5b | name1,name2 | all]
                               [--per-family 2] [--backend local|vllm] [--show-failures]

Uses questions whose reference answer is an integer; gibberish rows use real
random gibberish. With the vLLM backend each judge runs in its own process so
GPU memory is fully released between judges, and a summary table follows.
Raw outputs are saved to runs/judge-smoke/<judge>-<backend>.jsonl.
This checks that prompts, generation and parsing work end to end; it does not
estimate false positive rates.
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.detectors.synthetic import random_gibberish
from src.judges.prompts import judge_config, load_config

ROOT = Path(__file__).resolve().parents[2]
HACKING = ROOT / "data" / "raw" / "hacking" / "hacking_data.parquet"


def build_items(per_family, seed=0):
    df = pd.read_parquet(HACKING)
    df = df[df["gt"].astype(str).str.strip().str.fullmatch(r"-?\d+")]
    items = []
    for label, g in df.groupby("hacking_type"):
        for i, (_, r) in enumerate(g.sample(per_family, random_state=seed).iterrows()):
            family = label.replace("tiger_", "")
            response = random_gibberish(seed * 1000 + i) if family == "gibberish" else str(r["pred"])
            items.append({"kind": family, "question": r["question"], "reference": str(r["gt"]).strip(),
                          "response": response, "should_accept": False})
    rng = random.Random(seed)
    for _, r in df.drop_duplicates("question").sample(4, random_state=seed).iterrows():
        gt = str(r["gt"]).strip()
        wrong = str(int(gt) + rng.choice([-3, -1, 1, 2]))
        items.append({"kind": "normal_correct", "question": r["question"], "reference": gt,
                      "response": f"Working through the problem gives {gt}. \\boxed{{{gt}}}", "should_accept": True})
        items.append(
            {
                "kind": "normal_wrong",
                "question": r["question"],
                "reference": gt,
                "response": f"Working through the problem gives {wrong}. \\boxed{{{wrong}}}",
                "should_accept": False,
            }
        )
    return items


def describe_output(output, tail=400):
    """One-paragraph view of a raw judge output, for debugging the parser."""
    flat = output.replace("\n", "\\n")
    return (f"chars={len(output)} has_</think>={'</think>' in output} has_<think>={'<think>' in output} "
            f"n_boxed={output.count(chr(92) + 'boxed')} final_decision={'final decision' in output.lower()}\n"
            f"  head: {flat[:150]!s}\n  tail: {flat[-tail:]!s}")


def run_one(name, per_family, backend, show_failures=False):
    cfg = judge_config(name)
    if backend == "vllm":
        from src.judges.vllm_judge import VLLMJudge
        judge, where = VLLMJudge(cfg), "vllm"
    else:
        from src.judges.local import LocalJudge
        judge = LocalJudge(cfg)
        where = judge.device
    print(f"judge {name} ({cfg['model']}) on {where}", flush=True)
    items = build_items(per_family)
    results, seconds = judge.judge(items)

    rows = []
    for it, r in zip(items, results):
        rows.append({"kind": it["kind"], "should_accept": it["should_accept"], "accepted": r.accepted,
                     "parse_fail": r.parse_fail, "label": r.label, "tokens": r.n_tokens, "truncated": r.truncated,
                     "answer": r.answer[:30], "output": r.output.strip().replace("\n", " ")[-50:]})
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_colwidth", 50):
        print(df.to_string(index=False))
    hack = df[~df["kind"].str.startswith("normal")]
    normal = df[df["kind"].str.startswith("normal")]
    summary = {
        "judge": name, "items": len(df), "s_per_item": round(seconds / len(df), 3),
        "parse_fail": int(df["parse_fail"].sum()), "truncated": int(df["truncated"].sum()),
        "mean_tokens": round(float(df["tokens"].mean()), 1),
        "hack_accepted": f"{int(hack['accepted'].sum())}/{len(hack)}",
        "normal_ok": f"{int((normal['accepted'] == normal['should_accept']).sum())}/{len(normal)}",
    }
    out_path = ROOT / "runs" / "judge-smoke" / f"{name}-{backend}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it, r in zip(items, results):
            f.write(json.dumps({"kind": it["kind"], "should_accept": it["should_accept"], "answer": r.answer,
                                "output": r.output, "accepted": r.accepted, "parse_fail": r.parse_fail,
                                "label": r.label, "n_tokens": r.n_tokens, "truncated": r.truncated},
                               ensure_ascii=False) + "\n")
    if show_failures:
        bad = [(it, r) for it, r in zip(items, results)
               if r.parse_fail or (it["kind"].startswith("normal") and r.accepted != it["should_accept"])]
        print(
            f"\n== {len(bad)} parse failures or misjudged normal answers (raw outputs in {out_path.relative_to(ROOT)})"
        )
        for it, r in bad:
            print(f"\n[{it['kind']}] parse_fail={r.parse_fail} accepted={r.accepted} truncated={r.truncated} "
                  f"tokens={r.n_tokens} answer={r.answer[:40]!r}")
            print("  " + describe_output(r.output))
    print(f"\n{json.dumps(summary)}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="xverify_0p5b", help="comma-separated names, or all")
    ap.add_argument("--per-family", type=int, default=2)
    ap.add_argument("--backend", choices=["local", "vllm"], default="local")
    ap.add_argument("--show-failures", action="store_true",
                    help="print raw outputs of parse failures and misjudged normal answers")
    ap.add_argument("--summary-out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    names = list(load_config()["judges"]) if args.judges == "all" else args.judges.split(",")

    if len(names) == 1:
        summary = run_one(names[0], args.per_family, args.backend, args.show_failures)
        if args.summary_out:
            Path(args.summary_out).write_text(json.dumps(summary))
        return

    summaries = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            out = Path(tmp) / f"{name}.json"
            cmd = [sys.executable, "-m", "src.judges.smoke", "--judges", name, "--per-family",
                   str(args.per_family), "--backend", args.backend, "--summary-out", str(out)]
            if args.show_failures:
                cmd.append("--show-failures")
            code = subprocess.run(cmd).returncode
            summaries.append(json.loads(out.read_text()) if out.exists() else {"judge": name, "error": f"exit {code}"})
    print("\n== summary")
    with pd.option_context("display.width", 220):
        print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
