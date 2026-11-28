"""Summarize a GRPO run: speed, GPU memory, judge load and the periodic evaluations.

Usage:
    python -m src.train.summary runs/grpo/<run>

scripts/run_grpo.sh prints this at the end of every run. Peak memory comes from gpu_mem.csv,
which the script samples with nvidia-smi every 2 seconds (all processes on the GPU, so the
judge server is included).

The vLLM-trainer log-prob gap (TRL's sampling/sampling_logp_difference/mean) checks the weight
sync: the trainer rescores every completion that vLLM sampled, so with synced weights the gap
stays at bf16 noise whatever the learning rate; stale vLLM weights make it grow step by step.
"""

import argparse
import json
import statistics
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def peak_memory_mib(path):
    path = Path(path)
    if not path.exists():
        return None
    values = [int(v) for v in path.read_text().split() if v.strip().isdigit()]
    return max(values) if values else None


def summarize(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
    logs = [r for r in read_jsonl(run_dir / "train_log.jsonl") if "step_time" in r]
    metrics = read_jsonl(run_dir / "train_metrics.jsonl")
    evals = read_jsonl(run_dir / "eval_summary.jsonl")
    lines = [f"run {run_dir.name}: judge {meta.get('judge')}, pool {meta.get('pool')}, seed {meta.get('seed')}, "
             f"profile {meta.get('profile')}"]

    if logs:
        times = [r["step_time"] for r in logs]
        later = times[1:] or times
        lines.append(
            f"steps {logs[-1]['step']}: step time median {statistics.median(later):.1f}s "
            f"(first step {times[0]:.1f}s), mean reward {statistics.mean(r.get('reward', 0) for r in logs):.3f}, "
            f"mean completion length {statistics.mean(r.get('completions/mean_length', 0) for r in logs):.0f}, "
            f"clipped {statistics.mean(r.get('completions/clipped_ratio', 0) for r in logs):.3f}"
        )
    if metrics:
        judge_s = [m["judge_seconds"] for m in metrics]
        lines.append(f"judge: {statistics.mean(judge_s):.1f}s per batch of {metrics[0]['n']}, parse failures "
                     f"{statistics.mean(m['parse_fail_rate'] for m in metrics):.3f}, no-signal groups "
                     f"{statistics.mean(m['groups_no_signal'] for m in metrics):.3f}")
        lines.append("train no_box rate per step: " + " ".join(f"{m['no_box_rate']:.2f}" for m in metrics))
    gaps = [r["sampling/sampling_logp_difference/mean"] for r in logs if "sampling/sampling_logp_difference/mean" in r]
    if gaps:
        lines.append("vLLM-trainer log-prob gap per step: " + " ".join(f"{g:.4f}" for g in gaps))
    peak = peak_memory_mib(run_dir / "gpu_mem.csv")
    if peak is not None:
        lines.append(f"peak GPU memory (all processes): {peak} MiB")
    if evals:
        lines.append("evaluations:  step  accept  gold   fp     exploit  no_box  e_nobox  gen_s  judge_s")
        for e in evals:
            lines.append(f"              {e['step']:>4}  {e['accept_rate']:.3f}   {e['gold_accuracy']:.3f}  "
                         f"{e['false_positive_rate']:.3f}  {e['exploit_rate']:.3f}    {e['no_box_rate']:.3f}   "
                         f"{e.get('e/no_box', 0.0):.3f}    {e['gen_seconds']:>5.0f}  {e['judge_seconds']:>6.0f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    print(summarize(ap.parse_args().run_dir))


if __name__ == "__main__":
    main()
