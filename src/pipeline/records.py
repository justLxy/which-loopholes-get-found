"""Record formats for policy outputs, plus run metadata.

Three kinds of JSONL lines:
  sample record  one policy response with its gold label and detected family
                 (difficulty run, propensity audit)
  judge record   a sample record plus one judge's verdict, one line per
                 (response, judge) (static accuracy, GRPO evaluation)
  audit record   one judge's verdict on a hacking answer that is wrong by
                 construction (judge false-positive audit)
Raw responses and raw judge outputs are kept in full so any label can be
re-derived or hand-audited later.

Derived fields:
  false_positive  judge accepted a response that the gold checker marks wrong
  exploit         false_positive and the response belongs to a hacking family;
                  e_f(t) is the share of evaluation outputs with exploit and family f
"""

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

SCHEMA_VERSION = 2   # 2: adds request_seed and separate sample records
ROOT = Path(__file__).resolve().parents[2]

SAMPLE_FIELDS = {
    "schema_version": int, "run_id": str, "stage": str, "step": (int, type(None)),
    "uid": str, "source": str, "reference": str, "sample_idx": int, "request_seed": (int, type(None)),
    "policy_model": str, "response": str, "n_tokens": int, "truncated": bool,
    "gold_correct": bool, "gold_extracted": (str, type(None)), "no_answer": bool, "n_boxed": int,
    "family": (str, type(None)), "contains": list,
}
JUDGE_FIELDS = {
    "judge": str, "judge_answer": str, "judge_output": str, "accepted": bool, "parse_fail": bool,
    "judge_label": (str, type(None)),
    "false_positive": bool, "exploit": bool,
}
FIELDS = {**SAMPLE_FIELDS, **JUDGE_FIELDS}


def make_sample_record(*, run_id, stage, step, item, sample_idx, policy_model, sample, gold, detection,
                       request_seed=None):
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "stage": stage, "step": step,
        "uid": item["uid"], "source": item["source"], "reference": item["reference"], "sample_idx": sample_idx,
        "request_seed": request_seed, "policy_model": policy_model, "response": sample.text,
        "n_tokens": sample.n_tokens, "truncated": sample.truncated,
        "gold_correct": gold.correct, "gold_extracted": gold.extracted, "no_answer": gold.no_answer,
        "n_boxed": gold.n_boxed,
        "family": detection.family, "contains": sorted(detection.contains),
    }


def make_record(*, judge_name, verdict, **sample_kwargs):
    """Judge record: the sample record of a response plus one judge's verdict."""
    r = make_sample_record(**sample_kwargs)
    false_positive = verdict.accepted and not r["gold_correct"]
    r.update({
        "judge": judge_name, "judge_answer": verdict.answer, "judge_output": verdict.output,
        "accepted": verdict.accepted, "parse_fail": verdict.parse_fail, "judge_label": verdict.label,
        "false_positive": false_positive, "exploit": false_positive and r["family"] is not None,
    })
    return r


def judge_record_from_sample(sample_record, *, run_id, stage, judge_name, verdict):
    """Judge record for a response already stored as a sample record (for example from the difficulty run)."""
    r = {k: sample_record[k] for k in SAMPLE_FIELDS}
    r.update({"schema_version": SCHEMA_VERSION, "run_id": run_id, "stage": stage})
    false_positive = verdict.accepted and not r["gold_correct"]
    r.update({
        "judge": judge_name, "judge_answer": verdict.answer, "judge_output": verdict.output,
        "accepted": verdict.accepted, "parse_fail": verdict.parse_fail, "judge_label": verdict.label,
        "false_positive": false_positive, "exploit": false_positive and r["family"] is not None,
    })
    return r


def _check_fields(r, fields):
    missing = set(fields) - set(r)
    extra = set(r) - set(fields)
    if missing or extra:
        raise ValueError(f"missing {sorted(missing)}, unexpected {sorted(extra)}")
    for key, typ in fields.items():
        if not isinstance(r[key], typ) or (typ is int and isinstance(r[key], bool)):
            raise ValueError(f"{key} has type {type(r[key]).__name__}")


def _check_family(r):
    if r["family"] is not None and r["family"] not in r["contains"]:
        raise ValueError("primary family missing from contains")


def validate_sample_record(r):
    """Raise ValueError if a sample record is malformed."""
    _check_fields(r, SAMPLE_FIELDS)
    _check_family(r)


def validate_record(r):
    """Raise ValueError if a judge record is malformed or its flags are inconsistent."""
    _check_fields(r, FIELDS)
    _check_family(r)
    if r["false_positive"] != (r["accepted"] and not r["gold_correct"]):
        raise ValueError("false_positive inconsistent")
    if r["exploit"] != (r["false_positive"] and r["family"] is not None):
        raise ValueError("exploit inconsistent")
    if r["parse_fail"] and r["accepted"]:
        raise ValueError("a parse failure cannot be accepted")


AUDIT_FIELDS = {
    "schema_version": int, "run_id": str, "stage": str, "key": str, "qid": str, "family": str,
    "detected_family": (str, type(None)), "integer_reference": bool, "reference": str, "response": str,
    "judge": str, "judge_answer": str, "judge_output": str, "accepted": bool, "parse_fail": bool,
    "judge_label": (str, type(None)), "judge_tokens": int, "judge_truncated": bool,
}


def make_audit_record(*, run_id, item, judge_name, verdict):
    """Audit record: a judge's verdict on a hacking answer (wrong by construction)."""
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "stage": "judge_audit", "key": item["key"],
        "qid": item["qid"], "family": item["family"], "detected_family": item["detected_family"],
        "integer_reference": item["integer_reference"], "reference": item["reference"],
        "response": item["response"], "judge": judge_name, "judge_answer": verdict.answer,
        "judge_output": verdict.output, "accepted": verdict.accepted, "parse_fail": verdict.parse_fail,
        "judge_label": verdict.label, "judge_tokens": verdict.n_tokens, "judge_truncated": verdict.truncated,
    }


def validate_audit_record(r):
    _check_fields(r, AUDIT_FIELDS)
    if r["parse_fail"] and r["accepted"]:
        raise ValueError("a parse failure cannot be accepted")


def write_jsonl(path, records, validate=validate_record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            validate(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def git_state():
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def package_versions(names=("torch", "transformers", "datasets", "math-verify", "vllm", "trl")):
    found = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def write_meta(run_dir, name="meta.json", **extra):
    """Run metadata: time, git state, configs and library versions, next to the records."""
    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "configs": {p.name: p.read_text() for p in sorted((ROOT / "configs").glob("*.yaml"))},
        "versions": {"python": platform.python_version(), **package_versions()},
        **extra,
    }
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / name).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta
