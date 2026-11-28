from pathlib import Path

import pytest

from src.analysis.stats import wilson_interval
from src.audit.fpr import (
    HUANG_FAMILIES,
    MASTER_KEY_OPENERS,
    PLACEHOLDER,
    audit_one,
    build_items,
    judge_overview,
    report,
    summarize,
)
from src.judges.local import JudgeResult
from src.judges.parse import parse_verdict
from src.judges.prompts import judge_answer
from src.pipeline.records import validate_audit_record
from src.pipeline.shards import read_shards

HACKING = Path(__file__).resolve().parents[1] / "data" / "raw" / "hacking" / "hacking_data.parquet"
needs_data = pytest.mark.skipif(not HACKING.exists(), reason="run src.data.download_hacking first")


def test_wilson_interval():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and 0.03 < hi < 0.04
    lo, hi = wilson_interval(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert wilson_interval(100, 100)[1] == pytest.approx(1.0)


@needs_data
def test_full_test_set():
    items = build_items()
    assert len(items) == 471 * 15 and len({it["key"] for it in items}) == len(items)
    families = {it["family"] for it in items}
    assert families == set(HUANG_FAMILIES) | {"reasoning_opener", PLACEHOLDER}
    assert sum(it["integer_reference"] for it in items) == 263 * 15
    # every answer is detected as the family it audits (the placeholder is not real gibberish)
    mismatched = [it for it in items if it["family"] != PLACEHOLDER and it["detected_family"] != it["family"]]
    assert mismatched == []
    gib = [it for it in items if it["family"] == "gibberish"]
    assert all(len(it["response"]) == 1000 for it in gib) and len({it["response"] for it in gib}) == 471
    assert {it["response"] for it in items if it["family"] == "reasoning_opener"} <= set(MASTER_KEY_OPENERS)
    assert {it["response"] for it in items if it["family"] == PLACEHOLDER} == {"random_gibberish_1000chars"}


@needs_data
def test_subset_is_deterministic():
    a, b = build_items(5), build_items(5)
    assert len(a) == 5 * 15 and a == b


class FakeJudge:
    """Accepts empty symbols and simple words, rejects the rest, through the real parser."""

    def judge(self, items):
        out = []
        for it in items:
            answer = judge_answer(it["response"])
            text = r"\boxed{1}" if len(answer) <= 1 or answer in {"Correct", "Yes", "True"} else r"\boxed{0}"
            v = parse_verdict({"parser": "boxed01", "reasoning": False}, text)
            out.append(JudgeResult(answer, text, v.accepted, v.parse_fail, v.label, 5, False))
        return out, 0.0


@needs_data
def test_audit_run_resume_and_report(tmp_path, capsys):
    items = build_items(4)
    run_dir = tmp_path / "audit"
    assert audit_one("fake", items, FakeJudge(), run_dir / "fake", shard_size=25, log=lambda m: None) == len(items)
    assert audit_one("fake", items, FakeJudge(), run_dir / "fake", shard_size=25, log=lambda m: None) == 0
    records = read_shards(run_dir / "fake")
    assert len(records) == len(items)
    for r in records:
        validate_audit_record(r)
    rows = {r["family"]: r for r in summarize(records)}
    assert rows["gibberish"]["fpr"] == 0.0 and rows["answer_explanation"]["fpr"] == 0.0
    assert 0 < rows["empty_symbols"]["fpr"] <= 1
    report(run_dir)
    out = capsys.readouterr().out
    assert "per judge" in out and "FPR (%) by family" in out
    assert (run_dir / "fpr_by_family.csv").exists() and (run_dir / "fpr_overview.json").exists()


def test_overview_means():
    def row(family, fpr, n=10):
        return {"judge": "j", "family": family, "fpr": fpr, "n": n, "parse_fail": 0.0, "truncated": 0.0,
                "mean_tokens": 10.0}
    rows = [row(f, 0.0) for f in HUANG_FAMILIES if f != "gibberish"] + [row("gibberish", 0.13), row(PLACEHOLDER, 0.26),
                                                                        row("reasoning_opener", 0.5)]
    (o,) = judge_overview(rows)
    assert o["mean13_real_%"] == round(100 * 0.13 / 13, 1) and o["mean13_placeholder_%"] == round(100 * 0.26 / 13, 1)
    assert o["reasoning_opener_%"] == 50.0 and o["huang_reported_%"] is None
