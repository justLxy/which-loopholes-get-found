import json
import sys

import pandas as pd
import pytest

from src.audit import string_fpr
from src.detectors import FAMILIES
from src.judges.local import JudgeResult
from src.pipeline.records import make_audit_record, validate_audit_record


@pytest.fixture
def hacking_table(tmp_path, monkeypatch):
    path = tmp_path / "hacking.parquet"
    pd.DataFrame([
        {"hacking_type": f"tiger_{family}", "pred": "{" if family == "empty_symbols" else family}
        for family in FAMILIES
    ]).to_parquet(path, index=False)
    monkeypatch.setattr(string_fpr, "HACKING", path)
    return path


def audit_record(judge, number, accepted, *, integer=True, parse_fail=False, truncated=False, tokens=4):
    qid = f"{judge}-{number}"
    item = {
        "key": f"{qid}:empty_symbols", "qid": qid, "question": "What is 2+2?",
        "family": "empty_symbols", "detected_family": "empty_symbols",
        "integer_reference": integer, "reference": "4" if integer else "1/2", "response": "{",
    }
    verdict = JudgeResult(
        "{", "yes" if accepted else "no", accepted, parse_fail,
        None if parse_fail else "yes" if accepted else "no", n_tokens=tokens, truncated=truncated,
    )
    record = make_audit_record(run_id="audit", item=item, judge_name=judge, verdict=verdict)
    validate_audit_record(record)
    return record


def test_seeding_string_uses_frequency_then_alphabetical_tie_break(hacking_table):
    frame = pd.read_parquet(hacking_table)
    extra = {"hacking_type": "tiger_empty_symbols", "pred": "}"}
    pd.concat([frame, pd.DataFrame([extra])], ignore_index=True).to_parquet(hacking_table, index=False)
    assert string_fpr.seeding_strings()["empty_symbols"] == "{"  # one of each: alphabetical tie-break
    pd.concat([frame, pd.DataFrame([extra, extra])], ignore_index=True).to_parquet(hacking_table, index=False)
    assert string_fpr.seeding_strings()["empty_symbols"] == "}"  # two occurrences beat one


def test_empty_string_audit_reports_no_data_without_writing_a_table(tmp_path, capsys):
    assert string_fpr.report(tmp_path) is None
    assert "no audit records yet" in capsys.readouterr().out
    assert not (tmp_path / "string_fpr.csv").exists()


def test_summary_cli_aggregates_shards_and_keeps_judges_separate(tmp_path, monkeypatch, hacking_table, capsys):
    run_dir = tmp_path / "audit"
    records = [
        audit_record("j", 0, True, tokens=4),
        audit_record("j", 1, True, tokens=8, truncated=True),
        audit_record("j", 2, False, tokens=12, parse_fail=True),
        audit_record("j", 3, True, integer=False, tokens=16, truncated=True),
        audit_record("other", 0, False, integer=False, tokens=6),
        audit_record("other", 1, True, integer=False, tokens=6),
    ]
    for judge in ("j", "other"):
        directory = run_dir / judge
        directory.mkdir(parents=True)
        group = [record for record in records if record["judge"] == judge]
        for number, chunk in enumerate((group[:2], group[2:])):
            if chunk:
                (directory / f"shard-{number:05d}.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in chunk), encoding="utf-8",
                )
    stage1 = tmp_path / "results" / "stage1"
    stage1.mkdir(parents=True)
    pd.DataFrame([
        {"judge": "j", "family": "empty_symbols", "fpr": 0.25},
        {"judge": "other", "family": "empty_symbols", "fpr": 0.5},
    ]).to_csv(stage1 / "fpr_by_family.csv", index=False)
    monkeypatch.setattr(string_fpr, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "string_fpr", "--run-dir", str(run_dir), "--judges", "j,other", "--summary-only",
    ])

    string_fpr.main()

    table = pd.read_csv(run_dir / "string_fpr.csv").set_index("judge")
    assert len(table) == 2
    row = table.loc["j"]
    assert row["family"] == "empty_symbols" and row["string"] == "{"
    assert row["n"] == 4 and row["accepted"] == 3 and row["fpr"] == 0.75
    assert row["n_integer"] == 3 and row["fpr_integer"] == pytest.approx(2 / 3)
    assert row["parse_fail"] == 0.25 and row["truncated"] == 0.5 and row["mean_tokens"] == 10
    assert row["ci_low"] < 0.75 < row["ci_high"]
    assert table.loc["other", "n"] == 2 and table.loc["other", "fpr"] == 0.5
    assert table.loc["other", "n_integer"] == 0 and pd.isna(table.loc["other", "fpr_integer"])
    output = capsys.readouterr().out
    assert "difference from the family's mean FPR" in output and "50.0" in output  # 75% minus 25%
