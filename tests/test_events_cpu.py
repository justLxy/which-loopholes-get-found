import json
import sys

import pytest

from src.monitor import events


def test_empty_evaluations_have_no_step_zero_baseline():
    with pytest.raises(ValueError, match="no evaluations"):
        events.detect_events([], tau=0.05)


def test_events_cli_records_the_first_crossing_and_censors_missing_families(tmp_path, monkeypatch, capsys):
    rows = [
        {"step": 0, "e/gibberish": 0.02},
        {"step": 10, "e/gibberish": 0.07},
        {"step": 20, "e/gibberish": 0.10},
        {"step": 30, "e/gibberish": 0.03},
    ]
    (tmp_path / "eval_summary.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(sys, "argv", ["events", str(tmp_path), "--tau", "0.05"])

    events.main()

    result = json.loads((tmp_path / "events_tau0.05.json").read_text())
    assert result["tau"] == 0.05 and result["consecutive"] == 2
    # The increases at steps 10 and 20 are 0.05 and 0.08: the first event is at 10.
    assert result["events"]["gibberish"] == {
        "event": True, "time": 10, "baseline": 0.02, "peak": 0.10,
    }
    assert result["events"]["reasoning_opener"] == {
        "event": False, "time": 30, "baseline": 0.0, "peak": 0.0,
    }
    output = capsys.readouterr().out
    assert "event at step 10" in output and "censored at 30" in output
