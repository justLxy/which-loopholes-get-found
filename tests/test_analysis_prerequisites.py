"""Analysis CLIs explain how to produce missing upstream outputs."""

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "name,producer",
    [
        ("confirmatory", "python -m src.analysis.primary"),
        ("masking_outcomes", "python -m src.analysis.run_level"),
        ("lowshare_outcomes", "python -m src.train.lowshare --predict"),
        ("pooled_runs", "python -m src.analysis.run_level"),
    ],
)
def test_missing_prerequisite_has_one_clear_line(name, producer, tmp_path, monkeypatch, capsys):
    module = importlib.import_module(f"src.analysis.{name}")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [name, "--analysis", str(tmp_path / "analysis")])
    if name == "lowshare_outcomes":
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--plan", str(tmp_path / "lowshare.yaml")])
        (tmp_path / "lowshare.yaml").write_text("predictions: results/lowshare/predictions.csv\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert len(captured.err.splitlines()) == 1
    assert f"run {producer} first" in captured.err
    assert "Traceback" not in captured.err
