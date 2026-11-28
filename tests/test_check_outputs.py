"""scripts/check_outputs.py: regenerated CSVs pass within the tolerance and fail on any real change."""

import importlib.util
import subprocess

import pytest

from src.paths import ROOT

spec = importlib.util.spec_from_file_location("check_outputs", ROOT / "scripts" / "check_outputs.py")
check_outputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_outputs)

HEAD = b"run,family,auroc,hit\nr1,empty_symbols,0.9741760108732586,True\nr2,gibberish,,False\n"


def test_identical_bytes():
    assert check_outputs.compare_csv(HEAD, HEAD) == ("identical", "")


def test_last_digit_differences_are_close():
    new = HEAD.replace(b"0.9741760108732586", b"0.9741760108732581")
    status, detail = check_outputs.compare_csv(HEAD, new)
    assert status == "close" and detail.startswith("largest absolute difference ")
    assert 0 < float(detail.split()[-1]) < 1e-15          # the parsed difference varies by platform: 4.4e-16, 5.6e-16


@pytest.mark.parametrize("old,new,where", [
    (b"0.9741760108732586", b"0.9741760118732586", "column auroc, row 0"),      # 1e-9 off: a real change
    (b"True", b"False", "column hit, row 0"),
    (b"gibberish", b"simple_words", "column family, row 1"),
    (b"r2,gibberish,,False", b"r2,gibberish,0.5,False", "column auroc, row 1"),  # empty cell filled
])
def test_real_changes_are_different(old, new, where):
    status, detail = check_outputs.compare_csv(HEAD, HEAD.replace(old, new))
    assert status == "different" and detail.startswith(where)


def test_changed_columns_and_rows_are_different():
    assert check_outputs.compare_csv(HEAD, HEAD.replace(b"auroc", b"auc"))[0] == "different"
    assert check_outputs.compare_csv(HEAD, HEAD + b"r3,gibberish,0.5,True\n")[1] == "2 rows in HEAD, 3 now"


def test_the_committed_outputs_pass_and_stale_figures_fail(capsys):
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "results/analysis", "figures"], cwd=ROOT).returncode:
        pytest.skip("results/analysis or figures has local changes")
    newest = max((ROOT / p).stat().st_mtime for p in check_outputs.committed(check_outputs.PDF_GLOB))
    assert check_outputs.check(since=None) == 0
    assert check_outputs.check(since=newest + 1) == len(check_outputs.committed(check_outputs.PDF_GLOB))
    assert "not rewritten by this run" in capsys.readouterr().out
