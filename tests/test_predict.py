"""Audit your own judge (src/audit/predict.py): ranking, validation, the timing window and the command line."""

import math
import re

import numpy as np
import pandas as pd
import pytest

from src.audit.predict import (
    PredictError,
    check_values,
    main,
    measured_inputs,
    rank_patterns,
    read_values,
    timing,
    window,
)
from src.detectors import FAMILIES
from src.paths import ROOT
from src.train.masking import Z90, timing_fit

MAIN = ROOT / "results" / "main"
needs_results = pytest.mark.skipif(not (ROOT / "results" / "stage2" / "string_fpr.csv").exists()
                                   or not MAIN.exists(), reason="needs results/stage2 and results/main")


def uniform(value):
    return {f: value for f in FAMILIES}


def simple():
    """Exposure ~ (i + 1) (14 - i)^2: no ties; top three are i = 4, 5, 3."""
    n = len(FAMILIES)
    return ({f: (i + 1) / n for i, f in enumerate(FAMILIES)},
            {f: (n - i) ** 2 / (10 * n * n) for i, f in enumerate(FAMILIES)})


# ----------------------------------------------------------------------------- ranking

def test_exposure_and_the_three_rankings():
    fpr, p = simple()
    r = rank_patterns(fpr, p)
    t = r.table.set_index("family")
    assert t["exposure"].to_numpy() == pytest.approx([256 * p[f] * fpr[f] for f in t.index], abs=1e-12)
    assert list(r.table["exposure"]) == sorted(r.table["exposure"], reverse=True)
    assert list(r.table["rank"]) == list(range(1, 15))
    assert t.loc[FAMILIES[-1], "rank_fpr"] == 1 and t.loc[FAMILIES[0], "rank_p_hat"] == 1
    assert "ea" not in r.table and r.q0 is None
    assert list(r.table["family"][:3]) == [FAMILIES[4], FAMILIES[5], FAMILIES[3]] and r.top == FAMILIES[4]


def test_ties_share_the_lowest_rank_and_the_top():
    r = rank_patterns(uniform(0.5), uniform(0.02))
    assert set(r.table["rank"]) == {1} and r.top == "+".join(sorted(FAMILIES))
    assert r.exposure == pytest.approx(256 * 0.5 * 0.02)


def test_expected_advantage_by_hand():
    fpr, p = uniform(0.1), uniform(0.01)
    fpr["empty_symbols"], p["empty_symbols"] = 0.6, 0.04
    r = rank_patterns(fpr, p, q0=0.2)
    t = r.table.set_index("family")
    assert t.loc["empty_symbols", "ea"] == pytest.approx(0.04 * 0.4 / math.sqrt(0.16))       # 0.04
    assert t.loc["gibberish", "ea"] == pytest.approx(0.01 * -0.1 / 0.4)                       # -0.0025
    assert t.loc["empty_symbols", "rank_ea"] == 1 and r.q0 == 0.2


def test_the_window_is_the_timing_fit_of_the_main_runs():
    a, b, sd = timing_fit(ROOT / "results" / "analysis" / "event_timing.csv")
    (fit, level) = timing()
    assert fit == (a, b, sd) and level == 1.92
    assert (round(a, 2), round(b, 2), round(sd, 2)) == (5.64, -1.03, 0.38)                   # paper Appendix D
    step, low, high = window(3.33, fit)
    assert step == pytest.approx(math.exp(a + b * math.log(3.33)))
    assert (low, high) == pytest.approx((step * math.exp(-Z90 * sd), step * math.exp(Z90 * sd)))
    assert (round(step), round(low), round(high)) == (82, 44, 153)                            # Table 5, F1 seed 0


def test_the_prediction_of_the_top_family_and_the_level():
    fpr, p = uniform(0.0), uniform(0.0)
    fpr["reasoning_opener"], p["reasoning_opener"] = 0.5, 0.02                               # exposure 2.56
    r = rank_patterns(fpr, p)
    assert r.top == "reasoning_opener" and r.event_predicted
    assert (r.step, r.step_low, r.step_high) == pytest.approx(window(2.56, timing()[0]))
    p["reasoning_opener"] = 0.01                                                              # exposure 1.28
    assert not rank_patterns(fpr, p).event_predicted
    zero = rank_patterns(uniform(0.0), uniform(0.3))
    assert zero.exposure == 0 and zero.step is None and not zero.event_predicted


@needs_results
@pytest.mark.parametrize("checkpoint", [p.parent.name.removeprefix("sft-")
                                        for p in sorted(MAIN.glob("sft-*/risk_scores.csv"))])
def test_measured_inputs_reproduce_the_committed_risk_scores(checkpoint):
    risk = pd.read_csv(MAIN / f"sft-{checkpoint}" / "risk_scores.csv")
    risk = risk[risk["family"].isin(FAMILIES)]
    for (judge, pool), g in risk.groupby(["judge", "pool"]):
        r = rank_patterns(*measured_inputs(judge, checkpoint, pool))
        got = r.table.set_index("family")
        expected = g.set_index("family").reindex(got.index)
        assert np.abs(got["exposure"] - 256 * expected["risk"]).max() <= 1e-9, (judge, pool)
        # ranks may swap only within exact ties (e.g. 1.5 x 3/471 = 0.5 x 9/471 in main-clean1)
        committed = expected["rank_in_judge_pool"].astype(int)
        for f in got.index[got["rank"] != committed]:
            twins = got.index[np.isclose(got["exposure"], got.loc[f, "exposure"], rtol=1e-12, atol=0)]
            first = int(got.loc[twins, "rank"].min())
            assert len(twins) > 1 and set(committed[twins]) == set(range(first, first + len(twins))), (judge, pool, f)


@needs_results
def test_the_acceptance_example():
    r = rank_patterns(*measured_inputs("r1_distill_verifier_1p5b", "main-rot1", "hard"))
    committed = pd.read_csv(MAIN / "sft-main-rot1" / "risk_scores.csv")
    committed = committed[(committed["judge"] == "r1_distill_verifier_1p5b") & (committed["pool"] == "hard")]
    assert list(r.table["family"]) == list(committed.sort_values("rank_in_judge_pool")["family"])
    assert r.top == "reasoning_opener" and round(r.step) == 39 and r.event_predicted     # the run's event: step 40


# ----------------------------------------------------------------------------- validation

@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(empty_symbols=1.2), "between 0 and 1: empty_symbols (1.2)"),
    (lambda d: d.update(empty_symbols=-0.1), "between 0 and 1: empty_symbols (-0.1)"),
    (lambda d: d.update(empty_symbols="abc"), "not a number for empty_symbols ('abc')"),
    (lambda d: d.update(empty_symbols=float("nan")), "not a number for empty_symbols"),
    (lambda d: d.update(not_a_family=0.1), "unknown families: not_a_family"),
    (lambda d: d.pop("gibberish"), "missing families: gibberish"),
])
def test_each_invalid_value_is_named(change, message):
    values = uniform(0.1)
    change(values)
    with pytest.raises(PredictError, match="^FPR: .*" + re.escape(message)):
        rank_patterns(values, uniform(0.1))


def test_duplicated_families_and_bad_q0():
    s = pd.Series([0.1] * 15, index=list(FAMILIES) + ["gibberish"])
    with pytest.raises(PredictError, match="more than once: gibberish"):
        check_values(s, "FPR")
    for q0, message in [(0, "strictly between 0 and 1"), (1.0, "strictly between"), ("x", "must be a number")]:
        with pytest.raises(PredictError, match=message):
            rank_patterns(uniform(0.1), uniform(0.1), q0=q0)


def test_csv_errors(tmp_path):
    with pytest.raises(PredictError, match="no such file"):
        read_values(tmp_path / "none.csv", "fpr")
    (tmp_path / "cols.csv").write_text("name,fpr\na,0.1\n")
    with pytest.raises(PredictError, match="lacks the columns family"):
        read_values(tmp_path / "cols.csv", "fpr")
    (tmp_path / "ragged.csv").write_text('family,fpr\n"unclosed,0.1\n')
    with pytest.raises(PredictError, match="cannot be read as CSV"):
        read_values(tmp_path / "ragged.csv", "fpr")
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(PredictError, match="cannot be read as CSV"):
        read_values(tmp_path / "empty.csv", "fpr")


@needs_results
def test_unknown_judge_checkpoint_and_pool():
    with pytest.raises(PredictError, match="unknown judge 'nope'; choose one of .*r1_distill_verifier_1p5b"):
        measured_inputs("nope", "main-rot1")
    with pytest.raises(PredictError, match="unknown checkpoint 'rot9'; choose one of .*main-rot1"):
        measured_inputs("r1_distill_verifier_1p5b", "rot9")
    with pytest.raises(PredictError, match="no propensity for the 'medium' pool"):
        measured_inputs("r1_distill_verifier_1p5b", "main-rot1", "medium")


# ----------------------------------------------------------------------------- command line

def write_inputs(tmp_path, fpr, p):
    pd.DataFrame({"family": list(fpr), "fpr": list(fpr.values())}).to_csv(tmp_path / "fpr.csv", index=False)
    pd.DataFrame({"family": list(p), "p_hat": list(p.values())}).to_csv(tmp_path / "p.csv", index=False)
    return str(tmp_path / "fpr.csv"), str(tmp_path / "p.csv")


def test_cli_with_your_own_csvs(tmp_path, capsys):
    fpr, p = simple()
    f, pp = write_inputs(tmp_path, fpr, p)
    out = tmp_path / "ranking.csv"
    assert main(["--fpr", f, "--propensity", pp, "--q0", "0.2", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Top family:" in printed and "90% window" in printed
    written = pd.read_csv(out)
    expected = rank_patterns(fpr, p, 0.2).table
    assert list(written["family"]) == list(expected["family"])
    assert written["exposure"].to_numpy() == pytest.approx(expected["exposure"].to_numpy(), rel=1e-12)
    assert "ea" in written


@needs_results
def test_cli_with_our_measurements(capsys):
    assert main(["--judge", "r1_distill_verifier_1p5b", "--checkpoint", "main-rot1", "--pool", "hard"]) == 0
    printed = capsys.readouterr().out
    assert printed.splitlines()[1].split()[0] == "reasoning_opener"
    assert "Predicted first event at step 39, 90% window 21 to 73" in printed


def test_cli_errors_are_one_line(tmp_path, capsys):
    fpr, p = simple()
    fpr["empty_symbols"] = 3
    f, pp = write_inputs(tmp_path, fpr, p)
    assert main(["--fpr", f, "--propensity", pp]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: FPR: values must be between 0 and 1: empty_symbols (3)") and err.count("\n") == 1
    assert main(["--fpr", f, "--propensity", str(tmp_path / "none.csv")]) == 2
    assert "no such file" in capsys.readouterr().err
    assert main(["--judge", "nope", "--checkpoint", "main-rot1"]) == 2
    assert "unknown judge" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["--fpr", "a.csv"], ["--judge", "x"],
                                  ["--fpr", "a.csv", "--propensity", "b.csv", "--judge", "x", "--checkpoint", "y"]])
def test_cli_needs_exactly_one_input_mode(argv, capsys):
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2 and "Traceback" not in capsys.readouterr().err
