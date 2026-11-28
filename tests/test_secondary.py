"""Secondary analyses: partial likelihoods, separation, Kaplan-Meier, growth windows and cause-specific censoring."""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.secondary import (
    add_covariates,
    cause_specific,
    first_events,
    fit,
    km,
    loglik,
    risk_sets,
    separating_ratios,
    windows,
)
from src.paths import ROOT

UNITS = ROOT / "results" / "analysis" / "units.csv"
needs_units = pytest.mark.skipif(not UNITS.exists(), reason="run src.analysis.primary first")


def toy(times, events, x, run=None):
    return pd.DataFrame({"cs_time": times, "cs_event": events, "x": x, "run": run or ["r"] * len(times)})


def test_cox_fit_matches_a_grid_search_and_the_gradient_is_right():
    u = toy([1, 2, 3, 4, 5, 6], [True, True, False, True, True, False], [0.2, 1.5, -0.3, 0.1, -1.0, 0.7])
    sets = risk_sets(u, ["x"])
    beta, ll, finite = fit(sets, 1)
    grid = np.linspace(-5, 5, 20001)
    best = grid[np.argmax([loglik(np.array([b]), sets)[0] for b in grid])]
    assert finite and beta[0] == pytest.approx(best, abs=1e-3)
    b, h = np.array([0.4]), 1e-6
    numeric = (loglik(b + h, sets)[0] - loglik(b - h, sets)[0]) / (2 * h)
    assert loglik(b, sets)[1][0] == pytest.approx(numeric, rel=1e-5)


def test_separated_data_give_a_supremum_and_no_finite_estimate():
    # in every run the family with the larger x wins: the conditional logit separates
    u = pd.concat([toy([0, 0, 0], [True, False, False], [2.0, 1.0, 0.0], ["a"] * 3),
                   toy([0, 0, 0], [False, True, False], [0.5, 3.0, 1.0], ["b"] * 3)], ignore_index=True)
    beta, ll, finite = fit(risk_sets(u, ["x"], strata="run"), 1)
    assert not finite and beta[0] > 0 and ll > -1e-6


def test_kaplan_meier_and_growth_windows():
    assert km([10, 20, 300], [True, True, False], at=(5, 15, 50, 300)) == pytest.approx([0, 1 / 3, 2 / 3, 2 / 3])
    assert km([100, 300, 300], [True, False, False], at=(50, 300)) == pytest.approx([0, 1 / 3])
    assert [list(w) for w in windows(50)] == [list(range(10)), list(range(30, 40))]
    assert list(windows(20)[1]) == list(range(10, 20)) and list(windows(30)[1]) == list(range(10, 20))
    assert list(windows(None)[1]) == list(range(40, 50))


def test_first_event_ties_go_to_the_larger_rate_and_others_are_censored(tmp_path):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "eval_summary.jsonl").write_text(json.dumps({"step": 20, "e/a": 0.1, "e/b": 0.3}) + "\n")
    u = pd.DataFrame({"run": ["r", "r", "r", "s"], "family": ["a", "b", "c", "a"], "event": [True, True, False, False],
                      "time": [20, 20, 300, 300]})
    firsts = first_events(u, tmp_path)
    assert firsts == {"r": (20, "b")}
    cs = cause_specific(u, firsts)
    assert cs["cs_time"].tolist() == [20, 20, 20, 300] and cs["cs_event"].tolist() == [False, True, False, False]


@needs_units
def test_covariates_of_the_main_runs():
    units = pd.read_csv(UNITS)
    u = add_covariates(units[units["group"] != "control"])
    assert np.isfinite(u[["log_p", "log_fpr", "log_r"]].to_numpy()).all()
    assert (u.loc[u["fpr"] == 0, "log_fpr"] == np.log(0.5 / 472)).all()


def test_separating_ratios_bracket_the_weights_that_rank_every_winner_first():
    # run a: the winner has the higher FPR, lower p; run b: the winner has the higher p, lower FPR
    g = pd.DataFrame({"run": ["a", "a", "b", "b"], "log_p": [-4.0, -2.0, -1.0, -3.0],
                      "log_fpr": [-1.0, -2.0, -2.0, -1.5], "cs_event": [True, False, True, False]})
    lo, hi = separating_ratios(g, grid=np.linspace(0, 3, 301))
    # a needs -4 lam - 1 > -2 lam - 2, i.e. lam < 0.5; b needs -lam - 2 > -3 lam - 1.5, i.e. lam > 0.25
    assert lo == pytest.approx(0.26) and hi == pytest.approx(0.49)
