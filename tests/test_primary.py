"""The primary analysis: AUROC with ties, the run bootstrap within judges, and the units of the main runs."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.primary import (
    SCORES,
    analyse,
    analyses,
    auroc,
    average_ranks,
    bootstrap,
    load_units,
    paired_diff,
    two_sided_p,
)
from src.detectors import FAMILIES
from src.paths import ROOT

MAIN = ROOT / "results" / "main"
needs_main = pytest.mark.skipif(not MAIN.exists(), reason="needs results/main from the main experiment")


def test_auroc_counts_ties_as_one_half_and_skips_missing_scores():
    assert average_ranks(np.array([3.0, 1.0, 3.0])).tolist() == [2.5, 1.0, 2.5]
    assert auroc([0.9, 0.5, 0.1], [True, False, False]) == 1.0
    assert auroc([0.1, 0.5, 0.9], [True, False, False]) == 0.0
    assert auroc([0.5, 0.5, 0.5], [True, False, False]) == 0.5           # a constant score
    assert auroc([0.9, 0.5, 0.5], [True, True, False]) == 0.75
    assert auroc([0.9, np.nan, 0.1], [True, True, False]) == 1.0           # the nan unit is left out
    assert np.isnan(auroc([0.1, 0.2], [False, False]))
    assert paired_diff([0.9, 0.5, 0.1], [0.1, np.nan, 0.9], [True, False, False]) == 1.0


def toy_units():
    """Two judges with two runs each; in every run the event is on the family of highest risk."""
    rows = []
    for j in ("a", "b"):
        for s in (0, 1):
            for i, f in enumerate(FAMILIES[:4]):
                rows.append({"run": f"{j}{s}", "judge": j, "group": "core", "family": f, "event": i == 3,
                             "risk": i, "fpr": (i + s) % 4, "p_hat": 3 - i, "static": 0.0 if j == "a" else 1.0})
    return pd.DataFrame(rows)


def test_bootstrap_resamples_whole_runs_within_judges():
    units = toy_units()
    aurocs, diffs = bootstrap(units, 200, seed=1)
    assert aurocs.shape == (200, len(SCORES)) and diffs.shape == (200, len(SCORES) - 1)
    assert np.all(aurocs[:, 0] == 1.0)                       # the risk score is perfect in every replicate
    assert np.all(aurocs[:, 2] == 0.0)                       # propensity is reversed in every replicate
    assert np.all(diffs[:, 1] == 1.0)
    rows = analyse(units, "toy", 200, seed=1)
    risk = rows[0]
    assert risk["auroc"] == 1.0 and risk["ci_low"] == 1.0 and risk["runs_with_event"] == 4
    static = rows[3]
    assert static["auroc"] == 0.5 and static["p_two_sided"] == 0.0
    assert two_sided_p(np.array([0.1, -0.1, 0.2, 0.3])) == 0.5


@needs_main
def test_units_of_the_main_runs():
    units = load_units(MAIN)
    assert len(units) == 36 * len(FAMILIES)
    assert units.groupby("group")["run"].nunique().to_dict() == {"control": 3, "core": 24, "held_out": 9}
    assert units[["risk", "fpr", "p_hat"]].notna().all().all()
    assert units["risk"].to_numpy() == pytest.approx((units["fpr"] * units["p_hat"]).to_numpy())
    assert units.loc[units["judge"] == "qwen3p6_35b_a3b_fp8", "static"].isna().all()   # not measured in stage 1
    assert not units.loc[units["group"] == "control", "event"].any()
    labels = [label for label, _ in analyses(units, "r1_distill_verifier_1p5b")]
    assert labels[:2] == ["core pooled", "core pooled without the positive control"]
    assert sum(len(g) for label, g in analyses(units, "r1_distill_verifier_1p5b")[2:6]) == 24 * len(FAMILIES)
