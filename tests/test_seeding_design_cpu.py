import math
import sys

import pandas as pd
import pytest

from src.analysis import seeding_design as design
from src.detectors import FAMILIES
from src.train.seed_data import tier_shares


def test_winner_outcome_is_selected_separately_for_each_seed():
    # Exposures are exactly 8, 4, 4, 2. A strict threshold of 2 keeps the first three.
    units = [
        {"seed": 0, "p": 0.125, "fpr": 0.25},
        {"seed": 0, "p": 0.0625, "fpr": 0.25},
        {"seed": 1, "p": 0.0625, "fpr": 0.25},
        {"seed": 1, "p": 0.03125, "fpr": 0.25},
    ]
    assert design.evaluate(units, threshold=2)["positives"] == 3
    winner = design.evaluate(units, threshold=2, outcome="winner")
    assert winner["positives"] == 2 and winner["units"] == 4
    assert winner["auc_fpr"] == 0.5  # all FPRs tie
    assert winner["auc_p"] == 0.875  # three wins plus one half-tie, divided by four pairs
    assert winner["rho_r_p"] == pytest.approx(1)
    assert math.isnan(winner["rho_r_fpr"])


@pytest.mark.parametrize("labels", [[], [True, True], [False, False]])
def test_auroc_is_undefined_without_both_classes(labels):
    assert math.isnan(design.auroc([0.2] * len(labels), labels))


def test_design_cli_removes_measured_rotation_and_skips_a_negligible_judge(tmp_path, monkeypatch, capsys):
    tiers = [0.01, 0.02, 0.04]
    shares = tier_shares(tiers, 1, 0)
    prop = tmp_path / "prop.csv"
    pd.DataFrame([
        {"pool": pool, "family": family, "p_hat": shares[family] if pool == "hard" else 0.9}
        for pool in ("hard", "easy") for family in FAMILIES
    ]).to_csv(prop, index=False)
    fpr = tmp_path / "fpr.csv"
    pd.DataFrame([
        {"judge": judge, "family": family,
         "fpr": 0.005 if judge == "quiet" else 0.1 if family == "empty_symbols" else 0.0}
        for judge in ("active", "quiet") for family in FAMILIES
    ]).to_csv(fpr, index=False)
    monkeypatch.setattr(sys, "argv", [
        "seeding_design", "--propensity", str(prop), "--fpr", str(fpr),
        "--pool", "hard", "--thresholds", "0.5", "--tiers", "0.01", "0.02", "0.04",
        "--propensity-rotation", "1",
    ])

    design.main()

    output = capsys.readouterr().out
    assert "relative learnability 1.00 to 1.00" in output
    assert "skip quiet: no family above FPR 1%" in output
    rows = [line.split() for line in output.splitlines() if line.strip().startswith("active ")]
    # After removing rotation, uniform has one positive. Rotated exposures .256, .512, 1.024 give two.
    assert [(row[2], int(row[-2]), int(row[-1])) for row in rows] == [
        ("uniform", 1, 14), ("rotated", 2, 42),
    ]
