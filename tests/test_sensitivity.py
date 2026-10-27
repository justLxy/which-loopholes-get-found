"""Sensitivity variants of the primary test: event definitions, likelihood propensity, family-mean FPR."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.sensitivity import events_for, variants, with_family_mean_fpr, with_likelihood, with_seeded_dose
from src.train.seed_data import tier_shares
from src.paths import ROOT

UNITS = ROOT / "results" / "analysis" / "units.csv"
MAIN = ROOT / "results" / "main"
needs_data = pytest.mark.skipif(not (UNITS.exists() and MAIN.exists()), reason="needs results/main and units.csv")


def test_family_mean_fpr_replaces_the_string_fpr_and_drops_unaudited_judges(tmp_path):
    units = pd.DataFrame({"judge": ["a", "a", "b"], "family": ["f", "g", "f"], "p_hat": [0.1, 0.2, 0.1],
                          "fpr": [0.9, 0.9, 0.9], "risk": [0.09, 0.18, 0.09]})
    csv = tmp_path / "fpr.csv"
    pd.DataFrame({"judge": ["a", "a"], "family": ["f", "g"], "fpr": [0.5, 0.1]}).to_csv(csv, index=False)
    u = with_family_mean_fpr(units, csv)
    assert list(u["judge"]) == ["a", "a"] and u["fpr"].tolist() == [0.5, 0.1]
    assert u["risk"].tolist() == pytest.approx([0.05, 0.02])


def test_seeded_dose_follows_the_rotation_of_the_run():
    tiers = [0.003, 0.012, 0.04]
    fams = list(tier_shares(tiers, 0, 0))
    units = pd.DataFrame({"seed": [0, 1, 2] * 2, "family": [fams[0]] * 3 + [fams[-1]] * 3, "fpr": [0.5] * 6,
                          "p_hat": [0.9] * 6, "risk": [0.45] * 6})
    u = with_seeded_dose(units, tiers, 0)
    expect = [tier_shares(tiers, s, 0)[f] for s, f in zip(units["seed"], units["family"])]
    assert u["p_hat"].tolist() == expect and sorted(set(expect)) == tiers
    assert u["risk"].tolist() == pytest.approx([0.5 * d for d in expect])
    assert units["p_hat"].tolist() == [0.9] * 6                     # the input is left unchanged


@needs_data
def test_hard_pool_variant_keeps_every_event():
    units = pd.read_csv(UNITS)
    units = units[units["group"] != "control"].reset_index(drop=True)
    hard = dict(variants(units, MAIN))["runs: hard pool only"]
    assert set(hard["pool"]) == {"hard"} and hard["event"].sum() == units["event"].sum()


@needs_data
def test_event_definitions_nest_and_the_primary_one_matches_the_units():
    units = pd.read_csv(UNITS)
    units = units[units["group"] != "control"]
    primary = events_for(units, MAIN, 0.05, 2)
    assert (primary["event"] == units["event"]).all()
    looser = events_for(units, MAIN, 0.025, 1)
    stricter = events_for(units, MAIN, 0.10, 2)
    assert (looser["event"] >= primary["event"]).all() and (primary["event"] >= stricter["event"]).all()
    assert (primary["rise"] >= 0).all()


@needs_data
def test_likelihood_variant_combines_in_log_space():
    units = pd.read_csv(UNITS)
    u = with_likelihood(units[units["group"] != "control"], MAIN, "ell")
    assert np.isfinite(u["p_hat"]).all()
    assert (np.isneginf(u["risk"]) == (u["fpr"] == 0)).all()
