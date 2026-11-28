"""Pin the README headline results to the committed analysis tables."""

import pandas as pd
import pytest

from src.paths import ROOT

ANALYSIS = ROOT / "results" / "analysis"


@pytest.mark.parametrize(
    "score,expected,headline",
    [
        ("risk", 0.9741760108732586, "0.974"),
        ("fpr", 0.9373088685015291, "0.937"),
        ("p_hat", 0.8472646958885491, "0.847"),
        ("static", 0.37869520897043835, "0.379"),
    ],
)
def test_core_auroc(score, expected, headline):
    table = pd.read_csv(ANALYSIS / "primary_auroc.csv")
    core = table[table["analysis"] == "core pooled"].set_index("score")
    assert core.index.is_unique
    row = core.loc[score]
    assert (row["runs"], row["units"], row["events"]) == (24, 336, 9)
    assert row["auroc"] == pytest.approx(expected, rel=0, abs=1e-12)
    assert f"{row['auroc']:.3f}" == headline


def test_main_winners():
    table = pd.read_csv(ANALYSIS / "run_level_winners.csv")
    assert len(table) == table["run"].nunique() == 12
    assert tuple(int((table[f"rank_{score}"] == 1).sum()) for score in ("exposure", "fpr", "p_hat")) == (12, 6, 2)
    summary = pd.read_csv(ANALYSIS / "pooled_summary.csv").set_index("scope").loc["main"]
    assert tuple(summary[k] for k in ("runs", "top1_exposure", "top1_fpr", "top1_p_hat")) == (12, 12, 6, 2)


def test_masking_winners():
    table = pd.read_csv(ANALYSIS / "masking_outcomes.csv")
    assert len(table) == table["run"].nunique() == 4
    assert table["event"].all()
    assert int(table["hit"].sum()) == int((table["rank_exposure"] == 1).sum()) == 3
    summary = pd.read_csv(ANALYSIS / "masking_summary.csv").set_index("analysis").loc["masked runs"]
    assert (summary["n"], summary["winner_predicted_by_exposure"]) == (4, 3)


def test_lowshare_winners():
    table = pd.read_csv(ANALYSIS / "lowshare_outcomes.csv")
    assert len(table) == table["run"].nunique() == 6
    assert table["event"].all() and table["hit_exposure"].all()
    summary = pd.read_csv(ANALYSIS / "lowshare_summary.csv").set_index("scope").loc["all runs"]
    assert (summary["runs"], summary["exposure_correct"]) == (6, 6)


@pytest.mark.parametrize("baseline,wins,p_value,headline", [("fpr", 10, 0.001953125, "0.002"),
                                                         ("p_hat", 12, 0.00048828125, "0.0005")])
def test_pooled_comparisons(baseline, wins, p_value, headline):
    table = pd.read_csv(ANALYSIS / "pooled_summary.csv").set_index("scope")
    assert table.index.is_unique
    pooled = table.loc["main and low share"]
    assert (pooled["runs"], pooled["top1_exposure"]) == (18, 18)
    assert (pooled[f"only_exposure_vs_{baseline}"], pooled[f"only_{baseline}"]) == (wins, 0)
    assert pooled[f"mcnemar_p_vs_{baseline}"] == pytest.approx(p_value, rel=0, abs=1e-15)
    digits = 3 if baseline == "fpr" else 4
    assert f"{pooled[f'mcnemar_p_vs_{baseline}']:.{digits}f}" == headline
