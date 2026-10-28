"""Run-level checks: winner ranks, McNemar on top-1 hits, the out-of-sample level, event time and difficulty."""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.run_level import (event_timing, level_check, mcnemar_top1, permutation_p, pool_acceptance,
                                    second_events, seeded_units, step0_acceptance, timing_rows, winner_ranks,
                                    with_difficulty)
from src.analysis.secondary import first_events
from src.paths import ROOT

UNITS = ROOT / "results" / "analysis" / "units.csv"
MAIN = ROOT / "results" / "main"
needs_data = pytest.mark.skipif(not (UNITS.exists() and MAIN.exists()), reason="needs results/main and units.csv")


def unit(run, family, risk, fpr, p_hat, event=False, time=300, group="core", judge="j", pool="hard"):
    return {"run": run, "group": group, "judge": judge, "pool": pool, "family": family, "risk": risk, "fpr": fpr,
            "p_hat": p_hat, "event": event, "time": time}


def test_winner_ranks_and_mcnemar_count_only_discordant_runs():
    rows = [unit("r1", "a", 0.03, 0.6, 0.05, True, 40), unit("r1", "b", 0.02, 0.9, 0.02),
            unit("r2", "a", 0.03, 0.9, 0.04, True, 60), unit("r2", "b", 0.01, 0.5, 0.02)]
    u = seeded_units(pd.DataFrame(rows))
    ranks = winner_ranks(u, {"r1": (40, "a"), "r2": (60, "a")})
    assert ranks["rank_exposure"].tolist() == [1, 1] and ranks["rank_fpr"].tolist() == [2, 1]
    assert mcnemar_top1(ranks, "exposure", "fpr") == (1, 0, 1.0)
    assert u["exposure"].tolist() == pytest.approx([7.68, 5.12, 7.68, 2.56])


def test_level_comes_from_core_events_and_is_applied_to_held_out_runs():
    rows = [unit("c", "a", 2.0 / 256, 0.5, 0.1, True, 50), unit("c", "b", 1.0 / 256, 0.5, 0.1),
            unit("h1", "a", 3.0 / 256, 0.5, 0.1, True, 30, "held_out"), unit("h1", "b", 1.0 / 256, 0.5, 0.1, group="held_out"),
            unit("h2", "a", 2.5 / 256, 0.5, 0.1, group="held_out"), unit("h3", "a", 1.5 / 256, 0.5, 0.1, group="held_out")]
    out = level_check(seeded_units(pd.DataFrame(rows)))
    assert out["level"] == pytest.approx(2.0)
    assert (out["held_out_runs"], out["held_out_runs_correct"], out["held_out_runs_predicted"]) == (3, 2, 2)
    assert (out["held_out_families_below"], out["held_out_families_below_with_event"]) == (2, 0)


def test_second_event_is_ranked_among_the_families_left():
    rows = [unit("r", "a", 0.05, 0.9, 0.05, True, 40), unit("r", "b", 0.04, 0.8, 0.05),
            unit("r", "c", 0.03, 0.9, 0.03, True, 70)]
    out = second_events(seeded_units(pd.DataFrame(rows)), {"r": (40, "a")})
    assert out.loc[0, ["second", "second_step", "families_left", "rank_exposure", "rank_fpr"]].tolist() == ["c", 70, 2, 2, 1]


def test_event_time_inverse_to_exposure_has_log_slope_minus_one():
    exposure = np.array([2.0, 3.0, 5.0, 8.0, 10.0])
    rows = [unit(f"r{i}", "a", e / 256, 0.5, e / 128, True, int(round(600 / e))) for i, e in enumerate(exposure)]
    u = seeded_units(pd.DataFrame(rows))
    ev = event_timing(u, {f"r{i}": (0, "a") for i in range(5)}, {f"r{i}": 0.2 for i in range(5)})
    row = timing_rows(ev)[0]
    assert row["spearman_exposure"] == pytest.approx(-1.0)
    assert row["loglog_slope_exposure"] == pytest.approx(-1.0, abs=0.02)
    assert (ev["kinetic"] > 0).all()


def test_permutation_p_is_small_for_a_perfect_order_and_undefined_for_a_constant():
    assert permutation_p(range(10), range(10), n=20000) < 0.001
    assert np.isnan(permutation_p([1, 2, 3, 4], [1, 1, 1, 1], n=1000))


def test_difficulty_score_is_negative_below_the_mean_acceptance():
    u = seeded_units(pd.DataFrame([unit("r", "a", 0.03, 0.6, 0.05), unit("r", "b", 0.03, 0.3, 0.1)]))
    d = with_difficulty(u, {"r": 0.4})
    assert d["difficulty"].tolist() == pytest.approx([0.05 * 0.2 / np.sqrt(0.24), 0.1 * -0.1 / np.sqrt(0.24)])


def test_step0_acceptance_and_pool_acceptance_read_the_run_logs(tmp_path):
    for run, pool in (("rh", "hard"), ("re", "easy")):
        (tmp_path / run).mkdir()
        (tmp_path / run / "eval_summary.jsonl").write_text(
            json.dumps({"step": 0, "accept_rate": 0.2}) + "\n" + json.dumps({"step": 10, "accept_rate": 0.9}) + "\n")
        pd.DataFrame({"step": [0, 5, 20], "family": ["a"] * 3, "n": [40, 20, 99],
                      "accepted": [10, 5, 99] if pool == "hard" else [20, 10, 0]}).to_csv(
            tmp_path / run / "family_counts.csv", index=False)
    assert step0_acceptance(["rh"], tmp_path) == {"rh": 0.2}
    units = pd.DataFrame([unit("rh", "a", 0.1, 0.4, 0.2, pool="hard"), unit("re", "a", 0.1, 0.4, 0.2, pool="easy")])
    acc = pool_acceptance(units, tmp_path, min_n=50)
    assert acc.loc[0, ["n_hard", "n_easy", "accept_hard", "accept_easy", "audit_fpr"]].tolist() == [60, 60, 0.25, 0.5, 0.4]


@needs_data
def test_run_level_numbers_of_the_paper():
    u = seeded_units(pd.read_csv(UNITS))
    firsts = first_events(u, MAIN)
    ranks = winner_ranks(u, firsts)
    assert [int((ranks[f"rank_{s}"] == 1).sum()) for s in ("exposure", "fpr", "p_hat")] == [12, 6, 2]
    assert mcnemar_top1(ranks, "exposure", "fpr") == (6, 0, pytest.approx(0.03125))
    assert mcnemar_top1(ranks, "exposure", "p_hat") == (10, 0, pytest.approx(0.001953125))
    assert int(((ranks["rank_fpr"] != 1) & (ranks["rank_p_hat"] != 1)).sum()) == 5
    level = level_check(u)
    assert level["level"] == pytest.approx(1.92, abs=0.005)
    assert (level["held_out_runs_correct"], level["held_out_families_below_with_event"]) == (8, 0)
    q0 = step0_acceptance(u["run"].unique(), MAIN)
    ev = event_timing(u, firsts, q0)
    assert len(ev) == 14 and timing_rows(ev)[0]["spearman_exposure"] == pytest.approx(-0.844, abs=0.001)
    d = with_difficulty(u, q0)
    core = d[d["group"] == "core"]
    from src.analysis.primary import auroc
    assert auroc(core["difficulty"], core["event"]) == pytest.approx(0.993, abs=0.001)


def test_difficulty_auroc_covers_the_three_pooled_analyses():
    from src.analysis.run_level import difficulty_auroc
    rows = []
    for run, group, judge in (("c1", "core", "a"), ("c2", "core", "pc"), ("h1", "held_out", "b")):
        rows += [unit(run, "w", 0.05, 0.6, 0.1, True, 40, group, judge), unit(run, "l", 0.01, 0.1, 0.1, False, 300, group, judge)]
    u = seeded_units(pd.DataFrame(rows)).assign(difficulty=lambda d: d["fpr"] * d["p_hat"])
    out = difficulty_auroc(u, n_boot=200, seed=0, positive_control="pc").set_index("analysis")
    assert list(out.index) == ["core pooled", "core pooled without the positive control", "held-out pooled"]
    assert out["auroc"].tolist() == [1.0, 1.0, 1.0] and out.loc["core pooled", "runs"] == 2
    assert out.loc["core pooled without the positive control", "runs"] == 1
