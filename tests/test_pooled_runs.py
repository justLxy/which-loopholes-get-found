"""Run-level results pooled over the main, masked and low-share runs (src/analysis/pooled_runs.py)."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import pooled_runs


@pytest.fixture
def analysis(tmp_path):
    pd.DataFrame([
        {"run": "j1-hard-s0-main", "winner": "a", "step": 50, "rank_exposure": 1, "rank_fpr": 2, "rank_p_hat": 1},
        {"run": "j1-hard-s1-main", "winner": "b", "step": 60, "rank_exposure": 1, "rank_fpr": 1, "rank_p_hat": 3},
    ]).to_csv(tmp_path / "run_level_winners.csv", index=False)
    pd.DataFrame([
        {"run": "j1-hard-s1-mask-b", "event": True, "winner": "c", "step": 70, "rank_exposure": 3, "rank_fpr": 3,
         "rank_p_hat": 8, "later_events": "d (80)"},
        {"run": "j1-hard-s2-mask-a", "event": False, "winner": np.nan, "step": np.nan, "rank_exposure": np.nan,
         "rank_fpr": np.nan, "rank_p_hat": np.nan, "later_events": np.nan},
    ]).to_csv(tmp_path / "masking_outcomes.csv", index=False)
    pd.DataFrame([
        {"run": "j2-hard-s0-low-F1", "event": True, "winner": "a", "step": 40, "hit_exposure": True, "hit_fpr": False,
         "hit_propensity": True, "later_events": "b@50"},
        {"run": "j2-hard-s0-low-P2", "event": True, "winner": "c", "step": 60, "hit_exposure": True, "hit_fpr": True,
         "hit_propensity": False, "later_events": np.nan},
    ]).to_csv(tmp_path / "lowshare_outcomes.csv", index=False)
    pd.DataFrame([
        {"run": "j1-hard-s0-main", "family": "a", "time": 50, "first": True},
        {"run": "j1-hard-s0-main", "family": "e", "time": 90, "first": False},
        {"run": "j1-hard-s1-main", "family": "b", "time": 60, "first": True},
    ]).to_csv(tmp_path / "event_timing.csv", index=False)
    return tmp_path


def test_winners_pool_the_three_experiments_and_skip_runs_without_an_event(analysis):
    t = pooled_runs.winners(analysis)
    assert list(t["experiment"]) == ["main", "main", "masked", "low share", "low share"]
    assert list(t["hit_exposure"]) == [True, True, False, True, True]
    assert list(t["hit_fpr"]) == [False, True, False, False, True]
    assert list(t["hit_p_hat"]) == [True, False, False, True, False]


def test_summary_reports_each_experiment_and_combines_only_main_and_low_share(analysis):
    s = pooled_runs.summary(pooled_runs.winners(analysis)).set_index("scope")
    assert list(s.index) == ["main", "masked", "low share", "main and low share"]
    c = s.loc["main and low share"]
    assert (c["runs"], c["top1_exposure"], c["top1_fpr"], c["top1_p_hat"]) == (4, 4, 2, 2)   # the masked run is left out
    assert (c["only_exposure_vs_fpr"], c["only_fpr"], c["neither_vs_fpr"]) == (2, 0, 0)
    assert c["mcnemar_p_vs_fpr"] == pytest.approx(0.5)
    assert (c["only_exposure_vs_p_hat"], c["only_p_hat"]) == (2, 0)
    assert s.loc["low share", "runs"] == 2 and s.loc["masked", "top1_exposure"] == 0


def test_checkpoints_group_main_runs_by_seed_and_keep_each_low_share_run_apart():
    table = pd.DataFrame([
        {"experiment": "main", "run": "j1-hard-s0-main", "hit_exposure": True, "hit_fpr": True, "hit_p_hat": False},
        {"experiment": "main", "run": "j2-hard-s0-main", "hit_exposure": True, "hit_fpr": False, "hit_p_hat": False},
        {"experiment": "main", "run": "j1-hard-s1-main", "hit_exposure": True, "hit_fpr": True, "hit_p_hat": True},
        {"experiment": "masked", "run": "j1-hard-s1-mask-b", "hit_exposure": False, "hit_fpr": False, "hit_p_hat": False},
        {"experiment": "low share", "run": "j2-hard-s0-low-F1", "hit_exposure": True, "hit_fpr": False, "hit_p_hat": True},
        {"experiment": "low share", "run": "j2-hard-s0-low-P2", "hit_exposure": True, "hit_fpr": True, "hit_p_hat": False},
    ])
    cp = pooled_runs.checkpoints(table).set_index("checkpoint")
    assert list(cp.index) == ["main-s0", "main-s1", "low-F1-s0", "low-P2-s0"]          # the masked run is left out
    assert cp.loc["main-s0", "runs"] == 2
    assert not cp.loc["main-s0", "all_fpr"] and cp.loc["main-s1", "all_fpr"]           # one miss fails the checkpoint
    s = pooled_runs.checkpoint_summary(cp.reset_index()).iloc[0]
    assert (s["checkpoints"], s["all_exposure"], s["all_fpr"], s["all_p_hat"]) == (4, 4, 2, 2)
    assert (s["only_exposure_vs_fpr"], s["only_fpr"]) == (2, 0)
    assert s["mcnemar_p_vs_fpr"] == pytest.approx(0.5)


def test_later_events_carry_the_fpr_of_the_winner_and_of_the_later_family(analysis, tmp_path):
    fpr = {("j1", "a"): 0.4, ("j1", "e"): 0.6, ("j1", "c"): 0.5, ("j1", "d"): 0.3, ("j2", "a"): 0.6, ("j2", "b"): 0.9}
    counts = {"results/main/j1-hard-s0-main": [(40, "a", 10, 9), (50, "a", 20, 10), (60, "e", 4, 4), (90, "e", 9, 0)],
              "results/masking/j1-hard-s1-mask-b": [(70, "c", 5, 5)],
              "results/lowshare/j2-hard-s0-low-F1": [(40, "a", 10, 7), (45, "b", 10, 9)]}
    for d, rows in counts.items():
        (tmp_path / d).mkdir(parents=True)
        pd.DataFrame(rows, columns=["step", "family", "n", "accepted"]).to_csv(tmp_path / d / "family_counts.csv",
                                                                              index=False)
    t = pooled_runs.later_events(analysis, fpr, root=tmp_path)
    assert list(t["experiment"]) == ["main", "masked", "low share"]
    assert list(t["later"]) == ["e", "d", "b"] and list(t["steps_after_winner"]) == [40, 10, 10]
    assert list(t["later_fpr_above_winner"]) == [True, False, True]
    assert t["winner_accepted_in_training"].tolist()[0] == pytest.approx(10 / 20)     # steps 50 to 89
    assert t["later_accepted_in_training"].tolist()[0] == pytest.approx(1.0)
    assert np.isnan(t["later_accepted_in_training"].tolist()[1])                       # no d responses
    assert t["winner_accepted_in_training"].tolist()[2] == pytest.approx(0.7)


def test_judge_is_read_from_the_run_name():
    assert pooled_runs.judge_of("qwen2p5_7b_instruct-hard-s1-low-F2") == "qwen2p5_7b_instruct"
    assert pooled_runs.judge_of("r1_distill_verifier_1p5b-hard-s2-mask-empty_symbols") == "r1_distill_verifier_1p5b"
