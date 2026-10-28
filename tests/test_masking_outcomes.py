"""Experiment B outcomes: family events, masked scores and the numbers reported in the paper."""

import json

import pandas as pd
import pytest

from src.analysis.masking_outcomes import family_events, scored_families
from src.paths import ROOT

OUTCOMES = ROOT / "results" / "analysis" / "masking_outcomes.csv"
needs_outcomes = pytest.mark.skipif(not OUTCOMES.exists(), reason="run src.analysis.masking_outcomes first")


def test_family_events_skip_no_box_and_break_ties_by_the_larger_rise(tmp_path):
    events = {"no_box": {"event": True, "time": 40}, "empty_symbols": {"event": True, "time": 70},
              "reasoning_opener": {"event": True, "time": 70}, "gibberish": {"event": False, "time": 300}}
    (tmp_path / "events_tau0.05.json").write_text(json.dumps({"events": events}))
    rows = [{"step": 0, "e/empty_symbols": 0.0, "e/reasoning_opener": 0.0},
            {"step": 70, "e/empty_symbols": 0.1, "e/reasoning_opener": 0.3}]
    (tmp_path / "eval_summary.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert family_events(tmp_path) == [(70, "reasoning_opener"), (70, "empty_symbols")]


def test_scored_families_zero_the_masked_fpr():
    scores = pd.DataFrame({"judge": ["j"] * 2, "pool": ["hard"] * 2, "family": ["a", "b"], "fpr": [0.6, 0.3],
                           "p_hat": [0.04, 0.04], "risk": [0.024, 0.012]})
    s = scored_families(scores, {"judge": "j", "pool": "hard", "mask": "a"}, q0=0.2)
    assert s.loc["a", ["fpr", "exposure"]].tolist() == [0.0, 0.0]
    assert s.loc["b", "exposure"] == pytest.approx(256 * 0.012)
    assert s.loc["a", "difficulty"] < 0 < s.loc["b", "difficulty"]


@needs_outcomes
def test_masking_numbers_of_the_paper():
    o = pd.read_csv(OUTCOMES).set_index("masked", append=True)
    assert o["event"].all() and int(o["hit"].sum()) == 3 and o.loc[o["hit"], "in_band"].all()
    assert (o["winner"] == o["fpr_alone_predicted"]).sum() == 0
    miss = o[~o["hit"]].iloc[0]
    assert (miss["winner"], int(miss["step"]), int(miss["rank_exposure"])) == ("adversarial_prefixes", 70, 3)
    assert sorted(o.loc[o["hit"], "step"].astype(int)) == [50, 60, 80]
    s = pd.read_csv(ROOT / "results" / "analysis" / "masking_summary.csv").set_index("analysis")
    pooled = s.loc["all runs with an event, main and masked"]
    assert (pooled["n"], pooled["top1_exposure"], pooled["top1_fpr"], pooled["top1_p_hat"]) == (16, 15, 6, 4)
