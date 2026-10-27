"""Descriptive checks: rewards before an event, and evaluation acceptance by problem source."""

import json

import pandas as pd
import pytest

from src.analysis.descriptive import acceptance_by_source, eval_acceptance, rewards_before_event
from src.paths import ROOT

UNITS = ROOT / "results" / "analysis" / "units.csv"
needs_units = pytest.mark.skipif(not UNITS.exists(), reason="run src.analysis.primary first")


def test_rewards_count_only_the_event_family_before_the_event_step(tmp_path):
    (tmp_path / "r").mkdir()
    pd.DataFrame({"step": [0, 5, 9, 10, 5], "family": ["a", "a", "a", "a", "b"],
                  "n": [3, 4, 5, 6, 7], "accepted": [1, 2, 3, 4, 5]}).to_csv(tmp_path / "r" / "family_counts.csv",
                                                                         index=False)
    units = pd.DataFrame({"run": ["r", "r"], "group": ["core"] * 2, "judge": ["j"] * 2, "family": ["a", "b"],
                          "event": [True, False], "time": [10, 300], "risk": [0.01, 0.5]})
    out = rewards_before_event(units, tmp_path)
    assert len(out) == 1
    assert out.loc[0, ["rewarded_before", "sampled_before"]].tolist() == [6, 12]
    assert out.loc[0, "exposure"] == pytest.approx(2.56)


def test_acceptance_is_split_by_source_and_averaged_over_seeds(tmp_path):
    def write(run, rows):
        (tmp_path / run).mkdir()
        with (tmp_path / run / "eval.jsonl").open("w") as f:
            for step, source, accepted in rows:        # the eval log stores every field as a string
                f.write(json.dumps({"step": str(step), "source": source, "accepted": str(accepted)}) + "\n")
    write("j-hard-s0", [(0, "deepscaler", True), (0, "deepscaler", False), (0, "math_train", False), (10, "math_train", True)])
    write("j-hard-s1", [(0, "deepscaler", True), (0, "deepscaler", True), (0, "math_train", True)])
    assert eval_acceptance(tmp_path / "j-hard-s0" / "eval.jsonl") == {"deepscaler": [1, 2], "math_train": [0, 1]}
    units = pd.DataFrame({"run": ["j-hard-s0", "j-hard-s1"], "group": ["core"] * 2, "judge": ["j"] * 2,
                          "pool": ["hard"] * 2, "seed": [0, 1]})
    table = acceptance_by_source(units, tmp_path).set_index("source")
    assert table.loc["deepscaler", "accept"] == pytest.approx((0.5 + 1.0) / 2)
    assert table.loc["all", "accept"] == pytest.approx((1 / 3 + 1.0) / 2)
    assert table.loc["deepscaler", "outputs"] == 4
    assert table.loc["deepscaler", "advantage_of_a_rewarded_response"] == pytest.approx((7 / 8 * (1 - 0.75) / 0.75) ** 0.5)


@needs_units
def test_every_event_family_had_collected_hundreds_of_rewards():
    units = pd.read_csv(UNITS)
    out = rewards_before_event(units, ROOT / "results" / "main")
    assert len(out) == 14
    assert out["rewarded_before"].min() == 234 and out["rewarded_before"].max() == 804
    assert out["rewarded_before"].median() == 586
