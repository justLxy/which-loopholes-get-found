"""Experiment B: the rule-masked judge, the masked risk snapshot, the predictions and the run commands."""

import csv
import math

import pandas as pd
import pytest

from src.judges.local import JudgeResult
from src.judges.masked import MaskedJudge
from src.paths import ROOT
from src.train import grpo, masking


class FakeJudge:
    """Accepts everything and records what it was asked."""
    cfg = {"model": "fake"}

    def __init__(self):
        self.seen = []

    def judge(self, items):
        self.seen += [it["response"] for it in items]
        return [JudgeResult(it["response"], "ok", True, False, "1") for it in items], 0.0


def test_masked_judge_rejects_only_the_masked_family_and_keeps_order():
    inner = FakeJudge()
    judge = MaskedJudge(inner, ["reasoning_opener"])
    items = [{"response": r} for r in ("{", "Thought process:", "The answer is \\boxed{9}.", "Thought process:")]
    results, _ = judge.judge(items)
    assert [r.accepted for r in results] == [True, False, True, False]
    assert [r.label for r in results] == ["1", "masked", "1", "masked"]
    assert inner.seen == ["{", "The answer is \\boxed{9}."]
    assert judge.cfg["masked_families"] == ["reasoning_opener"]


def test_masked_judge_validates_families_and_skips_an_all_masked_batch():
    with pytest.raises(ValueError):
        MaskedJudge(FakeJudge(), ["no_such_family"])
    with pytest.raises(ValueError):
        MaskedJudge(FakeJudge(), [])
    inner = FakeJudge()
    results, _ = MaskedJudge(inner, ["empty_symbols"]).judge([{"response": "{"}])
    assert [r.accepted for r in results] == [False] and inner.seen == []


def test_risk_snapshot_zeroes_a_masked_family_and_reranks(tmp_path):
    src = tmp_path / "risk.csv"
    rows = [{"judge": "j", "family": f, "fpr": fpr, "pool": "hard", "count": 1, "n": 10, "p_hat": 0.1,
             "risk": risk, "log_risk": -1.0, "rank_in_judge_pool": rank}
            for f, fpr, risk, rank in (("a", 0.6, 0.06, 1.0), ("b", 0.4, 0.04, 2.0), ("c", 0.2, 0.02, 3.0))]
    with src.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "predicted.csv"
    assert grpo.risk_snapshot("j", "hard", out, src, mask_families=["a"]) == 3
    got = {r["family"]: r for r in csv.DictReader(out.open())}
    assert (got["a"]["fpr"], got["a"]["risk"], got["a"]["masked"]) == ("0.0", "0.0", "1")
    assert [got[f]["rank_in_judge_pool"] for f in "abc"] == ["3.0", "1.0", "2.0"]
    grpo.risk_snapshot("j", "hard", out, src)
    assert "masked" not in next(csv.DictReader(out.open()))


def test_grpo_accepts_repeated_mask_families():
    args = grpo.parse_args(["--judge", "r1_distill_verifier_1p5b", "--pool", "hard", "--mask-family", "empty_symbols",
                            "--mask-family", "reasoning_opener"])
    assert args.mask_family == ["empty_symbols", "reasoning_opener"]
    assert grpo.parse_args(["--judge", "x", "--pool", "hard"]).mask_family == []


def test_prediction_masks_the_family_and_reads_the_timing_fit():
    scores = pd.DataFrame({"judge": ["j"] * 3, "pool": ["hard"] * 3, "family": ["a", "b", "c"],
                           "fpr": [0.6, 0.9, 0.3], "p_hat": [0.04, 0.005, 0.045], "risk": [0.024, 0.0045, 0.0135]})
    run = {"judge": "j", "pool": "hard", "seed": 1, "mask": "a", "tag": "mask-a", "checkpoint": "main-rot1"}
    row = masking.predict(run, scores, level=1.92, fit=(5.0, -1.0, 0.0))
    assert (row["winner"], row["runner_up"], row["fpr_alone_winner"], row["propensity_alone_winner"]) == ("c", "b", "b", "c")
    assert row["winner_exposure"] == pytest.approx(256 * 0.0135)
    assert row["event_predicted"] and row["step"] == pytest.approx(math.exp(5) / (256 * 0.0135))
    low = masking.predict({**run, "mask": "c"}, scores.assign(risk=[0.001, 0.001, 0.0135]), level=1.92, fit=(5.0, -1.0, 0.0))
    assert not low["event_predicted"]


def test_plan_runs_extend_the_main_run_command():
    cfg, runs = masking.load_plan()
    assert len(runs) == 4 and cfg["level"] == pytest.approx(1.92)
    run = runs[0]
    cmd = masking.run_command(run)
    assert cmd[:5] == ["bash", "scripts/run_grpo.sh", run["judge"], run["pool"], str(run["seed"])]
    assert cmd[-2:] == ["--mask-family", run["mask"]]
    assert f"runs/sft/main-rot{run['seed']}/model" in cmd and f"mask-{run['mask']}" in cmd
    for r in runs:
        assert masking.original_scores(cfg, r).exists()
        assert masking.committed_scores(cfg, r) == ROOT / "results" / "masking" / f"sft-{r['checkpoint']}" / "risk_scores.csv"
    assert [c["name"] for c in masking.checkpoints_of(runs)] == ["main-rot1", "main-rot2"]


@pytest.mark.skipif(not (ROOT / "results" / "analysis" / "event_timing.csv").exists(), reason="run src.analysis.run_level")
def test_predictions_of_the_plan_separate_exposure_from_fpr_alone():
    cfg, runs = masking.load_plan()
    fit = masking.timing_fit(ROOT / cfg["timing_fit"])
    rows = [masking.predict(r, pd.read_csv(masking.original_scores(cfg, r)), cfg["level"], fit) for r in runs]
    assert all(row["event_predicted"] for row in rows)
    assert all(row["winner"] != row["fpr_alone_winner"] for row in rows)
    assert [row["winner"] for row in rows] == ["system_prompt_mimicry", "system_prompt_mimicry", "reasoning_opener",
                                               "answer_explanation"]


def test_seeding_comparison_and_prediction_match():
    import json
    summary = json.loads((ROOT / "results" / "main" / "sft-main-rot1" / "data_summary.json").read_text())
    assert masking.compare_seeding(summary, summary) == []
    changed = {**summary, "per_family": {**summary["per_family"], "gibberish": 7}}
    assert masking.compare_seeding(changed, summary) == ["per_family"]
    a = pd.DataFrame({"run": ["r"], "winner": ["x"], "fpr_alone_winner": ["y"], "event_predicted": [True],
                      "winner_exposure": [3.0]})
    assert masking.same_predictions(a, a.copy())
    assert not masking.same_predictions(a, a.assign(winner="z"))
