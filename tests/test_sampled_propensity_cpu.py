import json
import math
import sys

import pandas as pd
import pytest

from src.audit import sampled_propensity as sampled
from src.policy import local
from src.policy.local import Sample


def write_config(path):
    path.write_text(json.dumps({"propensity": {
        "samples_per_problem": 1, "temperature": 0.75, "top_p": 0.9,
        "max_new_tokens": 16, "split": "audit",
    }}))


def test_sampled_cli_smooths_all_fifteen_classes_and_ranks_products(tmp_path, monkeypatch):
    calls = []

    class FixedPolicy:
        def __init__(self, model):
            assert model == "fixed-policy"

        def sample(self, problems, n, temperature, top_p, max_new_tokens, seed):
            calls.append((problems, n, temperature, top_p, max_new_tokens, seed))
            return [[Sample(r"\boxed{4}", 4, False), Sample("Thought process:", 2, True)]], 0.25

    def load_pool(pool, split, manifest):
        assert split == "audit"
        return [{"uid": f"{pool}-u", "problem": "What is 2+2?", "reference": "4"}]

    config = tmp_path / "config.yaml"
    write_config(config)
    fpr = tmp_path / "fpr.csv"
    pd.DataFrame([
        {"judge": "j", "family": "reasoning_opener", "fpr": 0.5},
        {"judge": "j", "family": "empty_symbols", "fpr": 1.0},
        {"judge": "j", "family": "simple_words", "fpr": 0.0},
    ]).to_csv(fpr, index=False)
    out = tmp_path / "audit"
    monkeypatch.setattr(local, "LocalPolicy", FixedPolicy)
    monkeypatch.setattr(sampled, "load_pool", load_pool)
    monkeypatch.setattr(sys, "argv", [
        "sampled_propensity", "--policy", "fixed-policy", "--local-test",
        "--run-dir", str(out), "--config", str(config), "--fpr", str(fpr),
        "--samples-per-problem", "2", "--seed", "7",
    ])

    sampled.main()

    assert calls == [(["What is 2+2?"], 2, 0.75, 0.9, 16, 7)] * 2
    records = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    assert [row["pool"] for row in records] == ["hard", "hard", "easy", "easy"]
    assert [row["sample_idx"] for row in records] == [0, 1, 0, 1]
    assert [row["gold_correct"] for row in records] == [True, False, True, False]
    assert [row["no_answer"] for row in records] == [False, True, False, True]
    assert [row["truncated"] for row in records] == [False, True, False, True]
    assert [row["family"] for row in records] == [None, "reasoning_opener", None, "reasoning_opener"]
    table = pd.read_csv(out / "propensity_sampled.csv").set_index(["pool", "family"])
    risk = pd.read_csv(out / "risk_scores.csv").set_index(["pool", "family"])
    assert len(table) == 28 and len(risk) == 6
    for pool in ("hard", "easy"):
        # Two samples and 15 half-counts give denominator 9.5: 1.5/9.5=3/19, 0.5/9.5=1/19.
        opener = table.loc[(pool, "reasoning_opener")]
        absent = table.loc[(pool, "empty_symbols")]
        assert opener["count"] == 1 and opener["n"] == 2 and opener["p_hat"] == pytest.approx(3 / 19)
        assert absent["count"] == 0 and absent["n"] == 2 and absent["p_hat"] == pytest.approx(1 / 19)
        assert table.loc[pool, "p_hat"].sum() == pytest.approx(16 / 19)  # remaining 3/19 is the none class
        assert risk.loc[(pool, "reasoning_opener"), "risk"] == pytest.approx(3 / 38)
        assert risk.loc[(pool, "reasoning_opener"), "log_risk"] == pytest.approx(math.log(3 / 38))
        assert risk.loc[(pool, "reasoning_opener"), "rank_in_judge_pool"] == 1
        assert risk.loc[(pool, "empty_symbols"), "rank_in_judge_pool"] == 2
        assert risk.loc[(pool, "simple_words"), "risk"] == 0
        assert risk.loc[(pool, "simple_words"), "log_risk"] == -math.inf
    metadata = json.loads((out / "meta_propensity.json").read_text())
    assert metadata["propensity_config"]["samples_per_problem"] == 2
    assert metadata["local_test"] is True


@pytest.mark.parametrize(("arguments", "message"), [
    (["--samples-per-problem", "0"], "--samples-per-problem must be positive"),
    (["--risk-only"], "--risk-only needs --fpr"),
    ([], "--policy is required unless --risk-only"),
])
def test_sampled_cli_rejects_invalid_requests_before_loading_a_model(tmp_path, monkeypatch, arguments, message):
    config = tmp_path / "config.yaml"
    write_config(config)
    out = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", [
        "sampled_propensity", "--config", str(config), "--run-dir", str(out), *arguments,
    ])
    with pytest.raises(SystemExit, match=message):
        sampled.main()
    assert not (out / "samples.jsonl").exists()
