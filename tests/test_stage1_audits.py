import math
import os
from pathlib import Path

import pandas as pd
import pytest

from src.audit.cap_check import compare, merge, summary_line
from src.audit.likelihood import (
    aggregate_likelihood,
    canonical_strings,
    logmeanexp,
    logsumexp,
    risk_scores,
    sampled_propensity,
)
from src.audit.static_accuracy import make_process, sample_items, summarize, wrong_kind
from src.detectors import FAMILIES
from src.judges.local import JudgeResult
from src.pipeline.records import validate_record

HACKING = Path(__file__).resolve().parents[1] / "data" / "raw" / "hacking" / "hacking_data.parquet"


# --- likelihood propensity ------------------------------------------------------------

@pytest.mark.skipif(not HACKING.exists(), reason="run src.data.download_hacking first")
def test_canonical_strings():
    s = canonical_strings()
    assert list(s) == list(FAMILIES) and all(1 <= len(v) <= 5 for v in s.values())
    assert s["empty_symbols"] == ["{", ".", "-", "}", ""]           # most frequent; ties broken by string
    assert all(len(g) == 1000 for g in s["gibberish"]) and s == canonical_strings()
    assert s["reasoning_opener"][0] == "Thought process:" and len(s["reasoning_opener"]) == 5
    assert "random_gibberish_1000chars" not in s["gibberish"]


def test_log_helpers():
    assert logsumexp([math.log(0.2), math.log(0.3)]) == pytest.approx(math.log(0.5))
    assert logmeanexp([math.log(0.2), math.log(0.4)]) == pytest.approx(math.log(0.3))
    assert logsumexp([-1000.0, -1001.0]) == pytest.approx(-1000 + math.log(1 + math.exp(-1)))   # no underflow


def test_aggregate_likelihood():
    recs = [{"pool": "hard", "family": "f", "uid": "a", "logprob": math.log(0.1), "n_tokens": 2},
            {"pool": "hard", "family": "f", "uid": "a", "logprob": math.log(0.3), "n_tokens": 4},
            {"pool": "hard", "family": "f", "uid": "b", "logprob": math.log(0.2), "n_tokens": 2}]
    out = aggregate_likelihood(recs)[("hard", "f")]
    # problem a: 0.1 + 0.3 = 0.4; problem b: 0.2; mean 0.3
    assert out["ell"] == pytest.approx(math.log(0.3)) and out["n_problems"] == 2
    assert out["per_token"] == pytest.approx((math.log(0.1) / 2 + math.log(0.3) / 4 + math.log(0.2) / 2) / 3)


def test_sampled_propensity_jeffreys():
    recs = [{"uid": "a", "family": "gibberish"}, {"uid": "a", "family": None}, {"uid": "b", "family": None},
            {"uid": "z", "family": "gibberish"}]                     # z is not a training problem
    out = sampled_propensity(recs, {"hard": {"a", "b"}})
    k = len(FAMILIES) + 1
    assert out[("hard", "gibberish")] == {"count": 1, "n": 3, "p_hat": 1.5 / (3 + 0.5 * k)}
    assert out[("hard", "prompt_injection")]["p_hat"] == pytest.approx(0.5 / (3 + 0.5 * k))


def test_risk_scores(tmp_path):
    fpr = pd.DataFrame([{"judge": "j", "family": "gibberish", "fpr": 0.2},
                        {"judge": "j", "family": "simple_words", "fpr": 0.0},
                        {"judge": "j", "family": "gibberish_placeholder", "fpr": 0.5}])
    fpr.to_csv(tmp_path / "fpr.csv", index=False)
    table = pd.DataFrame([{"pool": "hard", "family": "gibberish", "ell": -50.0, "p_hat": 0.01},
                          {"pool": "hard", "family": "simple_words", "ell": -5.0, "p_hat": 0.001}])
    risk = risk_scores(table, tmp_path / "fpr.csv")
    assert set(risk["family"]) == {"gibberish", "simple_words"}      # placeholder is not a family
    g = risk[risk["family"] == "gibberish"].iloc[0]
    assert g["log_risk"] == pytest.approx(-50 + math.log(0.2)) and g["rank_in_judge_pool"] == 1
    assert risk[risk["family"] == "simple_words"].iloc[0]["log_risk"] == float("-inf")


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 to load a real model")
def test_scorer_matches_full_forward():
    import torch

    from src.audit.likelihood import LogProbScorer
    sc = LogProbScorer("Qwen/Qwen2.5-0.5B-Instruct", device="cpu", batch_size=2)   # fp32: exact comparison
    prompt = sc.tokenizer("Question: 2+2?\nAnswer:", add_special_tokens=False)["input_ids"]
    conts = [sc.encode_continuation(" 4"), sc.encode_continuation(" Final Decision: Yes")]
    fast = sc.score(prompt, conts)
    for c, (lp, n) in zip(conts, fast):
        ids = torch.tensor([prompt + c], device=sc.device)
        with torch.inference_mode():
            logits = sc.model(ids).logits[0].float()
        logp = torch.log_softmax(logits, -1)
        ref = sum(float(logp[len(prompt) - 1 + i, t]) for i, t in enumerate(c))
        assert n == len(c) and lp == pytest.approx(ref, abs=1e-3)


# --- static accuracy ------------------------------------------------------------------

def srec(uid, idx, correct, truncated=False, no_answer=False):
    return {"schema_version": 2, "run_id": "d", "stage": "difficulty", "step": None, "uid": uid, "source": "math_train",
            "reference": "6", "sample_idx": idx, "request_seed": 1, "policy_model": "p",
            "response": "\\boxed{6}" if correct else "\\boxed{7}", "n_tokens": 5, "truncated": truncated,
            "gold_correct": correct, "gold_extracted": "6" if correct else "7", "no_answer": no_answer,
            "n_boxed": 1, "family": None, "contains": []}


def fake_records():
    out = []
    for i in range(40):
        uid = f"u{i:02d}"
        out += [srec(uid, 0, True), srec(uid, 1, False), srec(uid, 2, False, truncated=True, no_answer=True)]
    return out


MANIFEST = {"pools": {"hard": {"train": [f"u{i:02d}" for i in range(20)]},
                      "easy": {"train": [f"u{i:02d}" for i in range(20, 35)]}}}


def test_static_sample():
    items = sample_items(fake_records(), MANIFEST, per_cell=10, seed=0)
    assert len(items) == 40 and len({it["key"] for it in items}) == 40
    for pool, uids in (("hard", range(20)), ("easy", range(20, 35))):
        cell = [it for it in items if it["pool"] == pool]
        assert sum(it["gold_correct"] for it in cell) == 10 and all(int(it["uid"][1:]) in uids for it in cell)
    assert sample_items(fake_records(), MANIFEST, 10, 0) == items
    assert wrong_kind(srec("a", 0, False, truncated=True)) == "truncated"


class AcceptAll:
    def judge(self, items):
        return [JudgeResult("6", "\\boxed{1}", True, False, "1") for _ in items], 0.0


def test_static_process_and_summary():
    items = sample_items(fake_records(), MANIFEST, per_cell=10, seed=0)
    records = make_process(AcceptAll(), "j", "run", {it["uid"]: "Q" for it in items})(items)
    for r in records:
        validate_record(r)
        assert r["stage"] == "static_accuracy"
    rows = {row["pool"]: row for row in summarize(records, {it["key"]: it["pool"] for it in items})}
    assert rows["all"]["tpr"] == 1.0 and rows["all"]["tnr"] == 0.0 and rows["all"]["balanced_accuracy"] == 0.5
    assert rows["hard"]["n_correct"] == 10 and rows["easy"]["n_wrong"] == 10
    assert rows["all"]["n_answered"] + rows["all"]["n_truncated"] == 20


# --- judge-limit check ------------------------------------------------------------------

def arec(key, family, accepted, truncated, parse_fail=False, tokens=100):
    return {"key": key, "family": family, "accepted": accepted, "judge_truncated": truncated,
            "parse_fail": parse_fail, "judge_tokens": tokens}


def test_cap_compare():
    original = [arec("a", "gibberish", False, True, True), arec("b", "gibberish", False, False),
                arec("c", "simple_words", True, False), arec("d", "simple_words", False, True, True)]
    rejudged = [arec("a", "gibberish", True, False, tokens=3000), arec("d", "simple_words", False, True, True, 4096)]
    merged = merge(original, rejudged)
    assert [r["accepted"] for r in merged] == [True, False, True, False]
    rows = {r["family"]: r for r in compare(original, rejudged)}
    assert rows["gibberish"]["fpr_before"] == 0.0 and rows["gibberish"]["fpr_after"] == 0.5
    assert rows["simple_words"]["truncated_after"] == 1
    line = summary_line("r1_distill_qwen_1p5b", list(rows.values()), rejudged)
    assert line["rejudged"] == 2 and line["now_finished"] == 1 and line["now_accepted"] == 1
    assert line["huang_reported_%"] == 12.3
