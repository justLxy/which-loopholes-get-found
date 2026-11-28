"""Propensity seeding without a GPU: the SFT set, the seeded propensity and the risk table."""

import dataclasses
import json

import pandas as pd
import pytest

from src.audit.fpr import HACKING
from src.audit.sampled_propensity import risk_table
from src.data.pools import CANDIDATES, DEFAULT_MANIFEST
from src.detectors import FAMILIES
from src.policy.local import Sample
from src.train import grpo
from src.train.seed_data import (
    clean_examples,
    family_answer_pools,
    family_shares,
    load_or_build_clean_pool,
    load_sft_config,
    outside_pools,
    pairing_problems,
    seed_counts,
    seeded_examples,
    tier_groups,
    tier_shares,
    to_sft_record,
)

needs_data = pytest.mark.skipif(
    not (HACKING.exists() and CANDIDATES.exists()), reason="needs data/raw and data/processed"
)


def test_rotated_tiers_form_a_latin_square():
    tiers = [0.003, 0.012, 0.04]
    groups = tier_groups(0)
    assert sorted(len(g) for g in groups) == [4, 5, 5] and sorted(f for g in groups for f in g) == sorted(FAMILIES)
    assert tier_groups(0) == groups and tier_groups(1) != groups              # fixed by group_seed
    rotations = [tier_shares(tiers, r, 0) for r in range(3)]
    for fam in FAMILIES:
        assert sorted(rot[fam] for rot in rotations) == tiers
    counts = seed_counts(2000, rotations[0])
    for fam, share in rotations[0].items():
        assert counts[fam] == round(share * 2000)                              # a share of the fixed set size
    with pytest.raises(SystemExit):
        family_shares({"seeding": "rotated", "tiers": tiers, "group_seed": 0})          # needs a rotation
    assert family_shares({"family_share": 0.02}) == {f: 0.02 for f in FAMILIES}       # uniform by default


def test_seeded_answers_replace_clean_ones_in_a_fixed_size_set():
    counts = seed_counts(2000, {f: 0.02 for f in FAMILIES})
    assert set(counts.values()) == {40} and 2000 - sum(counts.values()) == 1440
    assert sum(seed_counts(2000, family_shares({"seeding": "none"})).values()) == 0     # the 0% control
    with pytest.raises(ValueError):
        seed_counts(100, {f: 0.1 for f in FAMILIES})                                    # 140% of the set


def test_clean_pool_is_built_once_and_reused(tmp_path):
    calls = []

    def factory():
        calls.append(1)
        return FakePolicy([Sample("so \\boxed{4}", 3, False)])

    problems = [{"uid": f"u{i}", "problem": "What is 2+2?", "answer": "4"} for i in range(3)]
    d = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 8, "seed": 0}
    pool, stats = load_or_build_clean_pool(tmp_path / "pool.jsonl", factory, problems, d)
    again, stats2 = load_or_build_clean_pool(tmp_path / "pool.jsonl", factory, problems, d)
    assert len(calls) == 1 and pool == again and stats == stats2 and stats["kept"] == 3


@needs_data
def test_seeded_examples_cover_all_families_and_are_detected_as_such():
    pools = family_answer_pools()
    assert set(pools) | {"gibberish"} == set(FAMILIES)
    problems = [{"uid": f"u{i}", "problem": f"What is {i}+1?", "answer": str(i + 1)} for i in range(20)]
    ex = seeded_examples(problems, 3, seed=0)
    assert len(ex) == 3 * len(FAMILIES)
    assert all(e["detected"] == e["kind"] for e in ex)
    assert ex == seeded_examples(problems, 3, seed=0)                 # seeded, reproducible
    by_family = seeded_examples(problems, {"gibberish": 2, "simple_words": 1}, seed=0)
    assert sorted(e["kind"] for e in by_family) == ["gibberish", "gibberish", "simple_words"]
    gib = [e["response"] for e in ex if e["kind"] == "gibberish"]
    assert len(set(gib)) == 3 and all(len(g) == 1000 for g in gib)


@needs_data
def test_outside_pools_excludes_every_pool_problem():
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    used = {u for p in manifest["pools"].values() for split in p.values() for u in split}
    rest = outside_pools()
    assert len(rest) == 21845 - len({u for p in manifest["pools"].values() for u in p["train"] + p["eval"]})
    assert not used & {c["uid"] for c in rest}


class FakePolicy:
    def __init__(self, texts):
        self.texts = texts

    def sample(self, problems, n, temperature, top_p, max_new_tokens, seed):
        return [[self.texts[i % len(self.texts)]] for i in range(len(problems))], 0.0


def test_clean_examples_keep_finished_correct_ordinary_answers():
    problems = [{"uid": f"u{i}", "problem": "What is 2+2?", "answer": "4"} for i in range(5)]
    policy = FakePolicy([Sample("2+2=4, so \\boxed{4}.", 9, False), Sample("\\boxed{5}", 3, False),
                         Sample("2+2=4, so \\boxed{4}", 9, True), Sample("Final Decision: Yes", 4, False),
                         Sample("Adding, \\boxed{4}.", 4, False)])
    cfg = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 64}
    out, stats = clean_examples(policy, problems, cfg, seed=0)
    assert [e["uid"] for e in out] == ["u0", "u4"]
    assert stats["sampled"] == 5 and stats["correct"] == 3 and stats["truncated"] == 1 and stats["family"] == 1


def test_seeded_answers_are_paired_with_the_clean_examples_problems():
    sampled = [{"uid": f"u{i}", "problem": f"p{i}", "answer": "1"} for i in range(5)]
    clean = [{"uid": "u3"}, {"uid": "u1"}]
    assert [p["uid"] for p in pairing_problems(clean, sampled, ["fallback"])] == ["u3", "u1"]
    assert pairing_problems([], sampled, ["fallback"]) == ["fallback"]


def test_sft_record_uses_the_rl_prompt():
    rec = to_sft_record({"uid": "u", "kind": "clean", "problem": "What is 1+1?", "response": "\\boxed{2}",
                         "detected": None})
    assert rec["prompt"][-1]["role"] == "user" and rec["prompt"][-1]["content"].startswith("What is 1+1?")
    assert "\\boxed{}" in rec["prompt"][-1]["content"]
    assert rec["completion"] == [{"role": "assistant", "content": "\\boxed{2}"}]


def test_sft_config_keys_are_valid():
    from trl import SFTConfig

    names = {f.name for f in dataclasses.fields(SFTConfig)}
    for local in (False, True):
        cfg = load_sft_config(local_test=local)
        assert set(cfg["trainer"]) <= names, set(cfg["trainer"]) - names
        for rotation in range(3):
            assert 0 < sum(family_shares(cfg["data"], rotation).values()) < 1
    assert load_sft_config()["data"]["seeding"] == "rotated"
    assert load_sft_config(local_test=True)["data"]["seeding"] == "uniform"
    full = load_sft_config()
    assert full["trainer"]["max_length"] >= 512 + full["data"]["max_new_tokens"]
    assert (
        full["data"]["set_size"] == 2000
        and full["data"]["clean_pool"] != load_sft_config(local_test=True)["data"]["clean_pool"]
    )
    assert full["propensity"]["max_new_tokens"] == grpo.load_grpo_config()["trainer"]["max_completion_length"]


def test_risk_table_multiplies_sampled_propensity_and_fpr(tmp_path):
    fpr = tmp_path / "fpr.csv"
    pd.DataFrame(
        [
            {"judge": "j", "family": "gibberish", "fpr": 0.5},
            {"judge": "j", "family": "simple_words", "fpr": 0.1},
            {"judge": "j", "family": "gibberish_placeholder", "fpr": 0.9},
        ]
    ).to_csv(fpr, index=False)
    prop = pd.DataFrame([{"pool": "hard", "family": "gibberish", "count": 1, "n": 100, "p_hat": 0.01},
                         {"pool": "hard", "family": "simple_words", "count": 9, "n": 100, "p_hat": 0.1}])
    r = risk_table(prop, fpr)
    assert list(r["family"]) == ["simple_words", "gibberish"]           # 0.1 x 0.1 > 0.01 x 0.5
    assert r["risk"].tolist() == pytest.approx([0.01, 0.005]) and r["rank_in_judge_pool"].tolist() == [1.0, 2.0]


def test_grpo_accepts_a_seeded_policy_and_a_tag():
    a = grpo.parse_args(["--judge", "x", "--pool", "hard", "--policy-model", "runs/sft/s/model", "--tag", "s",
                         "--risk-csv", "r.csv"])
    assert a.policy_model == "runs/sft/s/model" and a.tag == "s" and a.risk_csv == "r.csv"
    assert grpo.parse_args(["--judge", "x", "--pool", "hard"]).risk_csv == str(grpo.RISK_SCORES)


def test_seeding_design_helpers():
    from src.analysis.seeding_design import auroc, evaluate, rotated_units, spearman

    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert auroc([0.9, 0.8, 0.1], [True, False, False]) == 1.0 and auroc([0.5, 0.5], [True, False]) == 0.5
    fpr = {f: 0.1 for f in FAMILIES}
    units = rotated_units(fpr, {f: 1.0 for f in FAMILIES}, [0.003, 0.012, 0.04])
    assert len(units) == 3 * len(FAMILIES)
    for fam in FAMILIES:                                   # Latin square: every family at every tier once
        assert sorted(u["p"] for u in units if u["family"] == fam) == [0.003, 0.012, 0.04]
    e = evaluate([{"p": 0.04, "fpr": 0.4}, {"p": 0.003, "fpr": 0.4}, {"p": 0.04, "fpr": 0.02}], threshold=1.4)
    assert e["positives"] == 1 and e["auc_fpr"] == 0.75 and e["auc_p"] == 0.75


@needs_data
def test_seeding_strings_follow_the_rule_and_cover_all_problems():
    from src.audit.string_fpr import GIBBERISH, build_items, seeding_strings

    strings = seeding_strings()
    assert set(strings) == set(FAMILIES)
    assert strings["empty_symbols"] == "{" and strings["simple_words"] == "True"
    assert strings["reasoning_opener"] == "Thought process:" and strings["gibberish"] == GIBBERISH
    items = build_items()
    assert len(items) == 471 * len(FAMILIES) and len({it["key"] for it in items}) == len(items)
    assert all(it["detected_family"] == it["family"] for it in items)
    gib = [it["response"] for it in items if it["family"] == "gibberish"]
    assert len(set(gib)) == 471                                     # a fresh random string per problem
    assert len(build_items(limit_questions=3)) == 3 * len(FAMILIES)


@needs_data
def test_single_string_seeding_uses_one_string_per_family():
    from src.audit.string_fpr import seeding_strings

    problems = [{"uid": "u", "problem": "p", "answer": "1"}]
    ex = seeded_examples(problems, 4, seed=0, strings="single")
    strings = seeding_strings()
    for e in ex:
        if e["kind"] != "gibberish":
            assert e["response"] == strings[e["kind"]]
    assert load_sft_config()["data"]["strings"] == "single"


def test_risk_only_recomputes_from_the_saved_propensity(tmp_path, monkeypatch):
    import subprocess
    import sys

    d = tmp_path / "prop"
    d.mkdir()
    pd.DataFrame([{"pool": "hard", "family": "empty_symbols", "count": 10, "n": 100, "p_hat": 0.1}]).to_csv(
        d / "propensity_sampled.csv", index=False)
    fpr = tmp_path / "fpr.csv"
    pd.DataFrame([{"judge": "a", "family": "empty_symbols", "fpr": 0.5},
                  {"judge": "b", "family": "empty_symbols", "fpr": 0.2}]).to_csv(fpr, index=False)
    subprocess.run([sys.executable, "-m", "src.audit.sampled_propensity", "--risk-only", "--run-dir", str(d),
                    "--fpr", str(fpr)], check=True, capture_output=True)
    r = pd.read_csv(d / "risk_scores.csv")
    assert sorted(r["judge"]) == ["a", "b"] and r.set_index("judge")["risk"].to_dict() == pytest.approx(
        {"a": 0.05, "b": 0.02}
    )
