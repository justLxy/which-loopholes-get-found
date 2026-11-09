"""Experiment C: custom seeding, the plan, the pre-run check, the predictions and the paired outcomes."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analysis import lowshare_outcomes
from src.detectors import FAMILIES
from src.paths import ROOT
from src.train import lowshare, seed_data


# ----------------------------------------------------------------------------- seeding

def test_custom_shares_seed_only_the_named_families():
    shares = seed_data.parse_shares(["adversarial_prefixes=0.02", "answer_explanation=0.01"])
    full = seed_data.family_shares({"seeding": "custom"}, custom=shares)
    assert set(full) == set(FAMILIES)
    assert {f for f, v in full.items() if v > 0} == {"adversarial_prefixes", "answer_explanation"}
    counts = seed_data.seed_counts(2000, full)
    assert counts["adversarial_prefixes"] == 40 and counts["answer_explanation"] == 20 and sum(counts.values()) == 60


@pytest.mark.parametrize("bad", [["no_such_family=0.01"], ["empty_symbols=0"], ["empty_symbols=1.5"],
                                 ["empty_symbols=0.01", "empty_symbols=0.02"]])
def test_custom_shares_are_validated(bad):
    with pytest.raises(ValueError):
        seed_data.parse_shares(bad)


def test_custom_seeding_needs_shares():
    with pytest.raises(SystemExit):
        seed_data.family_shares({"seeding": "custom"})


def test_clean_subset_keeps_the_main_behaviour_without_a_data_seed():
    pool = [{"uid": i} for i in range(100)]
    assert seed_data.clean_subset(pool, 50, 40) == pool[:40]
    a = seed_data.clean_subset(pool, 50, 40, data_seed=0)
    b = seed_data.clean_subset(pool, 50, 40, data_seed=1)
    assert len(a) == len(b) == 40 and a != b
    assert all(x["uid"] < 50 for x in a + b)
    assert [x["uid"] for x in a] == sorted(x["uid"] for x in a)       # pool order is kept
    assert a == seed_data.clean_subset(pool, 50, 40, data_seed=0)       # reproducible


# ----------------------------------------------------------------------------- plan

def test_plan_has_six_checkpoints_in_seed_order_at_three_percent():
    cfg, cks = lowshare.load_plan()
    assert [(c["setting"], c["seed"]) for c in cks] == [("F1", 0), ("F2", 0), ("P1", 0), ("F1", 1), ("F2", 1), ("P1", 1)]
    for c in cks:
        assert len(c["shares"]) == 2 and abs(sum(c["shares"].values()) - 0.03) < 1e-9
        assert c["pool"] == "hard" and c["name"] == f"low-{c['setting']}-s{c['seed']}"
        fb = lowshare.fallback_of(cfg, c)
        assert fb["name"] == c["name"] + "b" and fb["fallback"] and set(fb["shares"]) == set(c["shares"])
    assert {c["baseline"] for c in cks} == {"fpr", "propensity"}


def test_planned_judges_and_strings_have_audited_fprs():
    cfg, cks = lowshare.load_plan()
    fpr = pd.read_csv(ROOT / cfg["string_fpr"])
    for c in cks:
        got = fpr[fpr["judge"] == c["judge"]]
        assert set(c["shares"]) <= set(got["family"]) and len(got) == len(FAMILIES)


def test_checkpoint_steps_seed_measure_and_judge_the_hard_pool():
    cfg, cks = lowshare.load_plan()
    steps = {label: cmd for label, _, cmd in lowshare.checkpoint_steps(cfg, cks[0])}
    assert list(steps) == ["seeding set", "SFT", "sampled propensity", "q0"]
    seeding = steps["seeding set"]
    assert "--shares" in seeding and "adversarial_prefixes=0.02" in seeding and "answer_explanation=0.01" in seeding
    assert seeding[seeding.index("--data-seed") + 1] == "0"
    assert steps["SFT"][steps["SFT"].index("--seed") + 1] == "0"
    assert steps["sampled propensity"][-2:] == ["--pools", "hard"]
    assert steps["q0"][steps["q0"].index("--judge") + 1] == "qwen2p5_math_1p5b_instruct"


def test_runs_use_the_main_launcher_with_the_checkpoint_and_tag():
    _, cks = lowshare.load_plan()
    run = lowshare.run_of(cks[2])
    cmd = lowshare.queue.run_command(run)
    assert lowshare.queue.run_name(run) == "qwen2p5_math_1p5b_instruct-hard-s0-low-P1"
    assert cmd[cmd.index("--policy-model") + 1] == "runs/sft/low-P1-s0/model"
    assert cmd[cmd.index("--tag") + 1] == "low-P1"


# ----------------------------------------------------------------------------- pre-run check

def fake_checkpoint(root, cfg, ck, p, accept_extra=0.0, gold=0.15, seed=0):
    """judged.csv and risk_scores.csv of a checkpoint whose 4,000 hard-pool samples have family rates p."""
    rng = np.random.default_rng(seed)
    fpr = pd.read_csv(ROOT / cfg["string_fpr"])
    fpr = fpr[fpr["judge"] == ck["judge"]].set_index("family")["fpr"]
    rows = []
    for u in range(250):
        for j in range(16):
            r = rng.random()
            fam, acc = "", False
            edge = 0.0
            for f, pf in p.items():
                if edge <= r < edge + pf:
                    fam = f
                edge += pf
            if fam:
                acc = rng.random() < fpr[fam]
                g = False
            else:
                g = rng.random() < gold
                acc = g or rng.random() < accept_extra
            rows.append({"uid": f"u{u}", "sample_idx": j, "family": fam, "gold_correct": g, "truncated": False,
                         "accepted": acc, "parse_fail": False})
    judged = pd.DataFrame(rows)
    d = lowshare.checkpoint_results(cfg, ck["name"], root)
    d.mkdir(parents=True, exist_ok=True)
    judged.to_csv(d / "judged.csv", index=False)
    counts = judged["family"].value_counts()
    risk = pd.DataFrame([{"judge": ck["judge"], "family": f, "fpr": fpr[f], "pool": "hard",
                          "count": int(counts.get(f, 0)), "n": len(judged),
                          "p_hat": (counts.get(f, 0) + 0.5) / (len(judged) + 7.5)} for f in FAMILIES])
    risk["risk"] = risk["p_hat"] * risk["fpr"]
    risk.to_csv(d / "risk_scores.csv", index=False)
    return judged


@pytest.fixture
def root(tmp_path):
    for rel in ("results/stage2/string_fpr.csv", "results/analysis/event_timing.csv"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    return tmp_path


def test_check_passes_when_exposure_and_fpr_alone_disagree_clearly(root):
    cfg, cks = lowshare.load_plan()
    f1 = cks[0]
    fake_checkpoint(root, cfg, f1, {"adversarial_prefixes": 0.021, "answer_explanation": 0.008})
    row = lowshare.check_checkpoint(cfg, f1, root)
    assert row["exposure_favourite"] == "adversarial_prefixes"
    assert row["fpr_alone_favourite"] == "answer_explanation" == row["baseline_favourite"]
    assert row["propensity_alone_favourite"] == "adversarial_prefixes"
    assert row["ea_favourite"] == "adversarial_prefixes"
    assert 0.1 < row["q0"] < 0.25 and row["propensity_matches_audit"]
    assert row["exposure_top"] > 2.5 and row["margin"] > 1.3 and row["stability"] > 0.9
    assert row["passed"]


def test_check_fails_when_the_fpr_favourite_also_leads_exposure(root):
    cfg, cks = lowshare.load_plan()
    f1 = cks[0]
    fake_checkpoint(root, cfg, f1, {"adversarial_prefixes": 0.012, "answer_explanation": 0.016})
    row = lowshare.check_checkpoint(cfg, f1, root)
    assert row["exposure_favourite"] == "answer_explanation"
    assert not row["ok_disagreement"] and not row["passed"]


def test_check_fails_an_unstable_margin(root):
    cfg, cks = lowshare.load_plan()
    f1 = cks[0]
    # adversarial prefixes at 0.609 x 1.6% against answer explanation at 0.898 x 1.0%: a margin near 1.1
    fake_checkpoint(root, cfg, f1, {"adversarial_prefixes": 0.016, "answer_explanation": 0.0098})
    row = lowshare.check_checkpoint(cfg, f1, root)
    assert not row["ok_margin"] and row["stability"] < 0.9 and not row["passed"]


def test_p_setting_needs_the_propensity_favourite_to_lead_in_p_hat(root):
    cfg, cks = lowshare.load_plan()
    p1 = cks[2]
    fake_checkpoint(root, cfg, p1, {"answer_explanation": 0.0125, "reasoning_opener": 0.017})
    row = lowshare.check_checkpoint(cfg, p1, root)
    assert row["exposure_favourite"] == "answer_explanation"
    assert row["propensity_alone_favourite"] == "reasoning_opener" == row["baseline_favourite"]
    assert row["propensity_ratio"] > 1.2 and row["ok_propensity_ratio"]
    # Thought process below answer explanation: propensity alone now names the exposure favourite
    fake_checkpoint(root, cfg, p1, {"answer_explanation": 0.0125, "reasoning_opener": 0.0085}, seed=1)
    row = lowshare.check_checkpoint(cfg, p1, root)
    assert row["propensity_alone_favourite"] == row["exposure_favourite"] == "answer_explanation"
    assert not row["ok_propensity_ratio"] and not row["ok_disagreement"] and not row["passed"]


def test_ea_check_fails_when_q0_is_close_to_the_favourites_fpr(root):
    cfg, cks = lowshare.load_plan()
    f1 = cks[0]
    # a starting policy accepted half of the time: adversarial prefixes (0.609) barely beats an average response
    fake_checkpoint(root, cfg, f1, {"adversarial_prefixes": 0.021, "answer_explanation": 0.008}, accept_extra=0.45)
    row = lowshare.check_checkpoint(cfg, f1, root)
    assert row["q0"] > 0.5 and row["ea_favourite"] == "answer_explanation"
    assert not row["ok_ea_agrees"] and not row["passed"]


def test_bootstrap_is_reproducible(root):
    cfg, cks = lowshare.load_plan()
    fake_checkpoint(root, cfg, cks[0], {"adversarial_prefixes": 0.021, "answer_explanation": 0.008})
    a = lowshare.check_checkpoint(cfg, cks[0], root)
    b = lowshare.check_checkpoint(cfg, cks[0], root)
    assert a["stability"] == b["stability"] and a["ea_stability"] == b["ea_stability"]


def test_predictions_use_the_fallback_only_when_the_primary_fails(root):
    cfg, cks = lowshare.load_plan()
    f1, f2 = cks[0], cks[1]
    fake_checkpoint(root, cfg, f1, {"adversarial_prefixes": 0.012, "answer_explanation": 0.016})        # fails
    fake_checkpoint(root, cfg, lowshare.fallback_of(cfg, f1), {"adversarial_prefixes": 0.023, "answer_explanation": 0.007})
    fake_checkpoint(root, cfg, f2, {"empty_symbols": 0.021, "reasoning_opener": 0.008})                  # passes
    table, (a, b, sd) = lowshare.prediction_table(cfg, cks, root)
    assert list(table["checkpoint"]) == ["low-F1-s0b", "low-F2-s0"]
    assert list(table["run"]) == ["qwen2p5_math_1p5b_instruct-hard-s0-low-F1", "qwen2p5_7b_instruct-hard-s0-low-F2"]
    assert (table["step_low"] < table["step"]).all() and (table["step"] < table["step_high"]).all()
    assert b < 0 and sd > 0
    assert lowshare.same_predictions(table, table.copy())
    changed = table.copy()
    changed.loc[0, "exposure_favourite"] = "answer_explanation"
    assert not lowshare.same_predictions(table, changed)
    checks = lowshare.check_table(cfg, cks, root)
    assert list(checks["checkpoint"]) == ["low-F1-s0", "low-F1-s0b", "low-F2-s0"]
    assert list(checks["passed"]) == [False, True, True]


# ----------------------------------------------------------------------------- outcomes

def fake_run(path, e, events, counts):
    """eval_summary.jsonl from {step: {family: e_f}}, events_tau0.05.json and family_counts.csv."""
    path.mkdir(parents=True)
    with (path / "eval_summary.jsonl").open("w") as f:
        for step, rates in sorted(e.items()):
            f.write(json.dumps({"step": step, "accept_rate": 0.2, **{f"e/{k}": v for k, v in rates.items()}}) + "\n")
    ev = {fam: {"event": fam in events, "time": events.get(fam, 300), "baseline": 0.0, "peak": 0.0} for fam in FAMILIES}
    (path / "events_tau0.05.json").write_text(json.dumps({"tau": 0.05, "consecutive": 2, "events": ev}))
    pd.DataFrame(counts, columns=["step", "family", "n", "accepted"]).to_csv(path / "family_counts.csv", index=False)


def prediction(run, setting, baseline, exp_fav, fpr_fav, prop_fav, shares, exposures):
    return pd.Series({"run": run, "setting": setting, "baseline": baseline, "seed": 0, "checkpoint": f"low-{setting}-s0",
                      "fallback": False, "judge": "j", "q0": 0.2, "exposure_favourite": exp_fav, "exposure_top": 3.3,
                      "fpr_alone_favourite": fpr_fav, "propensity_alone_favourite": prop_fav, "ea_favourite": exp_fav,
                      "step_low": 45.0, "step_high": 160.0, "shares": shares,
                      **{f"exposure:{k}": v for k, v in exposures.items()}})


def test_outcome_scores_the_first_event_against_each_favourite(tmp_path):
    e = {0: {"adversarial_prefixes": 0.0, "answer_explanation": 0.0}, 10: {"adversarial_prefixes": 0.02},
         60: {"adversarial_prefixes": 0.3, "answer_explanation": 0.01}, 70: {"adversarial_prefixes": 0.5}}
    fake_run(tmp_path / "r", e, {"adversarial_prefixes": 60},
             [(0, "adversarial_prefixes", 5, 3), (30, "adversarial_prefixes", 20, 12), (60, "adversarial_prefixes", 80, 50)])
    pred = prediction("r", "F1", "fpr", "adversarial_prefixes", "answer_explanation", "adversarial_prefixes",
                      "adversarial_prefixes=0.02;answer_explanation=0.01",
                      {"adversarial_prefixes": 3.3, "answer_explanation": 1.8})
    row = lowshare_outcomes.outcome(pred, tmp_path / "r")
    assert row["event"] and row["winner"] == "adversarial_prefixes" and row["step"] == 60 and row["in_band"]
    assert row["hit_exposure"] and not row["hit_fpr"] and row["hit_propensity"] and row["hit_ea"]
    assert row["rewarded_before"] == 15 and row["sampled_before"] == 25
    assert row["rate_over_exposure"] == pytest.approx(15 / (60 * 3.3))
    assert row["rise:adversarial_prefixes"] == pytest.approx(0.5) and row["rise_favours_exposure"]


def test_outcome_without_an_event_keeps_the_threshold_free_rise(tmp_path):
    e = {0: {"answer_explanation": 0.0, "reasoning_opener": 0.0}, 10: {"answer_explanation": 0.03, "reasoning_opener": 0.01}}
    fake_run(tmp_path / "r", e, {}, [])
    pred = prediction("r", "P1", "propensity", "answer_explanation", "answer_explanation", "reasoning_opener",
                      "answer_explanation=0.013;reasoning_opener=0.017",
                      {"answer_explanation": 2.8, "reasoning_opener": 1.8})
    row = lowshare_outcomes.outcome(pred, tmp_path / "r")
    assert not row["event"] and "winner" not in row and row["rise_favours_exposure"]


def test_summary_pairs_exposure_with_each_baseline_by_run():
    rows = pd.DataFrame([
        {"baseline": "fpr", "event": True, "hit_exposure": True, "hit_fpr": False, "hit_propensity": True, "hit_ea": True,
         "in_band": True, "rewarded_before": 500, "rise_favours_exposure": True},
        {"baseline": "fpr", "event": True, "hit_exposure": True, "hit_fpr": False, "hit_propensity": True, "hit_ea": True,
         "in_band": False, "rewarded_before": 700, "rise_favours_exposure": True},
        {"baseline": "propensity", "event": True, "hit_exposure": False, "hit_fpr": False, "hit_propensity": True,
         "hit_ea": False, "in_band": True, "rewarded_before": 300, "rise_favours_exposure": False},
        {"baseline": "propensity", "event": False, "hit_exposure": None, "hit_fpr": None, "hit_propensity": None,
         "hit_ea": None, "in_band": None, "rewarded_before": np.nan, "rise_favours_exposure": True},
    ])
    s = lowshare_outcomes.summary(rows).set_index("scope")
    a = s.loc["all runs"]
    assert (a["runs"], a["runs_with_event"], a["exposure_correct"], a["events_in_band"]) == (4, 3, 2, 2)
    assert (a["exposure_only_vs_fpr"], a["fpr_only"], a["mcnemar_p_vs_fpr"]) == (2, 0, 0.5)
    assert (a["exposure_only_vs_propensity"], a["propensity_only"]) == (0, 1)
    assert a["runs_without_event"] == 1 and a["no_event_rise_favours_exposure"] == 1
    assert s.loc["F settings (exposure vs FPR alone)", "fpr_correct"] == 0


# ----------------------------------------------------------------------------- q0

def test_q0_items_follow_the_samples_and_carry_the_question():
    from src.audit import q0
    problems = {"a": {"problem": "1+1?", "reference": "2"}, "b": {"problem": "2+2?", "reference": "4"}}
    samples = [{"uid": "b", "response": "{", "truncated": False}, {"uid": "a", "response": "\\boxed{2}", "truncated": True}]
    items = q0.judge_items(samples, problems)
    assert [(i["question"], i["reference"], i["response"], i["truncated"]) for i in items] == \
        [("2+2?", "4", "{", False), ("1+1?", "2", "\\boxed{2}", True)]


def test_q0_summary_counts_all_outputs_and_each_familys_own():
    from src.audit import q0
    judged = pd.DataFrame({"uid": list("aabb"), "sample_idx": [0, 1, 0, 1],
                           "family": ["empty_symbols", "", "empty_symbols", ""],
                           "gold_correct": [False, True, False, False], "truncated": [False] * 4,
                           "accepted": [True, True, False, False], "parse_fail": [False] * 4})
    s = q0.summarize(judged)
    assert (s["n"], s["q0"], s["gold_accuracy"], s["family_share"]) == (4, 0.5, 0.25, 0.5)
    assert s["family_acceptance"] == {"empty_symbols": {"n": 2, "accepted": 1}}
