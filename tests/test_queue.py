"""The main-experiment queue: plan, resumption and result packing, without running anything."""

import json
from collections import Counter

from src.train import queue


def test_plan_matches_the_protocol():
    checkpoints, runs = queue.load_plan()
    assert [c["name"] for c in checkpoints] == ["main-rot0", "main-clean0", "main-rot1", "main-clean1",
                                                "main-rot2", "main-clean2"]
    assert len(runs) == 36 and len({queue.run_name(r) for r in runs}) == 36
    v1, added = runs[:33], runs[33:]
    assert [r["seed"] for r in v1] == sorted(r["seed"] for r in v1)              # protocol-v1: seed by seed
    assert [(r["judge"], r["pool"], r["seed"]) for r in added] == [
        ("qwen3p6_35b_a3b_fp8", "hard", s) for s in (0, 1, 2)]                   # then the added judge
    by_tag = Counter(r["tag"] for r in runs)
    assert by_tag == {"main": 33, "main-ctrl": 3}
    for r in runs:
        assert r["checkpoint"] == (f"main-clean{r['seed']}" if r["tag"] == "main-ctrl" else f"main-rot{r['seed']}")
    ctrl = [r for r in runs if r["tag"] == "main-ctrl"]
    assert {(r["judge"], r["pool"]) for r in ctrl} == {("qwen2p5_math_1p5b_instruct", "hard")}
    held_out = [r for r in runs if r["judge"] in ("general_verifier", "qwen2p5_7b_instruct")]
    assert len(held_out) == 6 and {r["pool"] for r in held_out} == {"hard"}


def test_plan_can_be_split_by_seed():
    checkpoints, runs = queue.load_plan(seeds=[1, 2])
    assert [c["name"] for c in checkpoints] == ["main-rot1", "main-clean1", "main-rot2", "main-clean2"]
    assert len(runs) == 24 and {r["seed"] for r in runs} == {1, 2}


def test_checkpoint_and_run_commands():
    rot = queue.checkpoint_steps({"name": "main-rot1", "seed": 1, "rotation": 1, "seeding": "rotated"})
    assert [label for label, *_ in rot] == ["seeding set", "SFT", "sampled propensity", "string likelihood"]
    assert "--rotation" in rot[0][2] and rot[0][2][rot[0][2].index("--rotation") + 1] == "1"
    assert rot[1][2][-2:] == ["--seed", "1"] and queue.STRING_FPR in rot[2][2]
    assert [tolerated for *_, tolerated in rot] == [False, False, False, True]
    clean = queue.checkpoint_steps({"name": "main-clean0", "seed": 0, "rotation": None, "seeding": "none"})
    assert clean[0][2][-2:] == ["--seeding", "none"]
    cmd = queue.run_command({"judge": "j", "pool": "hard", "seed": 2, "checkpoint": "main-rot2", "tag": "main"})
    assert cmd[:5] == ["bash", "scripts/run_grpo.sh", "j", "hard", "2"]
    assert cmd[cmd.index("--policy-model") + 1] == "runs/sft/main-rot2/model" and cmd[-2:] == ["--tag", "main"]


def test_run_state_resumes_and_sets_failed_attempts_aside(tmp_path):
    run = {"judge": "j", "pool": "hard", "seed": 0, "checkpoint": "main-rot0", "tag": "main"}
    d = tmp_path / "runs" / "grpo" / queue.run_name(run)
    assert queue.run_state(run, tmp_path) == "pending"
    d.mkdir(parents=True)
    (d / "gpu.txt").write_text("x")
    assert queue.run_state(run, tmp_path) == "pending"                      # no logs yet
    (d / "train_metrics.jsonl").write_text("{}\n")
    assert queue.run_state(run, tmp_path) == "failed attempt"
    moved = queue.set_aside(run, tmp_path)
    assert moved.name.startswith(d.name + ".failed-") and not d.exists()
    d.mkdir()
    for name in ("meta_end.json", "summary.txt", "eval_summary.jsonl"):
        (d / name).write_text("{}")
    assert queue.run_state(run, tmp_path) == "done"
    (d / "rollouts.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"step": 0, "family": "empty_symbols", "accepted": True},
                {"step": 0, "family": "empty_symbols", "accepted": False},
                {"step": 0, "family": None, "accepted": True},
                {"step": 1, "family": "gibberish", "accepted": False},
            ]
        )
        + "\n"
    )
    queue.pack_run(run, tmp_path)
    packed = tmp_path / "results" / "main" / queue.run_name(run)
    assert sorted(p.name for p in packed.iterdir()) == ["eval_summary.jsonl", "family_counts.csv", "meta_end.json",
                                                        "summary.txt"]
    assert (packed / "family_counts.csv").read_text().splitlines() == [
        "step,family,n,accepted", "0,empty_symbols,2,1", "1,gibberish,1,0"]


def test_pack_checkpoint_copies_the_propensity_files(tmp_path):
    src = tmp_path / "runs" / "sft" / "main-rot0"
    (src / "propensity").mkdir(parents=True)
    (src / "data_summary.json").write_text("{}")
    (src / "propensity" / "risk_scores.csv").write_text("judge\n")
    queue.pack_checkpoint({"name": "main-rot0"}, tmp_path)
    assert sorted(p.name for p in (tmp_path / "results" / "main" / "sft-main-rot0").iterdir()) == [
        "data_summary.json", "risk_scores.csv"]
