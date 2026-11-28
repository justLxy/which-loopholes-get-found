"""The demo data bundle: top responses, the risk table and a build from the main results."""

import json

import numpy as np
import pandas as pd
import pytest

from src.audit import predict
from src.demo.build_data import MAX_CHARS, PREREG_COLUMNS, build, build_prereg, top_responses, write_timing
from src.detectors import FAMILIES
from src.paths import ROOT
from src.train.masking import Z90, timing_fit

MAIN = ROOT / "results" / "main"
needs_main = pytest.mark.skipif(not MAIN.exists(), reason="needs results/main from the main experiment")


def test_top_responses_counts_distinct_texts_and_cuts_long_ones():
    recs = [{"response": t, "family": f, "accepted": a, "gold_correct": False}
            for t, f, a in [("{", "empty_symbols", True)] * 3 + [("x" * 900, None, False), ("y", None, True)]]
    top = top_responses(recs)
    assert [t["count"] for t in top] == [3, 1, 1] and top[0]["text"] == "{" and top[0]["accepted"] == 3
    assert top[1]["text"].endswith(" [...]") and len(top[1]["text"]) == MAX_CHARS + 6


@needs_main
def test_build_from_the_main_results(tmp_path):
    rows = build(MAIN, tmp_path / "no-logs", tmp_path / "out")          # without run logs: no samples
    assert len(rows) == 36 and not any(r["has_samples"] for r in rows)
    risk = pd.read_csv(tmp_path / "out" / "risk.csv")
    assert set(risk["family"]) == set(FAMILIES)
    assert risk["exposure"].to_numpy() == pytest.approx(256 * risk["risk"].to_numpy())
    top = risk[risk["rank_risk"] == 1].groupby(["checkpoint", "judge", "pool"]).size()
    assert (top >= 1).all()
    rot = risk[risk["seeding"] == "rotated tiers"]
    assert set(rot["tier"]) == {0.003, 0.012, 0.04}
    assert (risk.loc[risk["seeding"] != "rotated tiers", "tier"] == 0).all()
    runs = pd.read_csv(tmp_path / "out" / "runs.csv")
    assert (runs["first_event_family"].notna()).sum() == 12                # runs with a family event
    b = json.loads((tmp_path / "out" / "runs" / f"{runs['run'][0]}.json").read_text())
    assert len(b["eval"]) == 31 and len(b["train_share"]["empty_symbols"]) == 300


# ----------------------------------------------------------------------------- runs with recorded predictions

ANALYSIS = ROOT / "results" / "analysis"
needs_prereg = pytest.mark.skipif(not (ANALYSIS / "masking_outcomes.csv").exists()
                                  or not (ANALYSIS / "lowshare_outcomes.csv").exists(),
                                  reason="needs the outcomes of experiments B and C in results/analysis")


def later_from_outcomes(text):
    """Later events as the two outcomes files write them: 'fam (70)' (masking) or 'fam@50' (low share)."""
    if not isinstance(text, str) or not text:
        return []
    out = []
    for part in text.replace("; ", ",").split(","):
        part = part.strip()
        fam, step = part.split("@") if "@" in part else part.rstrip(")").split(" (")
        out.append((fam.strip(), int(step)))
    return out


@pytest.fixture(scope="module")
def prereg_bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("prereg")
    (out / "runs").mkdir()
    rows = build_prereg(out, out / "no-logs")
    return out, rows


@needs_prereg
def test_prereg_has_the_ten_runs_with_full_curves(prereg_bundle):
    out, rows = prereg_bundle
    prereg = pd.read_csv(out / "prereg.csv", keep_default_na=False)
    assert len(rows) == 10 and list(prereg.columns) == PREREG_COLUMNS
    assert prereg["experiment"].value_counts().to_dict() == {"B": 4, "C": 6}
    for run, kind in zip(prereg["run"], prereg["kind"]):
        b = json.loads((out / "runs" / f"{run}.json").read_text())
        assert b["kind"] == kind and b["samples"] == {}                    # no run logs given
        assert [ev["step"] for ev in b["eval"]] == list(range(0, 301, 10))
        assert all(len(b["train_share"][f]) == 300 for f in FAMILIES)
        assert b["events"][0] == {"family": prereg.set_index("run").loc[run, "winner"],
                                  "step": int(prereg.set_index("run").loc[run, "step"])}


@needs_prereg
def test_prereg_events_agree_with_the_outcomes_files(prereg_bundle):
    out, _ = prereg_bundle
    prereg = pd.read_csv(out / "prereg.csv", keep_default_na=False).set_index("run")
    for name in ("masking_outcomes.csv", "lowshare_outcomes.csv"):
        for _, o in pd.read_csv(ANALYSIS / name).iterrows():
            b = json.loads((out / "runs" / f"{o['run']}.json").read_text())
            events = [(ev["family"], ev["step"]) for ev in b["events"]]
            assert events == [(o["winner"], o["step"])] + later_from_outcomes(o["later_events"])
            assert prereg.loc[o["run"], "winner"] == o["winner"]


@needs_prereg
def test_prereg_matches_the_paper(prereg_bundle):
    """Paper counts (abstract, Appendix D), Table 5 rows and the timing-fit window."""
    _, rows = prereg_bundle
    p = pd.DataFrame(rows)
    b, c = p[p["experiment"] == "B"], p[p["experiment"] == "C"]
    assert b["hit"].sum() == 3 and c["hit"].sum() == 6
    assert (b["hit"] == (b["first_by_exposure"] == b["winner"])).all()
    assert (c["hit"] == (c["first_by_exposure"] == c["winner"])).all()
    assert (b["first_by_fpr"] == b["winner"]).sum() == 0 and (b["first_by_propensity"] == b["winner"]).sum() == 2
    assert p["in_band"].all() and ((p["step_low"] <= p["step"]) & (p["step"] <= p["step_high"])).all()
    a, slope, sd = timing_fit(ANALYSIS / "event_timing.csv")
    mid = a + slope * np.log(p["exposure_top"])
    assert p["predicted_step"].to_numpy() == pytest.approx(np.exp(mid).to_numpy(), rel=1e-9)
    assert p["step_low"].to_numpy() == pytest.approx(np.exp(mid - Z90 * sd).to_numpy(), rel=1e-9)
    assert p["step_high"].to_numpy() == pytest.approx(np.exp(mid + Z90 * sd).to_numpy(), rel=1e-9)
    table5 = {"qwen2p5_math_1p5b_instruct-hard-s0-low-F1": (82, 44, 153, "adversarial_prefixes", 60),
              "qwen2p5_math_1p5b_instruct-hard-s1-low-F1": (81, 43, 151, "adversarial_prefixes", 50),
              "qwen2p5_7b_instruct-hard-s0-low-F2": (62, 33, 117, "empty_symbols", 40),
              "qwen2p5_7b_instruct-hard-s1-low-F2": (59, 32, 111, "empty_symbols", 40),
              "qwen2p5_math_1p5b_instruct-hard-s0-low-P2": (84, 45, 158, "answer_explanation", 60),
              "qwen2p5_math_1p5b_instruct-hard-s1-low-P2": (74, 40, 139, "answer_explanation", 50)}
    for run, (step, low, high, winner, observed) in table5.items():
        r = c.set_index("run").loc[run]
        assert (round(r["predicted_step"]), round(r["step_low"]), round(r["step_high"])) == (step, low, high)
        assert (r["winner"], r["step"]) == (winner, observed)


@needs_prereg
def test_prereg_is_skipped_without_outcomes(tmp_path):
    assert build_prereg(tmp_path, tmp_path, analysis_dir=tmp_path / "nothing") == []
    assert not (tmp_path / "prereg.csv").exists()


@needs_main
@needs_prereg
def test_build_keeps_prereg_runs_out_of_runs_csv(tmp_path):
    build(MAIN, tmp_path / "no-logs", tmp_path / "out")
    runs = pd.read_csv(tmp_path / "out" / "runs.csv")
    prereg = pd.read_csv(tmp_path / "out" / "prereg.csv")
    assert len(runs) == 36 and len(prereg) == 10 and not set(runs["run"]) & set(prereg["run"])
    assert len(list((tmp_path / "out" / "runs").glob("*.json"))) == 46


def test_the_timing_fit_in_the_bundle_is_the_library_s(tmp_path):
    write_timing(tmp_path)
    t = json.loads((tmp_path / "timing.json").read_text())
    (a, b, sd), level = predict.timing()
    assert (t["a"], t["b"], t["sd"], t["z90"], t["level"], t["step_samples"]) == (a, b, sd, predict.Z90, level, 256)
    committed = json.loads((ROOT / "demo" / "data" / "timing.json").read_text())
    assert committed.keys() == t.keys() and committed["source"] == t["source"]
    for key in ("a", "b", "sd", "z90", "level", "step_samples"):
        assert committed[key] == pytest.approx(t[key], rel=1e-12), key   # other numpy versions move the last digit
