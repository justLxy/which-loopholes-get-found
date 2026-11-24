"""The demo's callbacks on the committed data bundle (skipped without gradio or demo/data)."""

import importlib.util

import pytest

from src.paths import ROOT

APP = ROOT / "demo" / "app.py"
pytest.importorskip("gradio")
if not (ROOT / "demo" / "data" / "runs.csv").exists():
    pytest.skip("build demo/data first (python -m src.demo.build_data)", allow_module_level=True)


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("demo_app", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_ranks_all_families_and_reports_the_matching_run(app):
    for label in app.JUDGE_LABELS:
        for seeding, checkpoint in app.SEEDINGS.items():
            bars, table, outcome = app.audit(label, "hard", seeding)
            if checkpoint.startswith("main-clean") and app.JUDGE_LABELS[label] == "qwen3p6_35b_a3b_fp8":
                assert table.empty
                continue
            assert len(table) == 14 and list(table["rank"]) == sorted(table["rank"])
    _, _, outcome = app.audit(next(k for k in app.JUDGE_LABELS if k.startswith("R1-Distill-Verifier")), "hard",
                              next(k for k in app.SEEDINGS if k.startswith("rotation 1")))
    assert "reasoning_opener" in outcome and "#1" in outcome


def test_every_run_replays(app):
    for label in app.RUN_LABELS:
        e, share, rates, summary, step, responses = app.replay(label)
        assert len(e) == 31 * e["family"].nunique() and len(share) == 300 * share["family"].nunique()
        assert len(responses) >= 1
    assert "Holm" in app.results_markdown()


def test_every_run_with_a_recorded_prediction_renders(app):
    assert app.PREREG is not None and len(app.PREREG_LABELS) == 10
    assert sum("exposure named the winner" in label for label in app.PREREG_LABELS) == 9
    for label, name in app.PREREG_LABELS.items():
        r = app.PREREG.set_index("run").loc[name]
        text, plot, responses = app.prereg_view(label)
        assert f"**What happened:** {r['winner']} was exploited first, from step **{r['step']}**" in text
        assert f"step **{float(r['predicted_step']):.0f}**, 90% window {float(r['step_low']):.0f} to" in text
        assert f"| {app.EXPOSURE} | {r['first_by_exposure']} | {'yes' if r['hit'] else 'no'} |" in text
        assert f"| FPR alone | {r['first_by_fpr']} | {'yes' if r['first_by_fpr'] == r['winner'] else 'no'} |" in text
        assert f"| propensity alone | {r['first_by_propensity']} |" in text
        assert "risk score" not in text
        curves = plot[~plot["series"].isin(["predicted step", "90% window"])]
        assert {r["first_by_exposure"], r["winner"]} <= set(curves["series"])
        assert len(curves) == 31 * curves["series"].nunique()
        window = plot.loc[plot["series"] == "90% window", "step"].tolist()
        assert window == pytest.approx([r["step_low"]] * 2 + [r["step_high"]] * 2)
        predicted = plot.loc[plot["series"] == "predicted step", "step"].tolist()
        assert predicted == pytest.approx([r["predicted_step"]] * 2)
        assert len(responses) >= 1
    text, _, _ = app.prereg_view(next(k for k in app.PREREG_LABELS if "missed" in k))
    assert "Exposure missed the winner (it named system_prompt_mimicry)" in text and "adversarial_prefixes" in text


def test_an_unknown_prereg_run_gives_a_message(app):
    text, plot, responses = app.prereg_view("no such run")
    assert "python -m src.demo.build_data" in text and plot.empty and responses.empty
