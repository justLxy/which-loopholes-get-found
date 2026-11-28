"""The demo's callbacks on the committed data bundle (skipped without gradio or demo/data)."""

import importlib.util
import json
import shutil
import subprocess
import sys

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


# ----------------------------------------------------------------------------- start-up, options and wording

def start(*args, cwd=ROOT):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=cwd, timeout=120)


@pytest.mark.parametrize("case", ["missing", "empty", "incomplete"])
def test_a_bad_data_folder_gives_one_line_and_a_non_zero_exit(tmp_path, case):
    data = tmp_path / "data"
    if case != "missing":
        data.mkdir()
    if case == "incomplete":
        (data / "judges.json").write_text("{}")
    out = start(APP, "--data", data)
    assert out.returncode == 2 and out.stdout == ""
    lines = out.stderr.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("error: ") and "python -m src.demo.build_data" in lines[0]
    assert "Traceback" not in out.stderr


def test_a_demo_folder_without_data_imports_and_exits_cleanly(tmp_path):
    """demo/ without data/: import works, start-up explains the fix."""
    for f in ("app.py", "store.py", "views.py"):
        shutil.copy(APP.parent / f, tmp_path / f)
    load = ("import importlib.util; s = importlib.util.spec_from_file_location('a', 'app.py'); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.DEFAULT)")
    out = start("-c", load, cwd=tmp_path)
    assert out.returncode == 0 and out.stdout.strip() == "None" and "Traceback" not in out.stderr
    out = start(tmp_path / "app.py", cwd=tmp_path)
    assert out.returncode == 2 and "is missing" not in out.stderr and "does not exist" in out.stderr
    assert "Traceback" not in out.stderr


def test_host_port_and_data_options(app, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(app.gr.Blocks, "launch", lambda self, **kw: calls.append(kw))
    assert app.main([]) == 0 and calls[-1] == {"server_name": "127.0.0.1", "server_port": 7860}
    assert app.main(["--host", "0.0.0.0", "--port", "7999", "--data", str(ROOT / "demo" / "data")]) == 0
    assert calls[-1] == {"server_name": "0.0.0.0", "server_port": 7999}


def test_a_port_in_use_gives_a_message(app, monkeypatch, capsys):
    def busy(self, **kw):
        raise OSError("Cannot find empty port in range: 7860-7860")
    monkeypatch.setattr(app.gr.Blocks, "launch", busy)
    assert app.main([]) == 1
    err = capsys.readouterr().err
    assert "port in use" in err and "--port 7861" in err


def test_risk_score_is_gone_from_the_ui(app):
    """The UI uses the paper's term, exposure."""
    config = json.dumps(app.build_app().get_config_file(), ensure_ascii=False)
    assert "risk score" not in config and "exposure (256 × p̂ × FPR)" in config
    assert "risk score" not in app.results_markdown() and "exposure (256 × p̂ × FPR)" in app.results_markdown()
    for label in app.JUDGE_LABELS:
        for pool in ("hard", "easy"):
            for seeding in app.SEEDINGS:
                assert "risk score" not in app.audit(label, pool, seeding)[2]


def test_events_come_from_the_structured_run_files(app):
    """The audit outcome lists every event from runs/<run>.json."""
    s = app.DEFAULT.store
    for _, run in s.runs.iterrows():
        events = s.run(run["run"])["events"]
        assert [e["family"] for e in events][:1] == ([run["first_event_family"]] if run["first_event_family"] else [])
        label = next(k for k, v in app.JUDGE_LABELS.items() if v == run["judge"])
        seeding = next(k for k, v in app.SEEDINGS.items() if v == run["checkpoint"])
        text = app.audit(label, run["pool"], seeding)[2]
        for e in events:
            assert f"**{e['family']}** was exploited from step {e['step']}:" in text
