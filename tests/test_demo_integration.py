"""The demo through the running app: launch it on a free port and call every endpoint with gradio_client.

Run only these with `pytest -m integration`.
"""

import importlib.util
import socket
import time

import pytest

from src.paths import ROOT

pytest.importorskip("gradio")
pytest.importorskip("gradio_client")
if not (ROOT / "demo" / "data" / "runs.csv").exists():
    pytest.skip("build demo/data first (python -m src.demo.build_data)", allow_module_level=True)

from gradio_client import Client  # noqa: E402

pytestmark = pytest.mark.integration
BUDGET_S = 60
R1, HARD, ROTATION_1 = "R1-Distill-Verifier-1.5B", "hard", "rotation 1"
R1_RUN = "R1-Distill-Verifier-1.5B, hard pool, seed 1: reasoning_opener at step 40"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def demo():
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location("demo_app_integration", ROOT / "demo" / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    blocks = app.build_app()
    port = free_port()
    blocks.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True)
    try:
        client = Client(f"http://127.0.0.1:{port}/", verbose=False)
        yield app, client, port, started
    finally:
        blocks.close()


def label(options, prefix):
    return next(k for k in options if k.startswith(prefix))


def rows(table):
    """A Dataframe or plot value as the client returns it: {'headers' or 'columns': [...], 'data': [[...]]}."""
    cols = table.get("headers") or table.get("columns")
    return [dict(zip(cols, r)) for r in table["data"]]


def test_every_callback_has_one_stable_endpoint(demo):
    app, client, _, _ = demo
    api = client.view_api(return_format="dict", print_info=False)
    assert set(api["named_endpoints"]) == {f"/{n}" for n in app.API_NAMES} and not api["unnamed_endpoints"]


def test_before_training_tab(demo):
    app, client, _, _ = demo
    bars, table, outcome = client.predict(label(app.JUDGE_LABELS, R1), HARD, label(app.SEEDINGS, ROTATION_1),
                                          api_name="/audit")
    ranked = rows(table)
    assert len(ranked) == 14 and ranked[0]["rank"] == 1 and ranked[0]["family"] == "reasoning_opener"
    assert rows(bars)[0]["family"] == "reasoning_opener"
    assert "**reasoning_opener** was exploited from step 40: ranked **#1** by exposure" in outcome


def test_during_training_tab(demo):
    _, client, _, _ = demo
    e, share, rates, summary, step, responses = client.predict(R1_RUN, api_name="/replay")
    assert "**reasoning_opener** from step 40" in summary and step["value"] == 40
    curves = rows(e)
    families = {r["family"] for r in curves}
    assert "reasoning_opener" in families and len(curves) == 31 * len(families)
    assert len(rows(share)) == 300 * len(families) and {r["rate"] for r in rows(rates)} == {
        "gold accuracy", "judge acceptance", "accepted but wrong"}
    at_40 = client.predict(R1_RUN, 40, api_name="/samples")
    assert rows(at_40) == rows(responses) and len(rows(at_40)) >= 1


def test_predictions_recorded_before_training_tab(demo):
    app, client, _, _ = demo
    assert len(app.PREREG_LABELS) == 10
    for name, run in app.PREREG_LABELS.items():
        winner = app.PREREG.set_index("run").loc[run, "winner"]
        text, plot, responses = client.predict(name, api_name="/prereg_view")
        assert f"**What happened:** {winner} was exploited first" in text
        assert {"predicted step", "90% window", winner} <= {r["series"] for r in rows(plot)}
        assert len(rows(responses)) >= 1


def test_audit_your_own_judge_tab(demo):
    app, client, _, _ = demo
    table = client.predict(label(app.JUDGE_LABELS, R1), HARD, label(app.SEEDINGS, ROTATION_1),
                           api_name="/own_prefill")
    assert len(table["data"]) == 14
    bars, ranking, text = client.predict(table, "", api_name="/own_rank")
    assert rows(ranking)[0]["family"] == "reasoning_opener" and "step **39**, 90% window 21 to 73" in text
    _, ranking, text = client.predict(table, "0.235", api_name="/own_rank")
    assert "EA" in rows(ranking)[0] and "expected-advantage score, reasoning_opener" in text


def test_bad_input_gives_a_message_and_the_app_keeps_serving(demo):
    app, client, _, _ = demo
    table = client.predict(label(app.JUDGE_LABELS, R1), HARD, label(app.SEEDINGS, ROTATION_1),
                           api_name="/own_prefill")
    measured, table["data"][0][1] = table["data"][0][1], 1.5                      # an FPR above 1
    bars, ranking, text = client.predict(table, "", api_name="/own_rank")
    assert text.startswith("**Cannot rank these values:** FPR must be between 0 and 1: empty_symbols (1.5)")
    assert not rows(ranking) and not rows(bars)
    table["data"][0][1] = measured
    assert "q0 must be strictly between 0 and 1" in client.predict(table, "1", api_name="/own_rank")[2]
    with pytest.raises(Exception, match="not in the list of choices"):          # Gradio rejects unknown labels
        client.predict("no such run", api_name="/replay")
    assert "reasoning_opener" in client.predict(R1_RUN, api_name="/replay")[3]   # and still serves


def test_results_tab(demo):
    _, _, port, _ = demo
    import httpx
    config = httpx.get(f"http://127.0.0.1:{port}/config", timeout=10).json()
    texts = [c["props"].get("value") or "" for c in config["components"] if c.get("type") == "markdown"]
    results = next(t for t in texts if "Confirmatory tests" in t)
    assert "exposure (256 × p̂ × FPR)" in results and "Holm" in results and "risk score" not in results


def test_the_whole_module_takes_under_a_minute(demo):
    *_, started = demo
    assert time.perf_counter() - started < BUDGET_S
