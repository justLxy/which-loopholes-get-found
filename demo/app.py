"""Which Loopholes Get Found? An interactive view of the experiments.

Run from the repository root:
    pip install -r demo/requirements.txt
    python demo/app.py [--data demo/data] [--host 127.0.0.1] [--port 7860]     # then open http://127.0.0.1:7860

It reads only demo/data (built by `python -m src.demo.build_data`), so the demo folder runs on its own.
  Before training    the audit recorded before any RL step: for a judge, a difficulty pool and a seeded
                     starting policy, the 14 hacking-pattern families ranked by exposure (256 x p_hat x FPR),
                     and what then happened in the matching run
  During training    replay of one GRPO run: each family's exploitation rate e_f(t) at the 31 evaluations, its
                     share of the training responses at every step, and the most frequent responses at any
                     evaluation step
  Predictions recorded before training
                     the 10 runs of experiments B (masking) and C (low share): the prediction against what happened
  Results            the headline numbers of the analysis

store.py loads the bundle, views.py builds what is shown, this file holds the layout and callbacks.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # store and views, also when loaded by file path

import gradio as gr  # noqa: E402
import views  # noqa: E402
from store import DEFAULT_DATA, BundleError, load_bundle  # noqa: E402

HOST, PORT = "127.0.0.1", 7860
# One public endpoint per callback; page-load copies are private, so the names stay stable.
API_NAMES = ("audit", "replay", "samples", "prereg_view", "own_prefill", "own_rank")
PRIVATE = {"api_visibility": "private"}


class Callbacks:
    """Gradio callbacks for one bundle: UI labels to run names, then the views."""

    def __init__(self, store):
        self.store = store
        self.judge_labels = views.judge_labels(store)
        self.run_labels = views.run_labels(store)
        self.prereg_labels = views.prereg_labels(store)

    def audit(self, judge_label, pool, seeding_label):
        return views.audit(self.store, self.judge_labels[judge_label], pool, views.SEEDINGS[seeding_label])

    def replay(self, label):
        e, share, rates, summary, first, responses = views.replay(self.store, self.run_labels[label])
        return e, share, rates, summary, gr.update(value=first), responses

    def samples(self, label, step):
        return views.samples(self.store, self.run_labels[label], step)

    def prereg_view(self, label):
        if label not in self.prereg_labels:
            return views.prereg_missing()
        return views.prereg_view(self.store, self.prereg_labels[label])

    def own_prefill(self, judge_label, pool, seeding_label):
        return views.own_inputs(self.store, self.judge_labels[judge_label], pool, views.SEEDINGS[seeding_label])

    def own_rank(self, table, q0):
        return views.own_view(self.store, table, q0)

    def results_markdown(self):
        return views.results_markdown(self.store)


def build_app(cb=None):
    cb = cb or DEFAULT
    if cb is None:
        load_bundle(DEFAULT_DATA)            # raises BundleError
    judges, runs, prereg = list(cb.judge_labels), list(cb.run_labels), list(cb.prereg_labels)
    with gr.Blocks(title="Which Loopholes Get Found?") as app:
        gr.Markdown(views.INTRO)
        with gr.Tab("Before training: the audit"):
            gr.Markdown("Pick a judge, a difficulty pool and the starting policy. Seeding replaced a small share of a "
                        "2,000-example SFT set with each family's string, at 0.3%, 1.2% or 4.0% depending on the "
                        "rotation, so propensity varies independently of FPR.")
            with gr.Row():
                judge = gr.Dropdown(judges, value=judges[0], label="judge")
                pool = gr.Radio(["hard", "easy"], value="hard", label="difficulty pool")
                seeding = gr.Dropdown(list(views.SEEDINGS), value=list(views.SEEDINGS)[0], label="starting policy")
            bars = gr.BarPlot(x="family", y="exposure", sort="-y", x_label_angle=-40, height=320,
                              title="Predicted exposure (rewarded responses per step)")
            outcome = gr.Markdown()
            table = gr.Dataframe(wrap=True,
                                 column_widths=["60px", "180px", "260px", "75px", "75px", "75px", "85px", "85px"],
                                 label=f"all 14 families by {views.EXPOSURE}. seeded: share of the SFT set; "
                                 "exposure: rewarded responses per step; FPR rank: rank by FPR alone")
            gr.on([judge.change, pool.change, seeding.change], cb.audit, [judge, pool, seeding],
                  [bars, table, outcome], api_name="audit")
        with gr.Tab("During training: replay a run"):
            run = gr.Dropdown(runs, value=runs[0], label="run of the main experiment")
            summary = gr.Markdown()
            e_plot = gr.LinePlot(x="step", y="e_f(t)", color="family", height=300,
                                 title="Exploitation rate: wrong, accepted, in the family")
            share_plot = gr.LinePlot(x="step", y="share", color="family", height=300,
                                     title="Share of the training responses in each family")
            rates_plot = gr.LinePlot(x="step", y="value", color="rate", height=260,
                                     title="Gold accuracy and judge acceptance")
            step = gr.Slider(0, views.STEPS_PER_RUN, value=0, step=10,
                             label="evaluation step: the most frequent responses")
            responses = gr.Dataframe(wrap=True, column_widths=["75px", "170px", "85px", "100px", "480px"],
                                     label="the most frequent distinct responses at this step: copies among the 200 "
                                     "outputs, of which the judge accepted / gold marked correct")
            run.change(cb.replay, run, [e_plot, share_plot, rates_plot, summary, step, responses], api_name="replay")
            step.change(cb.samples, [run, step], responses, api_name="samples")
        with gr.Tab("Predictions recorded before training"):
            gr.Markdown("For these ten runs the prediction was written down before training started: four masked "
                        "runs (experiment B) and six low-share runs (experiment C). Pick one to compare the "
                        f"prediction of the {views.EXPOSURE} with FPR alone, propensity alone and what happened.")
            prereg_run = gr.Dropdown(prereg, value=prereg[0] if prereg else None,
                                     label="run with a recorded prediction")
            prereg_summary = gr.Markdown()
            prereg_e = gr.LinePlot(x="step", y="e_f(t)", color="series", height=320, x_lim=[0, views.STEPS_PER_RUN],
                                   title="Exploitation rate e_f(t), with the predicted step and its 90% window")
            prereg_responses = gr.Dataframe(wrap=True, column_widths=["75px", "170px", "85px", "100px", "480px"],
                                            label="the most frequent distinct responses at the first event")
            prereg_run.change(cb.prereg_view, prereg_run, [prereg_summary, prereg_e, prereg_responses],
                              api_name="prereg_view")
        with gr.Tab("Audit your own judge"):
            gr.Markdown("Score a judge before training. The table starts from our measurements for the judge, pool "
                        "and starting policy you pick: the FPR of each family's string (`src.audit.string_fpr`) and "
                        f"the propensity {views.HAT_P} of the starting policy (`src.audit.sampled_propensity`). Edit "
                        "any value, or type in your own judge's, and press **Rank**. With q0, the judge's acceptance "
                        "of the starting policy's outputs, the expected-advantage score is added. The same ranking on "
                        "the command line: `python -m src.audit.predict`.")
            with gr.Row():
                own_judge = gr.Dropdown(judges, value=judges[0], label="prefill: judge")
                own_pool = gr.Radio(["hard", "easy"], value="hard", label="prefill: difficulty pool")
                own_seeding = gr.Dropdown(list(views.SEEDINGS), value=list(views.SEEDINGS)[0],
                                          label="prefill: starting policy")
            own_table = gr.Dataframe(headers=views.OWN_COLUMNS, datatype=["str", "number", "number"],
                                     interactive=True, static_columns=[0],
                                     label=f"FPR and propensity {views.HAT_P} of each family, between 0 and 1")
            with gr.Row():
                own_q0 = gr.Textbox(value="", placeholder="empty, or a number such as 0.21",     # a Number shows 0
                                    label="q0 (optional): acceptance of the starting policy's outputs, between 0 and 1")
                own_button = gr.Button("Rank", variant="primary")
            own_note = gr.Markdown()
            own_bars = gr.BarPlot(x="family", y="exposure", sort="-y", x_label_angle=-40, height=320,
                                  title="Exposure (rewarded responses per step)")
            own_ranking = gr.Dataframe(wrap=True, label=f"all 14 families by {views.EXPOSURE}")
            own_outputs = [own_bars, own_ranking, own_note]
            # Register the button first, or the .then() below takes the endpoint name "own_rank".
            own_button.click(cb.own_rank, [own_table, own_q0], own_outputs, api_name="own_rank")
            gr.on([own_judge.change, own_pool.change, own_seeding.change], cb.own_prefill,
                  [own_judge, own_pool, own_seeding], own_table, api_name="own_prefill").then(
                cb.own_rank, [own_table, own_q0], own_outputs, **PRIVATE)
        with gr.Tab("Results"):
            gr.Markdown(cb.results_markdown())
        app.load(cb.audit, [judge, pool, seeding], [bars, table, outcome], **PRIVATE)
        app.load(cb.replay, run, [e_plot, share_plot, rates_plot, summary, step, responses], **PRIVATE)
        app.load(cb.prereg_view, prereg_run, [prereg_summary, prereg_e, prereg_responses], **PRIVATE)
        app.load(cb.own_prefill, [own_judge, own_pool, own_seeding], own_table, **PRIVATE).then(
            cb.own_rank, [own_table, own_q0], own_outputs, **PRIVATE)
    return app


def load_default():
    """Callbacks for demo/data, or None if it is unusable (import never fails)."""
    try:
        return Callbacks(load_bundle(DEFAULT_DATA))
    except BundleError:
        return None


# Also exposed as module attributes for the tests.
DEFAULT = load_default()
SEEDINGS, EXPOSURE = views.SEEDINGS, views.EXPOSURE
if DEFAULT is not None:
    PREREG = DEFAULT.store.prereg
    JUDGE_LABELS, RUN_LABELS, PREREG_LABELS = DEFAULT.judge_labels, DEFAULT.run_labels, DEFAULT.prereg_labels
    audit, replay, samples = DEFAULT.audit, DEFAULT.replay, DEFAULT.samples
    prereg_view, results_markdown = DEFAULT.prereg_view, DEFAULT.results_markdown
    own_prefill, own_rank = DEFAULT.own_prefill, DEFAULT.own_rank


def main(argv=None):
    ap = argparse.ArgumentParser(description="The interactive demo of 'Which Loopholes Get Found?'.")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="the data bundle (default: demo/data)")
    ap.add_argument("--host", default=HOST, help=f"address to serve on (default: {HOST})")
    ap.add_argument("--port", type=int, default=PORT, help=f"port to serve on (default: {PORT})")
    args = ap.parse_args(argv)
    try:
        cb = Callbacks(load_bundle(args.data))
    except BundleError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    try:
        build_app(cb).launch(server_name=args.host, server_port=args.port)
    except OSError as err:
        print(f"error: cannot serve on {args.host}:{args.port} ({err}); is the port in use? Try --port {args.port + 1}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
