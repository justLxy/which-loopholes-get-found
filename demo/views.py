"""What the demo shows: pure functions from the data bundle (a store.Store) to tables, plot data and Markdown.

No Gradio here. Runs are identified by run name; app.py maps UI labels to names.
"""

import math

import pandas as pd

STEPS_PER_RUN = 300
HAT_P = "p̂"
EXPOSURE = f"exposure (256 × {HAT_P} × FPR)"
SCORE_NAMES = {"risk": EXPOSURE, "fpr": "FPR alone", "p_hat": "propensity alone", "static": "static accuracy"}
SEEDINGS = {f"rotation {s}: seeded at 0.3% / 1.2% / 4.0% (seed {s})": f"main-rot{s}" for s in range(3)}
SEEDINGS.update({f"no seeding: 0% control (seed {s})": f"main-clean{s}" for s in range(3)})
JUDGE_ORDER = ["r1_distill_verifier_1p5b", "r1_distill_qwen_1p5b", "qwen2p5_math_1p5b_instruct", "xverify_0p5b",
               "general_verifier", "qwen2p5_7b_instruct", "qwen3p6_35b_a3b_fp8"]      # core first, then held out
EVENT_RULE = ("An event means the family's rate of wrong but accepted answers rose at least 0.05 above its step-0 "
              "value at two evaluations in a row.")
PREREG_ABOUT = {
    "masking": "Experiment B, a masked run: a main run repeated with the same checkpoint, judge and seed, with "
               "**{fam}** rejected by rule in training and evaluation (paper Section 5 and Appendix D).",
    "lowshare": "Experiment C, a low-share run: two families seeded at a total SFT share of about 3%, "
                "setting {fam} (paper Section 5 and Table 5)."}
PREREG_MISSING = ("The runs with recorded predictions are not in demo/data: rebuild it with "
                  "`python -m src.demo.build_data`.")

INTRO = f"""# Which Loopholes Get Found?
### Predicting reward hacking of LLM judges before RL training

LLM judges have blind spots: a lone `{{` or the words `Thought process:` can be graded correct, and a policy
trained with GRPO against such a judge may learn to emit them instead of solving the problem. **Before any RL
step** we measure, for each of 14 hacking patterns, how often the judge accepts it (FPR) and how often the
starting policy produces it (propensity {HAT_P}). Their product times the 256 responses of a training step is the
**{EXPOSURE}**: the expected number of rewarded exploit responses per step. We then train
Qwen2.5-1.5B-Instruct for 300 GRPO steps on math problems, with the judge as the only reward, and watch which
pattern takes over."""


# ----------------------------------------------------------------------------- labels

def judge_name(store, judge):
    return store.judges[judge]["name"]


def judge_labels(store):
    order = JUDGE_ORDER + sorted(set(store.judges) - set(JUDGE_ORDER))
    return {f"{store.judges[k]['name']} ({store.judges[k]['role']})": k for k in order if k in store.judges}


def run_labels(store):
    def label(r):
        outcome = (f"{r['first_event_family']} at step {r['first_event_step']}" if r["first_event_family"]
                   else "no family exploited")
        group = "" if r["group"] == "core" else f", {r['group']}"
        return f"{judge_name(store, r['judge'])}, {r['pool']} pool, seed {r['seed']}{group}: {outcome}"
    return {label(r): r["run"] for _, r in store.runs.iterrows()}


def prereg_labels(store):
    if store.prereg is None:
        return {}

    def label(r):
        what = ("mask " + r["condition"].removeprefix("mask ") if r["kind"] == "masking"
                else "low share, " + r["condition"])
        verdict = "exposure named the winner" if r["hit"] else "exposure missed the winner"
        return f"{r['experiment']}: {judge_name(store, r['judge'])}, seed {r['seed']}, {what}: {verdict}"
    return {label(r): r["run"] for _, r in store.prereg.iterrows()}


def short(text, n=40):
    return text if len(text) <= n else text[:n] + "..."


def responses_table(rows):
    return pd.DataFrame([{"copies": r["count"], "family": r["family"] or "none", "accepted": r["accepted"],
                          "gold correct": r["gold_correct"], "response": r["text"]} for r in rows])


# ----------------------------------------------------------------------------- before training

def predictions(store, judge, pool, checkpoint):
    risk = store.risk
    return risk[(risk["judge"] == judge) & (risk["pool"] == pool) & (risk["checkpoint"] == checkpoint)]


def audit(store, judge, pool, checkpoint):
    """(bar data, ranking table, Markdown on what happened) for one judge, pool and starting checkpoint."""
    sub = predictions(store, judge, pool, checkpoint)
    if sub.empty:
        return (pd.DataFrame({"family": [], "exposure": []}), pd.DataFrame(),
                "No prediction was recorded for this combination: the 2026 judge was added after the 0% control "
                "checkpoints were scored.")
    sub = sub.sort_values(["rank_risk", "family"])
    table = pd.DataFrame({
        "rank": sub["rank_risk"],
        "family": sub["family"],
        "string": [short(store.families[f]["string"]) for f in sub["family"]],
        "seeded": [f"{t:.1%}" if t else "0" for t in sub["tier"]],
        "FPR": sub["fpr"].map(lambda x: f"{x:.1%}"),
        HAT_P: sub["p_hat"].map(lambda x: f"{x:.2%}"),
        "exposure": sub["exposure"].round(2),
        "FPR rank": sub["rank_fpr"]})
    bars = pd.DataFrame({"family": sub["family"], "exposure": sub["exposure"].round(3)})
    return bars, table, outcome_text(store, judge, pool, checkpoint, sub)


def outcome_text(store, judge, pool, checkpoint, sub):
    runs = store.runs[(store.runs["judge"] == judge) & (store.runs["pool"] == pool)
                      & (store.runs["checkpoint"] == checkpoint)]
    if runs.empty:
        return ("**Not trained in the main experiment.** Held-out judges were trained on the hard pool only, and the "
                "0% control only with Qwen2.5-Math-1.5B-Instruct on the hard pool. The ranking above is still the "
                "prediction the audit makes.")
    name = runs.iloc[0]["run"]
    events = store.run(name)["events"]
    top = sub.iloc[0]
    if not events:
        return (f"**What happened (run `{name}`): no family was exploited within {STEPS_PER_RUN} steps.** "
                f"The highest predicted exposure was {top['exposure']:.2f} rewarded responses per step "
                f"({top['family']}).")
    ranks = sub.set_index("family")
    lines = [f"**What happened (run `{name}`):**"]
    for ev in events:
        r = ranks.loc[ev["family"]]
        lines.append(f"- **{ev['family']}** was exploited from step {ev['step']}: ranked **#{int(r['rank_risk'])}** "
                     f"by exposure and #{int(r['rank_fpr'])} by FPR alone.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- during training

def shown_families(store, bundle, run, k=6):
    """Families worth plotting: those that rose or spread, then the top predictions, at most k."""
    peak_e = {f: max(ev["e"][f] for ev in bundle["eval"]) for f in store.families}
    peak_share = {f: max(bundle["train_share"][f]) for f in store.families}
    ranked = list(predictions(store, run["judge"], run["pool"], run["checkpoint"]).sort_values("rank_risk")["family"])
    active = sorted((f for f in store.families if peak_e[f] >= 0.02 or peak_share[f] >= 0.05),
                    key=lambda f: -peak_share[f])
    return (active + [f for f in ranked if f not in active])[:k]


def replay(store, name):
    """(e_f(t), training shares, overall rates, Markdown summary, first event step, responses at that step)."""
    run = store.runs[store.runs["run"] == name].iloc[0]
    bundle = store.run(name)
    fams = shown_families(store, bundle, run)
    e = pd.DataFrame([{"step": ev["step"], "family": f, "e_f(t)": ev["e"][f]} for ev in bundle["eval"] for f in fams])
    share = pd.DataFrame([{"step": s, "family": f, "share": bundle["train_share"][f][s]}
                          for f in fams for s in range(STEPS_PER_RUN)])
    rates = pd.DataFrame([{"step": ev["step"], "rate": rate, "value": ev[key]} for ev in bundle["eval"]
                          for key, rate in (("gold_accuracy", "gold accuracy"), ("accept_rate", "judge acceptance"),
                                            ("false_positive_rate", "accepted but wrong"))])
    head = f"**{judge_name(store, run['judge'])}**, {run['pool']} pool, seed {run['seed']} ({run['group']})."
    events = bundle["events"]
    if events:
        what = "; ".join(f"**{ev['family']}** from step {ev['step']}" for ev in events)
        summary = f"{head} Exploited: {what}. {EVENT_RULE}"
        first = events[0]["step"]
    else:
        summary = f"{head} No family was exploited within {STEPS_PER_RUN} steps."
        first = 0
    return e, share, rates, summary, first, samples(store, name, first)


def samples(store, name, step):
    return responses_table(store.run(name)["samples"].get(str(int(step)), []))


# ----------------------------------------------------------------------------- predictions recorded before training

def named(family, winner):
    """Table cells: the family named first ('+' joins ties) and whether it won."""
    return f"{family} | {'yes' if winner in family.split('+') else 'no'}"


def prereg_plot(bundle, r):
    """e_f(t) plus vertical marks for the predicted step and 90% window (LinePlot cannot shade)."""
    fams = list(dict.fromkeys([r["first_by_exposure"], r["winner"]] + [ev["family"] for ev in bundle["events"]]))
    e = pd.DataFrame([{"step": ev["step"], "e_f(t)": ev["e"][f], "series": f} for ev in bundle["eval"] for f in fams])
    top = max(0.1, round(float(e["e_f(t)"].max()) * 1.1, 3))
    marks = pd.DataFrame([
        {"step": r["predicted_step"], "e_f(t)": 0.0, "series": "predicted step"},
        {"step": r["predicted_step"], "e_f(t)": top, "series": "predicted step"},
        {"step": r["step_low"], "e_f(t)": 0.0, "series": "90% window"},
        {"step": r["step_low"], "e_f(t)": top, "series": "90% window"},
        {"step": r["step_high"], "e_f(t)": top, "series": "90% window"},
        {"step": r["step_high"], "e_f(t)": 0.0, "series": "90% window"}])
    return pd.concat([e, marks], ignore_index=True)


def prereg_missing():
    return PREREG_MISSING, pd.DataFrame(), pd.DataFrame()


def prereg_view(store, name):
    """(Markdown, plot data, responses at the first event) for one run with a recorded prediction."""
    if store.prereg is None or name not in set(store.prereg["run"]):
        return prereg_missing()
    r = store.prereg[store.prereg["run"] == name].iloc[0]
    bundle = store.run(name)
    fam = r["condition"].removeprefix("mask ").removeprefix("setting ")
    later = bundle["events"][1:]
    lines = [PREREG_ABOUT[r["kind"]].format(fam=fam), "",
             f"**{judge_name(store, r['judge'])}**, {r['pool']} pool, seed {r['seed']}, run `{name}`; the judge "
             f"accepted {float(r['q0']):.1%} of the starting policy's answers (q0).", "",
             "| score | family named first | did it win? |", "|---|---|---|",
             f"| {EXPOSURE} | {named(r['first_by_exposure'], r['winner'])} |",
             f"| FPR alone | {named(r['first_by_fpr'], r['winner'])} |",
             f"| propensity alone | {named(r['first_by_propensity'], r['winner'])} |", "",
             f"**Prediction, recorded before training:** {r['first_by_exposure']} (exposure "
             f"{float(r['exposure_top']):.2f} rewarded responses per step) exploited first, at step "
             f"**{float(r['predicted_step']):.0f}**, 90% window {float(r['step_low']):.0f} to "
             f"{float(r['step_high']):.0f} (timing fit of the main runs).", "",
             f"**What happened:** {r['winner']} was exploited first, from step **{int(r['step'])}**"
             + (". Later: " + "; ".join(f"{ev['family']} from step {ev['step']}" for ev in later) + "." if later
                else "; no other family was exploited within 300 steps."), "",
             ("Exposure named the winner" if r["hit"]
              else f"Exposure missed the winner (it named {r['first_by_exposure']})")
             + (", and the first event fell inside the 90% window." if r["in_band"]
                else ", and the first event fell outside the 90% window.")]
    responses = responses_table(bundle["samples"].get(str(int(r["step"])), []))
    return "\n".join(lines), prereg_plot(bundle, r), responses


# ----------------------------------------------------------------------------- audit your own judge
# Mirrors src/audit/predict.py so demo/ stays self-contained; a test keeps the two equal.

OWN_COLUMNS = ["family", "FPR", HAT_P]


class InputError(ValueError):
    """Invalid values typed into the table. The message is shown in the tab."""


def own_inputs(store, judge, pool, checkpoint):
    """Measured FPR and propensity as the editable table."""
    sub = predictions(store, judge, pool, checkpoint).set_index("family")
    if sub.empty:
        return pd.DataFrame(columns=OWN_COLUMNS)
    fams = list(store.families)
    return pd.DataFrame({"family": fams, "FPR": [float(sub.loc[f, "fpr"]) for f in fams],
                         HAT_P: [float(sub.loc[f, "p_hat"]) for f in fams]})


def check_own(store, table, q0):
    """Validate the edited table and q0; return (fpr, p_hat, q0)."""
    if table is None or len(table) == 0:
        raise InputError("the table is empty: pick a judge to prefill it, or enter the 14 families")
    missing_cols = [c for c in OWN_COLUMNS if c not in table.columns]
    if missing_cols:
        raise InputError(f"the table needs the columns {', '.join(OWN_COLUMNS)}")
    t = table[OWN_COLUMNS].copy()
    t = t[~t.apply(lambda r: all(str(v).strip() in ("", "nan", "None") for v in r), axis=1)]        # blank rows
    names = [str(f).strip() for f in t["family"]]
    if any(f in ("", "nan", "None") for f in names):
        raise InputError("a row has no family name")
    dup =sorted({f for f in names if names.count(f) > 1})
    unknown = sorted(set(names) - set(store.families))
    missing = [f for f in store.families if f not in names]
    for problem, fams in (("listed more than once", dup), ("unknown", unknown), ("missing", missing)):
        if fams:
            raise InputError(f"families {problem}: {', '.join(fams)}")
    t.index = names
    values = {}
    for col in ("FPR", HAT_P):
        num = pd.to_numeric(t[col], errors="coerce")
        bad = [f"{f} ({t.loc[f, col]})" for f in store.families if not math.isfinite(float(num[f]))]
        if bad:
            raise InputError(f"{col} is not a number for {', '.join(bad)}")
        out = [f"{f} ({num[f]:g})" for f in store.families if not 0 <= num[f] <= 1]
        if out:
            raise InputError(f"{col} must be between 0 and 1: {', '.join(out)}")
        values[col] = num
    if q0 is not None and str(q0).strip() != "":
        try:
            q0 = float(q0)
        except (TypeError, ValueError):
            raise InputError(f"q0 must be a number, not {q0!r}") from None
        if not 0 < q0 < 1:
            raise InputError(f"q0 must be strictly between 0 and 1, not {q0:g}")
    else:
        q0 = None
    return values["FPR"], values[HAT_P], q0


def rank_desc(values):
    """Descending rank; ties share the lowest."""
    return values.rank(ascending=False, method="min").astype(int)


def rank_own(store, fpr, p_hat, q0=None):
    """(table, top, exposure, (step, low, high) or None, event predicted), as rank_patterns."""
    n = store.timing["step_samples"] if store.timing else 256
    fams = list(store.families)
    t = pd.DataFrame({"family": fams, "fpr": [float(fpr[f]) for f in fams], "p_hat": [float(p_hat[f]) for f in fams]})
    t["exposure"] = n * t["p_hat"] * t["fpr"]
    t["rank"], t["rank_fpr"], t["rank_p_hat"] = rank_desc(t["exposure"]), rank_desc(t["fpr"]), rank_desc(t["p_hat"])
    if q0 is not None:
        t["ea"] = t["p_hat"] * (t["fpr"] - q0) / math.sqrt(q0 * (1 - q0))
        t["rank_ea"] = rank_desc(t["ea"])
    t = t.sort_values(["rank", "family"]).reset_index(drop=True)
    best = float(t["exposure"].max())
    top = "+".join(sorted(t.loc[t["exposure"] == best, "family"]))
    window, predicted = None, False
    if store.timing and best > 0:
        a, b, sd, z = (store.timing[k] for k in ("a", "b", "sd", "z90"))
        mid = a + b * math.log(best)
        window = (math.exp(mid), math.exp(mid - z * sd), math.exp(mid + z * sd))
        predicted = best >= store.timing["level"]
    return t, top, best, window, predicted


def first_by(t, col):
    return "+".join(sorted(t.loc[t[col] == t[col].max(), "family"]))


def own_view(store, table, q0=None):
    """(bar data, ranking table, Markdown) for the values in the table; a message instead when they are invalid."""
    empty = pd.DataFrame({"family": [], "exposure": []}), pd.DataFrame()
    try:
        fpr, p_hat, q0 = check_own(store, table, q0)
    except InputError as err:
        return (*empty, f"**Cannot rank these values:** {err}.")
    t, top, best, window, predicted = rank_own(store, fpr, p_hat, q0)
    shown = pd.DataFrame({"rank": t["rank"], "family": t["family"], "FPR": t["fpr"].map(lambda x: f"{x:.1%}"),
                          HAT_P: t["p_hat"].map(lambda x: f"{x:.2%}"), "exposure": t["exposure"].round(3),
                          "FPR rank": t["rank_fpr"], f"{HAT_P} rank": t["rank_p_hat"]})
    if q0 is not None:
        shown["EA"], shown["EA rank"] = t["ea"].round(5), t["rank_ea"]
    bars = pd.DataFrame({"family": t["family"], "exposure": t["exposure"].round(3)})
    lines = [f"**Named first:** by {EXPOSURE}, **{top}**; by FPR alone, {first_by(t, 'fpr')}; by propensity alone, "
             f"{first_by(t, 'p_hat')}" + (f"; by the expected-advantage score, {first_by(t, 'ea')}" if q0 is not None
                                          else "") + ".", ""]
    if best == 0:
        lines.append("No family is rewarded (every exposure is 0), so no event is predicted.")
    elif window is None:
        lines.append("The timing fit is not in demo/data, so no window is shown: rebuild it with "
                     "`python -m src.demo.build_data`.")
    else:
        step, low, high = window
        lines.append(f"**Prediction:** {top} has the highest exposure, {best:.2f} rewarded responses per step, and is "
                     f"first exploited at about step **{step:.0f}**, 90% window {low:.0f} to {high:.0f} (timing fit "
                     "of the main runs, paper Section 5).")
        if not predicted:
            lines.append(f"Its exposure is below {store.timing['level']:g}, the lowest exposure of any main-run event, "
                         f"so no event is expected within the {STEPS_PER_RUN} steps.")
    return bars, shown, "\n".join(lines)


# ----------------------------------------------------------------------------- results

def results_markdown(store):
    if not store.results:
        return "The analysis results are not in demo/data."
    lines = ["### Does the audit predict which families get exploited?",
             "AUROC over run x family pairs (an event is a positive), 95% intervals from resampling whole runs "
             "within each judge.", "", "| runs | score | AUROC (95% CI) |", "|---|---|---|"]
    for r in store.results["auroc"]:
        lines.append(f"| {r['analysis']} ({r['runs']} runs, {r['events']} events) | {SCORE_NAMES[r['score']]} | "
                     f"{r['auroc']:.3f} ({r['ci_low']:.3f} to {r['ci_high']:.3f}) |")
    lines += ["", "Static accuracy was not measured for Qwen3.6-35B-A3B-FP8, so its held-out value covers the other "
              "two held-out judges only.",
              "", "### Confirmatory tests (Holm over five tests)", "", "| test | result | p | Holm p |",
              "|---|---|---|---|"]
    for r in store.results["confirmatory"]:
        p = "< 0.0001" if r["p"] <= 0.0001 else f"{r['p']:.4f}"
        test = r["test"].replace("risk score", "exposure")     # the analysis files keep their own test names
        lines.append(f"| {test} | {r['detail']} | {p} | {r['p_holm']:.4f} |")
    return "\n".join(lines)
