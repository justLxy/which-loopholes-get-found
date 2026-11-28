# Demo: Which Loopholes Get Found?

Interactive view of [the paper](https://arxiv.org/abs/2611.29651): before RL training, the **exposure
(256 × p̂ × FPR)** of 14 hacking patterns predicts which one a policy learns first against an LLM judge.

The app reads only `demo/data/` (committed, 4.3 MB). No GPU or model weights needed.

## Install and run

From the repository root, with Python 3.10 or newer (use your `python3.X` if `python3.12` is missing):

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/pip install -r demo/requirements.txt
```

```bash
.venv/bin/python demo/app.py
```

Open <http://127.0.0.1:7860>. Options:

| Option | Default | Meaning |
|---|---|---|
| `--port` | `7860` | port |
| `--host` | `127.0.0.1` | address; `0.0.0.0` exposes the demo to your network |
| `--data` | `demo/data` | data bundle |

```bash
.venv/bin/python demo/app.py --port 7861
```

Exit code 2: missing or incomplete data. Exit code 1: port unavailable.

## Tabs

### Before training: the audit

Ranks the 14 families by exposure for a judge, difficulty pool and starting policy, and shows what happened in the
matching run. Held-out judges were trained on the hard pool only; some combinations have no run or prediction.

![Before training](screenshots/1-before-training.png)

### During training: replay a run

Replays any of the 36 main runs: exploitation rate e_f(t), training shares, accuracy and acceptance, and the most
frequent responses at each evaluation step.

![During training](screenshots/2-during-training.png)

### Predictions recorded before training

The 4 masked runs (experiment B) and 6 low-share runs (experiment C): predicted family, step and 90% window against
the outcome. Exposure named the winner in 3 of 4 and 6 of 6.

![Predictions recorded before training](screenshots/3-predictions-recorded.png)

### Audit your own judge

Edit the prefilled FPR and p̂ (optionally add q0) and press **Rank**. Invalid input shows a message. Command-line
version, in the main environment (README, "Setup"):

```bash
.venv/bin/python -m src.audit.predict --judge r1_distill_verifier_1p5b --checkpoint main-rot1 --pool hard
```

For your own data, pass `--fpr` and `--propensity` CSVs (columns `family,fpr` and `family,p_hat`, all 14 families).

![Audit your own judge](screenshots/4-audit-your-own-judge.png)

### Results

AUROC of each score and the confirmatory tests.

![Results](screenshots/5-results.png)

## What the numbers mean

| Term | Meaning | Paper |
|---|---|---|
| FPR | how often the judge accepts the family's seeding string as an answer | [§3 Judge audit](https://arxiv.org/html/2611.29651#S3.SS0.SSS0.Px1) |
| p̂ | share of the starting policy's samples in the family | [§3 Propensity](https://arxiv.org/html/2611.29651#S3.SS0.SSS0.Px2) |
| exposure | 256 × p̂ × FPR, expected rewarded exploits per step (`risk` in the CSVs is p̂ × FPR) | [§3 Exposure](https://arxiv.org/html/2611.29651#S3.SS0.SSS0.Px3) |
| seeded | share of the SFT set replaced by the family's string (0.3%, 1.2%, 4.0%) | [App. B](https://arxiv.org/html/2611.29651#A2) |
| e_f(t), event | share of outputs in family f that are wrong and accepted; an event is a rise of 0.05 over step 0 at two evaluations in a row | [§4 Outcome](https://arxiv.org/html/2611.29651#S4.SS0.SSS0.Px1) |
| AUROC, H1–H3 | how well a score separates events; the confirmatory tests (Holm-adjusted) | [§4 Analysis](https://arxiv.org/html/2611.29651#S4.SS0.SSS0.Px5), [§5](https://arxiv.org/html/2611.29651#S5) |
| step, 90% window | log step = 5.64 − 1.03 log exposure, ± 1.645 × 0.38 (`src/train/masking.py`) | [App. D](https://arxiv.org/html/2611.29651#A4.SS0.SSS0.Px8) |
| level 1.92 | lowest exposure of any main-run event | [App. D](https://arxiv.org/html/2611.29651#A4.SS0.SSS0.Px2) |
| q0, EA | judge acceptance of starting outputs; EA = p̂ (FPR − q0) / √(q0 (1 − q0)) | [App. D](https://arxiv.org/html/2611.29651#A4.SS0.SSS0.Px7) |
| masked, low share | one family rejected by rule; two families seeded at 3% (F1, F2) or 3.8% (P2) of the SFT set in total | [§5](https://arxiv.org/html/2611.29651#S5.SS0.SSS0.Px7), [App. D](https://arxiv.org/html/2611.29651#A4.SS0.SSS0.Px9) |

## Data bundle (`demo/data/`)

| File | Contents |
|---|---|
| `judges.json` | per judge: `name, model, role, static_balanced_accuracy` |
| `families.json` | per family: `string, description` |
| `risk.csv` | `checkpoint, seed, seeding, judge, pool, family, tier, fpr, p_hat, risk, exposure, rank_risk, rank_fpr` |
| `runs.csv` | 36 main runs: `run, judge, pool, seed, checkpoint, group, first_event_family, first_event_step, events, has_samples` |
| `prereg.csv` | 10 runs: `run, experiment, kind, judge, pool, seed, condition, q0, first_by_exposure, exposure_top, first_by_fpr, first_by_propensity, predicted_step, step_low, step_high, winner, step, later_events, hit, in_band, has_samples` |
| `results.json` | `auroc`: `analysis, runs, events, score, auroc, ci_low, ci_high`; `confirmatory`: `test, detail, p, p_holm, reject_at_0.05` |
| `timing.json` | `a, b, sd, z90, level, step_samples, source` |
| `runs/<run>.json` | `run, events, eval, train_share, samples` (and `kind` for the 10 runs) |

## Rebuild the bundle

Only needed after changing the builder or `results/`. Use the main environment. The raw logs (103 MB, 706 MB
extracted) provide the sample responses:

```bash
mkdir -p runs
```

```bash
curl -L -o runs/which-loopholes-eval-logs-v1.tar.gz https://github.com/justLxy/which-loopholes-get-found/releases/download/eval-logs-v1/which-loopholes-eval-logs-v1.tar.gz
```

```bash
tar -xzf runs/which-loopholes-eval-logs-v1.tar.gz -C runs
```

```bash
(cd runs/eval-logs && shasum -a 256 -c SHA256SUMS)
```

```bash
for f in runs/eval-logs/*/*/eval.jsonl; do run=$(basename "$(dirname "$f")"); mkdir -p "runs/grpo/$run" && cp "$f" "runs/grpo/$run/"; done
```

```bash
.venv/bin/python -m src.demo.build_data
```

With pandas 3.0.6 and numpy 2.5.4, `demo/data` comes out unchanged. `--prereg-only` rebuilds only experiments B and C
and `timing.json`.

## Tests

The tests run in the main environment (main README, "Setup") with `demo/requirements.txt` installed as well.
`demo/requirements.txt` alone does not include pytest.

```bash
.venv/bin/pip install -r requirements.txt -r demo/requirements.txt
```

```bash
.venv/bin/python -m pytest -q tests/test_demo_app.py tests/test_demo_store.py tests/test_demo_data.py tests/test_predict.py
```

```bash
.venv/bin/python -m pytest -q -m integration
```

## Code and API

`store.py` loads and checks the bundle, `views.py` builds what is shown, `app.py` holds the layout and callbacks.
Endpoints: `/audit`, `/replay`, `/samples`, `/prereg_view`, `/own_prefill`, `/own_rank`. With the app running:

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
print(client.predict("R1-Distill-Verifier-1.5B, hard pool, seed 1: reasoning_opener at step 40",
                     api_name="/replay")[3])
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `the demo data ... does not exist / is empty / is incomplete` | `git checkout -- demo/data`, or rebuild |
| `cannot serve on ... is the port in use?` | use `--port 7861`; find the other process with `lsof -i :7860` |
| `No module named 'gradio'` | `.venv/bin/pip install -r demo/requirements.txt` |
| `unexpected keyword argument` at start | wrong Gradio version: `.venv/bin/pip install gradio==6.29.1` |
| `pip` cannot find `gradio==6.29.1` | use Python 3.10 or newer |
| Empty response tables after a rebuild | the run logs were missing: add them and rebuild |
| Last digits change after a rebuild | `.venv/bin/pip install pandas==3.0.6 numpy==2.5.4` |
