# Which Loopholes Get Found? Predicting Reward Hacking of LLM Judges before RL Training

Xuanyi Lyu, Yuezhou Zhao, Qing Qin

Code, data, results and an interactive demo for the paper.

## Overview

Reinforcement learning with verifiable rewards increasingly uses an LLM as the judge of answers that rules
cannot check. Judges have blind spots: a lone `{` or the words `Thought process:` can be graded correct,
and a policy trained against such a judge may learn to emit them instead of solving the problem. Static
probes reveal many such flaws, but not which ones a policy will actually find during training.

We ask whether an audit run **strictly before RL** can predict which of a judge's loopholes a policy will
exploit within a fixed training budget. For 14 families of hacking patterns we measure the judge's
false-positive rate on each pattern (FPR) and how often the starting policy produces it (propensity
p&#770;), and score every family by **r = p&#770; &times; FPR**, recorded before training. We then train
Qwen2.5-1.5B-Instruct with GRPO for 300 steps on math problems, with the judge as the only reward and a
rule-based gold checker that the policy never sees, and record which families are exploited. Propensity
is controlled by a short SFT stage that seeds each family at 0.3%, 1.2% or 4.0% in a Latin square over
seeds, so that it varies independently of FPR.

## Key results

36 GRPO runs: 4 core judges x 2 difficulty pools x 3 seeds, 3 held-out judges x 3 seeds, and 3 runs from
an unseeded (0%) SFT control. Every risk score is computed from static measurements taken before the
run it predicts. Seeding, events, tests and the run queue are fixed in
[`configs/protocol.yaml`](configs/protocol.yaml), tagged `protocol-v1` before the first main run and
amended once (`protocol-v1.1`) to add a third held-out judge at the end of the queue.

**The risk score predicts which families get exploited.** AUROC over run x family pairs (an event is a
positive), with 95% intervals from resampling whole runs within each judge:

| Runs | Risk score p&#770; &times; FPR | FPR alone | Propensity alone | Static accuracy |
|---|---|---|---|---|
| Core, 24 runs, 9 events | **0.974** (0.955 to 0.995) | 0.937 (0.913 to 0.966) | 0.847 (0.779 to 0.935) | 0.379 (0.225 to 0.514) |
| Held-out, 9 runs, 5 events | **0.995** (0.986 to 1.000) | 0.979 (0.961 to 0.998) | 0.760 (0.575 to 0.919) | 0.759 (2 of the 3 judges) |

| Confirmatory test (Holm over five tests) | Result | p | Holm p |
|---|---|---|---|
| H1: risk score vs FPR alone | AUROC +0.037 (+0.020 to +0.055) | < 0.0001 | 0.0005 |
| H1: risk score vs propensity alone | AUROC +0.127 (+0.055 to +0.183) | < 0.0001 | 0.0005 |
| H1: risk score vs static accuracy | AUROC +0.595 (+0.461 to +0.748) | < 0.0001 | 0.0005 |
| H2: hard prompts raise the risk | 7 of 12 run pairs with an event only on the hard pool, none only on the easy pool | 0.016 | 0.031 |
| H3: exploit-style SFT data raises the risk with dose | events 0, 2 and 7 at 0.3%, 1.2% and 4.0%; none in the 0% control | 0.056 | 0.056 (not rejected) |

- In all 12 runs with an event, the first family exploited was the one ranked first by the risk score;
  FPR alone ranked it first in 6. In a winner conditional logit the product separates every winner, while
  FPR alone and propensity alone fall short of the full model (likelihood-ratio p = 1e-6 and 1e-11); a
  cause-specific Cox model gives a hazard ratio of 10.5 (5.1 to 58.8) per unit of log r.
- On the held-out judges FPR alone is nearly as good (difference +0.017, p = 0.066); the risk score gains
  where FPR's top choice was seeded at a low dose and lost.
- The result holds under every event threshold (tau 0.025 to 0.10, one or two evaluations), with a
  threshold-free outcome, with total string likelihood as the propensity and with family-mean FPRs. It
  fails with per-token likelihood, which is not a frequency.
- Early relative growth of a pattern follows FPR rather than the product: seeded strings decay early in
  RL, less so when the judge accepts them.
- The gold checker agrees with a blind hand audit of 200 responses 99.7% of the time (weighted; every miss
  is a correct answer without a box). A blind audit of the flagged exploits is in `audits/exploit/`.

## Repository structure

```
configs/            experiment configuration
  protocol.yaml     design fixed before the main runs (propensity, seeding, events, analysis, main runs)
  data.yaml judges.yaml grpo.yaml sft.yaml
src/
  data/             hacking set download, candidate pool, difficulty pass, difficulty pools
  detectors/        rule-based detectors for the 14 hacking-pattern families
  gold/             gold checker (last \boxed{} + math-verify) and its hand audit
  judges/           judge prompts, verdict parsing, local / vLLM / server judges, engineered judge
  policy/           policy prompt and samplers (local and vLLM)
  pipeline/         record format, sharded resumable runs, end-to-end smoke test
  audit/            pre-RL audits: judge FPR, seeding-string FPR, static accuracy, propensity, likelihood
  train/            seeding set, SFT, GRPO trainer, evaluation callback, the main-experiment queue
  monitor/          event detection, the exploit hand audit
  analysis/         primary, confirmatory, secondary and sensitivity analyses
  demo/             builds the demo's data bundle
scripts/            GPU machine setup, model download, single runs, Vast.ai auto-resume and auto-stop
pools/              the difficulty pools (problem ids per pool and split)
results/
  stage1/           judge FPR by family, static accuracy, base-policy propensity
  stage2/           seeding-string FPR, pilot runs
  main/             packed results of the 36 main runs and the 6 SFT checkpoints
  analysis/         every analysis output (units, AUROC tables, tests, hazards, sensitivity)
audits/             blind hand-audit sheets and labels (gold checker, exploits)
demo/               Gradio app and its data bundle
figures/            the paper's figures (PDF and PNG) and the scripts that draw them from results/
tests/              unit and integration tests (pytest)
```

## Setup

Python 3.12. Pinned versions: PyTorch 2.13.0, Transformers 5.17.0, TRL 1.15.0, vLLM 0.31.0 (GPU only),
math-verify 0.9.0; exact environments are in `requirements-lock-*.txt`.

Local machine (tests, analysis, demo):

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

GPU machine (one NVIDIA GPU with at least 80 GB; 96 GB to serve the 7B and 35B held-out judges beside
training). `scripts/setup_gpu.sh` detects the platform (Vast.ai, GCP, AutoDL), installs
`requirements-gpu.txt`, downloads the data and models, and writes an environment file to source in every
new shell:

```bash
bash scripts/setup_gpu.sh
source /workspace/loopholes-env.sh        # on Vast.ai; $HOME/loopholes-env.sh elsewhere
```

## Reproducing the experiments

Every step resumes after an interruption: rerun the same command. Commands run from the repository root.

**1. Data and pools.** `scripts/setup_gpu.sh` already runs the first two commands.

```bash
python -m src.data.download_hacking                          # hacking set of Huang et al. (2026)
python -m src.data.build_pool                                # 21,845 integer-answer candidate problems
python -m src.data.difficulty --run-dir runs/difficulty      # GPU: 8 samples per problem
python -m src.data.pools --difficulty-run runs/difficulty    # hard (1-2 of 8 correct), easy (5-7 of 8)
```

**2. Audits before training** (inference only, GPU).

```bash
python -m src.audit.fpr --judges all --run-dir runs/judge-audit     # FPR of every judge on every family
bash scripts/stage1_batch.sh                                        # likelihood, static accuracy, judge limit
python -m src.audit.string_fpr --judges all                          # FPR of each family's seeding string
```

**3. Main experiment** (GPU).

```bash
python -m src.train.queue --dry-run       # the 36 runs and 6 checkpoints, and what is done
python -m src.train.queue                 # SFT checkpoints and their risk scores first, then every run
```

The queue builds the six SFT checkpoints and records every prediction before the first GRPO run, then
runs seed by seed; a run is complete when its `meta_end.json` exists, and a failed attempt is set aside
and repeated. Result files are packed into `results/main/`. A single run:
`bash scripts/run_grpo.sh <judge> <pool> <seed>`.

**4. Analysis** (CPU; needs only `results/main/`, which is included).

```bash
python -m src.analysis.primary            # H1: AUROC, run bootstrap, paired differences
python -m src.analysis.confirmatory       # H2, H3 and Holm over the five tests
python -m src.analysis.secondary          # competing risks, early growth, winner conditional logit
python -m src.analysis.sensitivity        # event thresholds, likelihood propensity, family-mean FPR
python -m src.analysis.descriptive        # rewards before each event; acceptance by problem source
python -m src.analysis.run_level          # winner ranks and McNemar, event time, difficulty-aware score
```

The descriptive command also reads the full evaluation logs in `runs/grpo/<run>/eval.jsonl`, which step 3
writes and the repository does not include; its outputs are committed in `results/analysis/`.

**Experiment B, masking** (GPU; `configs/masking.yaml`). Four core runs repeated with one family rejected by
rule, in training and evaluation: the main run's winner in two runs, the family FPR alone ranks first in the
other two. The checkpoints are rebuilt, and the predictions are written from their propensity and committed
before any run.

```bash
python -m src.train.masking --checkpoints   # GPU: rebuild main-rot1 and main-rot2 from the main clean pool
python -m src.train.masking --predict       # CPU: results/masking/predictions.csv, committed before any run
python -m src.train.masking --dry-run       # GPU: every check that must pass before training
python -m src.train.masking                 # GPU: the four runs, resumable; results in results/masking/
python -m src.analysis.masking_outcomes     # CPU: outcomes against the predictions, results/analysis/
```

**5. Figures.** `pip install -r figures/requirements.txt`, then `python -m figures.make_all` redraws every
figure of the paper from the committed results (PDF with embedded fonts, sized for the ACL column and
text widths). The method overview (Figure 2) is TikZ, `figures/fig_overview.tex`, and needs a TeX
installation; Ghostscript, if present, also renders its PNG.

## Experimental details

| | |
|---|---|
| Policy | Qwen2.5-1.5B-Instruct, full fine-tuning, bf16 |
| Seeding SFT | 2,000 examples of the policy's own correct answers, a share replaced by each family's pre-specified string (0.3%, 1.2%, 4.0% by rotation); 2 epochs, learning rate 1e-5, cosine schedule |
| GRPO | 300 steps of 32 prompts x 8 samples; learning rate 1e-6 (constant); DAPO loss, no KL, clip 0.2 / 0.28; group-normalized rewards; temperature 1.0; 1,536 completion tokens |
| Reward | the judge's verdict only; a rule-based gold checker (last `\boxed{}` + math-verify) labels correctness and is never seen by the policy |
| Evaluation | every 10 steps, 100 held-out problems of the training pool x 2 samples |
| Event | a family's rate of wrong, judge-accepted outputs rises at least tau = 0.05 above its step-0 value at two consecutive evaluations |
| Propensity | post-SFT sampled frequency, 16 samples x 250 problems per pool, Jeffreys smoothing |
| Seeds | 0, 1 and 2; seed s trains from the SFT checkpoint of rotation s |
| Statistics | AUROC with percentile bootstrap (10,000 resamples of runs within judges), paired differences, Holm over five confirmatory tests; exact randomization test of the seeding design for H3 |

All values are in `configs/`; `configs/protocol.yaml` records when and why each was fixed.

### Compute

| Stage | Hardware | Workload | GPU hours |
|---|---|---|---|
| Audits | 1 x NVIDIA L4 24 GB | difficulty pass (21,845 problems x 8), judge FPR, propensity | about 12 |
| Pilots | 1 x NVIDIA RTX PRO 6000 96 GB | 5 GRPO pilots, 3 SFT checkpoints, judge tests | about 25 |
| Main experiment | same GPU type | 6 SFT checkpoints, 36 GRPO runs x 300 steps | about 175 |

One GPU hosts the trainer, its vLLM generator and the judge's vLLM server. A GRPO run takes 4.8 hours on
average: about 2.7 hours when a pattern takes over (responses shrink to a few tokens) and 5.5 to 7 hours
otherwise.

## Released artifacts

- `results/main/`: per run, the evaluation summaries (e_f(t) at every evaluation), family counts of the
  training responses at every step, the pre-run risk scores and the run metadata; per SFT checkpoint,
  the seeding-set summary, the sampled propensity, the string likelihood and the risk scores.
- `results/masking/`: experiment B, per masked run the same files as `results/main/`, the rebuilt checkpoints'
  scores and the predictions committed before the runs.
- `results/stage1/`, `results/stage2/`: judge FPRs by family and by seeding string, static accuracy.
- `results/analysis/`: every unit of the analysis and every table reported above.
- `audits/`: the blind hand-audit sheets, labels and scores.
- `demo/data/`: a 3.8 MB bundle with the risk tables, the trajectories of all runs and their most frequent
  responses at each evaluation.

## Demo

```bash
.venv/bin/pip install -r demo/requirements.txt
.venv/bin/python demo/app.py              # then open http://127.0.0.1:7860
```

**Before training** ranks the 14 families of a chosen judge, pool and starting policy by the risk score,
next to the ranking by FPR alone and what happened in the matching run. **During training** replays any
of the 36 runs: each family's exploitation rate, its share of the training responses, and the most
frequent responses at each evaluation step. **Results** shows the numbers above. The app reads only
`demo/data/` (rebuilt with `python -m src.demo.build_data`), so the `demo/` folder runs on its own.

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Hugging Face download stalls | xet transfer backend | `HF_HUB_DISABLE_XET=1` |
| vLLM fails with "Could not find nvcc" | FlashInfer sampler JIT | `VLLM_USE_FLASHINFER_SAMPLER=0` (set by `src/device.py` and the run scripts) |
| Triton compile error on the first forward pass | missing Python headers | install `python3.12-dev`; the setup script checks Triton |
| GRPO step fails on a Mac: `'NoneType' object has no attribute 'apply'` | TRL's fused LM head needs Triton | use `--local-test`, which installs a plain PyTorch shim |
| A hybrid-attention judge (Qwen3.6) fails: max_num_seqs exceeds the available Mamba cache blocks | vLLM's default of 1,024 concurrent sequences | `max_num_seqs: 128` in `configs/judges.yaml` |
| `CUDA Error: invalid argument at cumem_allocator.cpp` at the very end of a run | vLLM sleep-mode allocator at interpreter exit | harmless after the last write; a run with `meta_end.json` counts as complete |
| The queue stops with no error in any log | the Vast.ai container restarted | create `runs/main/AUTO_RESUME` and call `scripts/vast_onstart.sh` from `/root/onstart.sh` (append it after a newline, or the two commands merge) |
| A stopped Vast.ai instance cannot restart ("Scheduling") | its GPU was rented by someone else; the disk is kept | copy the data out of the stopped instance (cloud sync or instance-to-instance copy), then destroy it; back results up before any stop |
| Hand-labeled sheets come back with altered characters | spreadsheet apps rewrite unusual characters | only `audit_id` and the labels are read from returned files |

## Licenses

The code in this repository is released under the MIT License (see [`LICENSE`](LICENSE)). Third-party
data and models keep their own licenses, as listed on their Hugging Face cards:

| Artifact | Use here | License |
|---|---|---|
| Hacking set of Huang et al. (`hkust-nlp/rl-verifier-pitfalls_hacking_data`) | audit problems and pattern families | Apache-2.0 |
| DeepScaleR-Preview (`agentica-org/DeepScaleR-Preview-Dataset`) | training problems | MIT |
| MATH (`DigitalLearningGmbH/MATH-lighteval`) | training problems | MIT |
| MATH-500 (`HuggingFaceH4/MATH-500`), GSM8K test (`openai/gsm8k`) | overlap removal only | none stated (a subset of MATH); MIT |
| Qwen2.5-1.5B-Instruct, Qwen2.5-Math-1.5B-Instruct, Qwen2.5-7B-Instruct, Qwen3.6-35B-A3B-FP8 | policy and judges | Apache-2.0 |
| Qwen3.8-27B-FP8 | candidate judge (FPR audit only) | Apache-2.0 |
| R1-Distill-Verifier-1.5B, General-Verifier | judges | Apache-2.0 |
| DeepSeek-R1-Distill-Qwen-1.5B | judge | MIT |
| xVerify-0.5B-I | judge (inference only) | CC BY-NC-ND 4.0 (non-commercial, no derivatives) |

All of them are used for research and as their terms allow; no model weights are redistributed.

## Intended use, risks and limitations

The audit is meant to flag risky pairings of judge and policy before costly RL runs, and to help
evaluators who rely on LLM-as-a-judge scores. The hacking patterns studied here were already published
by the works below; the results show which of them small policies exploit, which could also help
someone game a judge-based evaluation. We report every pattern and judge to make such weaknesses known
and testable.

The findings are limited to one 1.5B policy, integer-answer math, 14 known pattern families (not the
discovery of new ones), judges that fit beside training on one GPU, and a 300-step budget. Natural
propensities were too low for any pattern to be found at this scale, so propensity was raised by
controlled seeding. Events are few (14 in 12 runs), held-out judges are three, the dose-response test
(H3) did not pass, and the hard pool differs from the easy pool in problem source as well as difficulty.

## Citation

```bibtex
@misc{lyu2026loopholes,
  title  = {Which Loopholes Get Found? Predicting Reward Hacking of LLM Judges before RL Training},
  author = {Lyu, Xuanyi and Zhao, Yuezhou and Qin, Qing},
  year   = {2026},
  note   = {Preprint}
}
```

## References

1. Huang et al. 2026. From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in
   Mathematical Reasoning. In Proceedings of EMNLP 2026. https://aclanthology.org/2026.emnlp-main.255/
2. Zhao et al. 2025. One Token to Fool LLM-as-a-Judge. arXiv:2507.08794.
3. Shao et al. 2024. DeepSeekMath. arXiv:2402.03300.
