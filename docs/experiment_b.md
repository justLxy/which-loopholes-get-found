# Experiment B on a new machine

The masking runs (`configs/masking.yaml`) start from the SFT checkpoints `main-rot1` and `main-rot2`, which were
lost with the main-run machine. They are rebuilt from the clean pool of the main experiment, which the Mac keeps in
`runs/vast/sft/`, so the seeding sets are the same; the propensity is measured again, the predictions are made
from it and committed, and only then do the four runs start. `<HOST>` and `<PORT>` are the SSH address of the
instance as Vast shows it.

## 0. Machine

Vast.ai, one NVIDIA RTX PRO 6000 Blackwell 96 GB (the GPU of the main runs; any card with at least 80 GB works,
since trainer, vLLM generator and judge share it and peaked at 67 GB), NVIDIA driver 580 or later (CUDA 13.0),
image `vastai/base-image:cuda-13.0.3-auto` (Ubuntu 24.04, Python 3.12, nvcc 13.0), SSH launch, at least 12 CPU
cores and 64 GB RAM, reliability 99% or more, 150 to 200 GB disk. The three models take about 10 GB and the Python
environment about 15 GB; the rest is checkpoints and run logs. Keep the instance running until the results are
copied: a stopped Vast instance keeps its disk but may lose its GPU to another renter.

## 1. Set up (GPU machine)

```bash
cd /workspace
git clone https://github.com/justLxy/which-loopholes-get-found.git
SKIP_MODELS=1 bash which-loopholes-get-found/scripts/setup_gpu.sh      # environment, data, tests (about 15 min)
source /workspace/loopholes-env.sh
python scripts/download_models.py --only policy r1_distill_verifier_1p5b qwen2p5_math_1p5b_instruct
mkdir -p runs/sft runs/masking
```

The setup prints `candidate pool identical to the local build` and ends with the test summary.

## 2. Copy the clean pool (Mac, repository root)

```bash
scp -P <PORT> runs/vast/sft/clean_pool.jsonl runs/vast/sft/clean_pool.json root@<HOST>:/workspace/which-loopholes-get-found/runs/sft/
```

## 3. Rebuild main-rot1 and main-rot2 (GPU machine, about 40 minutes)

```bash
tmux new -s masking
source /workspace/loopholes-env.sh
python -m src.train.masking --checkpoints 2>&1 | tee -a runs/masking/checkpoints.out
```

For each checkpoint this builds the seeding set, runs SFT and measures the propensity on both pools (4,000
samples each), then copies `data_summary.json`, `propensity_sampled.csv` and `risk_scores.csv` to
`results/masking/sft-main-rot<s>/`. The log must say `seeding set matches the main checkpoint` for both; it also
reports how far the new propensity is from the main experiment's.

## 4. Predictions (Mac, repository root)

```bash
scp -P <PORT> -r root@<HOST>:/workspace/which-loopholes-get-found/results/masking/sft-main-rot1 \
    root@<HOST>:/workspace/which-loopholes-get-found/results/masking/sft-main-rot2 results/masking/
.venv/bin/python -m src.train.masking --predict
```

Check that every run still has `separates_exposure_from_fpr` True; then commit and push
`results/masking/sft-main-rot1/`, `results/masking/sft-main-rot2/` and `results/masking/predictions.csv`.

## 5. Start the runs (GPU machine, about 12 to 20 hours)

The copies made in step 3 are now in the repository, so remove them before pulling:

```bash
source /workspace/loopholes-env.sh
rm -r results/masking/sft-main-rot1 results/masking/sft-main-rot2
git pull --ff-only
python -m src.train.masking --dry-run                # every check must print [ok]
touch runs/masking/AUTO_RESUME                       # restart the queue after a container restart
printf '\nbash /workspace/which-loopholes-get-found/scripts/vast_onstart.sh\n' >> /root/onstart.sh
python -m src.train.masking 2>&1 | tee -a runs/masking/masking.out
```

Run the last command inside the tmux session (`tmux attach -t masking`; detach with Ctrl+B then D). The queue
refuses to start if the predictions are not committed, differ from the committed scores, or a checkpoint has a
different propensity. It runs the four runs in turn, skips finished ones and repeats an interrupted one; progress is
in `runs/masking/queue.log`. Optionally, in a second tmux window, `bash scripts/stop_when_done.sh` packs the results
and stops the instance when no run is active (`DRY_RUN=1` first checks that it can).

## 6. Results (Mac, repository root)

```bash
scp -P <PORT> -r root@<HOST>:/workspace/which-loopholes-get-found/results/masking results/
```

Each run directory holds `eval_summary.jsonl`, `events_tau0.05.json`, `family_counts.csv`, the predicted risk with
the masked family set to zero, and the run metadata. Destroy the instance only after this copy.

## 7. Analysis (Mac, repository root)

```bash
.venv/bin/python -m src.analysis.masking_outcomes
```

It compares each run's first family event with `results/masking/predictions.csv` and writes
`results/analysis/masking_outcomes.csv` and `masking_summary.csv`.
