# Analysis notebooks

- [01_audit_a_judge.ipynb](01_audit_a_judge.ipynb): rank the 14 patterns for a chosen judge and starting checkpoint, then compare the pre-training exposure ranking with the recorded first event. Uses `src.audit.predict.rank_patterns` when D2 is available; otherwise shows the formula explicitly.
- [02_results_walkthrough.ipynb](02_results_walkthrough.ipynb): read the AUROC table, main-run winners, H2/H3, masking and low-share outcomes, pooled run/checkpoint tests, and one evaluation trajectory.

Both notebooks read only committed files, require no GPU or data download, and do not overwrite anything under `results/`. Outputs are included so they can also be read directly on GitHub.

Use the project's CPU Python environment and install the notebook tools:

```bash
python -m pip install pandas matplotlib nbconvert ipykernel
python -m ipykernel install --sys-prefix --name python3
```

When D2 is present, notebook 01 also uses its library dependencies from the project's CPU environment. Choose that same Python environment as the notebook kernel.

Run from the repository root:

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 notebooks/01_audit_a_judge.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 notebooks/02_results_walkthrough.ipynb
```

Each notebook is expected to finish in under two minutes on a laptop CPU. Execution updates only the notebooks' saved outputs. The first notebook uses rotation 1 of R1-Distill-Verifier-1.5B on the hard pool; adjust its selection cell to explore another main run with a recorded first event.

Validated on Python 3.12.16 in WSL Ubuntu: notebook 01 took 4.01 s and notebook 02 took 3.01 s. A separate notebook 01 check using the D2 library took 8.01 s. All executions saved their plots and left `results/` and `configs/` unchanged.
