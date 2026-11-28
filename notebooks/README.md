# Analysis notebooks

- [01_audit_a_judge.ipynb](01_audit_a_judge.ipynb): rank the 14 patterns for a chosen judge and starting checkpoint with `src.audit.predict.rank_patterns`, then compare the ranking with the first event of the matching run.
- [02_results_walkthrough.ipynb](02_results_walkthrough.ipynb): read the AUROC table, main-run winners, H2/H3, masking and low-share outcomes, pooled run/checkpoint tests, and one evaluation trajectory.

Both notebooks read only committed files, require no GPU or data download, and do not overwrite anything under `results/`. Outputs are included so they can also be read directly on GitHub.

Use the project's environment (main README, "Setup") with the notebook tools added, and register it as a Jupyter kernel under its own name. A `python3` kernel registered earlier for another Python installation would otherwise take precedence and lack the project's packages.

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m ipykernel install --user --name which-loopholes --display-name "Python (which-loopholes)"
```

Run from the repository root:

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=which-loopholes notebooks/01_audit_a_judge.ipynb
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=which-loopholes notebooks/02_results_walkthrough.ipynb
```

In Jupyter, choose the kernel "Python (which-loopholes)". Each notebook runs in a few seconds on a laptop CPU, and execution updates only the saved outputs. The first notebook uses rotation 1 of R1-Distill-Verifier-1.5B on the hard pool; change its selection cell to explore another main run with a recorded first event.
