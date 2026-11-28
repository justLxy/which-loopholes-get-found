#!/usr/bin/env bash
# Rebuild the CPU analyses and figures, then check them against the committed outputs (scripts/check_outputs.py:
# CSVs equal to within a relative tolerance of 1e-9, every figure PDF rewritten).
set -Eeuo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif [[ -x .venv/bin/python ]]; then
    python_bin=.venv/bin/python
else
    python_bin=python
fi

step=preflight
started=$SECONDS
since=$(date +%s)
trap 'code=$?; printf "FAIL: %s (exit %d; total %ds)\n" "$step" "$code" "$((SECONDS - started))" >&2; exit "$code"' ERR

git rev-parse --show-toplevel >/dev/null
"$python_bin" -c 'import matplotlib, numpy, pandas, yaml'
# Do not attribute pre-existing staged or unstaged output edits to this run.
git diff --exit-code HEAD -- 'results/analysis/*.csv' 'figures/*.pdf'

run_step() {
    step="$1"
    shift
    local before=$SECONDS
    printf '\n== %s ==\n' "$step"
    "$@"
    printf 'OK: %s (%ds)\n' "$step" "$((SECONDS - before))"
}

run_step primary "$python_bin" -m src.analysis.primary
run_step confirmatory "$python_bin" -m src.analysis.confirmatory
run_step secondary "$python_bin" -m src.analysis.secondary
run_step sensitivity "$python_bin" -m src.analysis.sensitivity
# Descriptive analysis needs released raw logs; its committed CSVs stay in place.
run_step run_level "$python_bin" -m src.analysis.run_level
run_step masking_outcomes "$python_bin" -m src.analysis.masking_outcomes
run_step lowshare_outcomes "$python_bin" -m src.analysis.lowshare_outcomes
run_step pooled_runs "$python_bin" -m src.analysis.pooled_runs
run_step figures "$python_bin" -m figures.make_all
run_step output_check "$python_bin" scripts/check_outputs.py --since "$since"
printf '\nPASS: all CPU analyses and figures reproduced (%ds).\n' "$((SECONDS - started))"
