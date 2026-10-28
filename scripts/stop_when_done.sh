#!/usr/bin/env bash
# Stop this Vast.ai instance once every GRPO run has finished, after saving the run summaries.
# Run it on the Vast machine in its own tmux window (Ctrl+B then C), from the repository root:
#   DRY_RUN=1 bash scripts/stop_when_done.sh   # check that the instance can stop itself, then exit
#   bash scripts/stop_when_done.sh             # wait for the runs, summarize them, stop the instance
#   STOP_NOW=1 bash scripts/stop_when_done.sh  # do not wait: pack, summarize and stop right away
# It waits until a run is seen (up to 30 minutes), then until no run has been active for 3 checks
# a minute apart (so the gap between two chained runs does not count as the end). Summaries and
# events (tau 0.05) of every run directory go to runs/grpo/summaries-<time>.txt before stopping, and the
# result files are packed into results/main/ and results/masking/ (the --pack-only of both queues).
# Stopping ends the GPU charge; the disk is kept and still billed ($0.056/h for 200 GB on Vast), and
# someone else may rent the GPU meanwhile, so the instance may not restart until it is free again.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
POLL_SECONDS=${POLL_SECONDS:-60}
IDLE_POLLS=${IDLE_POLLS:-3}
WAIT_FOR_START=${WAIT_FOR_START:-1800}
PATTERN="src.train.grpo|scripts/run_grpo.sh|scripts/run_seeded_pilot.sh|src.train.queue|src.train.masking"

# A new tmux window does not have the project environment, and SSH shells may miss the container's
# variables; take them from the environment file and from the container's first process.
if [ -f "${DATA:-/workspace}/loopholes-env.sh" ]; then
  # shellcheck source=/dev/null
  source "${DATA:-/workspace}/loopholes-env.sh"
fi
for name in CONTAINER_ID CONTAINER_API_KEY; do
  if [ -z "${!name:-}" ] && [ -r /proc/1/environ ]; then
    value=$(tr '\0' '\n' < /proc/1/environ | sed -n "s/^$name=//p" | head -1)
    if [ -n "$value" ]; then export "$name=$value"; fi
  fi
done
if [ -z "${CONTAINER_ID:-}" ]; then echo "CONTAINER_ID is not set: is this a Vast instance?"; exit 1; fi
API_KEY=${VAST_API_KEY:-${CONTAINER_API_KEY:-}}
if [ -z "$API_KEY" ]; then echo "no API key: CONTAINER_API_KEY is not set (set VAST_API_KEY instead)"; exit 1; fi

VASTAI=${VASTAI:-/root/vastcli/bin/vastai}
if [ ! -x "$VASTAI" ]; then
  echo "installing the vastai command-line tool in /root/vastcli"
  python3 -m venv /root/vastcli && /root/vastcli/bin/pip install -q vastai || exit 1
fi
if ! "$VASTAI" show instance "$CONTAINER_ID" --api-key "$API_KEY" > /dev/null; then
  echo "the API key cannot read instance $CONTAINER_ID, so it could not stop it either; not waiting"; exit 1
fi
echo "[$(date '+%F %T')] instance $CONTAINER_ID can be managed with this key"
if [ "${DRY_RUN:-0}" = 1 ]; then echo "dry run: not waiting, not stopping"; exit 0; fi

if [ "${STOP_NOW:-0}" = 1 ]; then POLL_SECONDS=0; IDLE_POLLS=0; WAIT_FOR_START=0; fi   # pack, summarize, stop now
waited=0
until [ "${STOP_NOW:-0}" = 1 ] || pgrep -f "$PATTERN" > /dev/null; do
  if [ "$waited" -ge "$WAIT_FOR_START" ]; then echo "no run started within $WAIT_FOR_START s; not stopping"; exit 1; fi
  sleep "$POLL_SECONDS"; waited=$((waited + POLL_SECONDS))
done
echo "[$(date '+%F %T')] runs are active; checking every $POLL_SECONDS s"
idle=0
while [ "$idle" -lt "$IDLE_POLLS" ]; do
  sleep "$POLL_SECONDS"
  if pgrep -f "$PATTERN" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
done

# Main experiment and experiment B: copy every finished run's small result files (with family counts) to
# results/main and results/masking first.
if [ -f src/train/queue.py ]; then python -m src.train.queue --pack-only || true; fi
if [ -f configs/masking.yaml ]; then python -m src.train.masking --pack-only || true; fi
OUT="runs/grpo/summaries-$(date '+%Y%m%d-%H%M').txt"
echo "[$(date '+%F %T')] no run active; writing $OUT"
{
  for d in runs/grpo/*/; do
    [ -f "$d/meta.json" ] || continue
    python -m src.train.summary "$d"
    if [ -f "$d/eval_summary.jsonl" ]; then python -m src.monitor.events "$d" --tau 0.05; fi
    echo
  done
} > "$OUT" 2>&1
sync
echo "[$(date '+%F %T')] stopping instance $CONTAINER_ID"
"$VASTAI" stop instance "$CONTAINER_ID" --api-key "$API_KEY"
