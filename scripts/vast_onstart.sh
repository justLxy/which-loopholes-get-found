#!/usr/bin/env bash
# Resume the main experiment after the Vast container restarts (findings section 2).
# Vast runs /root/onstart.sh at every container start; install this script there once:
#   printf '\nbash /workspace/which-loopholes-get-found/scripts/vast_onstart.sh\n' >> /root/onstart.sh
# (the leading newline matters: Vast's file can end without one, and the two commands would merge).
# It does nothing unless runs/main/AUTO_RESUME (main experiment) or runs/masking/AUTO_RESUME (experiment B)
# exists (create it to opt in, remove it to opt out). Then it starts detached tmux sessions: "queue"
# (python -m src.train.queue) or "masking" (python -m src.train.masking), each of which skips what is done
# and repeats the interrupted run, and "autostop" (scripts/stop_when_done.sh), unless a session of that name
# is already running. Attach with `tmux attach -t queue` or `-t masking`. Output also goes to runs/*/<name>.out.

REPO=/workspace/which-loopholes-get-found
cd "$REPO" || exit 0
[ -f runs/main/AUTO_RESUME ] || [ -f runs/masking/AUTO_RESUME ] || exit 0
ENV="source /workspace/loopholes-env.sh"
start() {
  local name=$1 dir=$2 cmd=$3
  if tmux has-session -t "$name" 2>/dev/null; then echo "tmux session $name already running"; return; fi
  tmux new-session -d -s "$name" "cd $REPO && $ENV && $cmd 2>&1 | tee -a $dir/$name.out; exec bash"
  echo "[$(date '+%F %T')] started tmux session $name"
}
if [ -f runs/main/AUTO_RESUME ]; then start queue runs/main "python -m src.train.queue"; fi
if [ -f runs/masking/AUTO_RESUME ]; then start masking runs/masking "python -m src.train.masking"; fi
sleep 60   # let the queue start before the watcher looks for it
if [ -f runs/masking/AUTO_RESUME ]; then LOGDIR=runs/masking; else LOGDIR=runs/main; fi
start autostop "$LOGDIR" "bash scripts/stop_when_done.sh"
