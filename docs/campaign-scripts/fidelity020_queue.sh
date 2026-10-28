#!/bin/sh
# Spec 020 campaign queue (operator decision 2026-10-28: sharper reconstructions; surrogate
# rulings Q5/Q7): on smpl_proc_v3, rescore the five references (R0-v3, R0b-v3, R0c-v3, P2,
# P2-seed1) with scripts/fidelity.py, then F1 plain AE + gradient_weight 0.5 and F2 pose 0.03 +
# gradient_weight 0.5, each probed (configs/smpl_proc_v3/probe.yaml) and rescored. ~14.5 h.
# Runs AFTER the slice 019 queue: same per-machine lock data/smpl_queue.lock (exits 0 while
# 019 holds it; re-run later). 12 GB memory gate after F1, OOM guard with boot id, watchdog
# SIGTERM (exit 143) not counted as OOM (pattern of pose018_queue.sh).
# Copy to data/fidelity020_queue.sh; watchdog: LOG=data/fidelity020_queue.log
# QUEUE=data/fidelity020_queue.sh data/watchdog.sh
# Idempotent: every step skips if done and resumes if unfinished. Re-run after any restart.
# Terminal lines: the done marker (printed once, last line), QUEUE_INCOMPLETE, MEMORY_STOP.
cd /home/user/strike-a-pose
exec 9> data/smpl_queue.lock
flock -n 9 || { echo "[$(date -u +%H:%M:%S)] another queue holds the lock; exiting"; exit 0; }
U="uv run --locked"
MEM_LIMIT_MB=12288
latest() { ls -d runs/$1-2026* 2>/dev/null | sort | tail -1; }
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
memstop() { log "MEMORY_STOP $*"; exit 0; }
selected() {  # $1 train run dir -> its scored checkpoint path
  ck=$($U python -c "import json;print(json.load(open('$1/metrics.json'))['selection']['checkpoint'])" 2>/dev/null)
  [ -n "$ck" ] && echo "$1/$ck"
}
probe_run() {  # $1 train run dir, $2 probe config
  [ -f "$1/metrics.json" ] || return 0; [ -f "$1/probed.txt" ] && return 0
  ck=$(selected "$1")
  out=$($U strike-a-pose-eval probe --config "$2" --checkpoint "$ck" 2>&1 | tee -a data/fidelity020_queue_probe.log | grep -o "runs/probe-[a-z0-9_.]*-[0-9TZ]*" | head -1)
  [ -n "$out" ] && { echo "$out ($ck)" > "$1/probed.txt"; log "probed $1 -> $out"; } || log "PROBE FAILED for $1"
}
rescore() {  # $1 train run dir: fidelity.json (spec 020 FR-005), skipped when present
  [ -n "$1" ] && [ -f "$1/metrics.json" ] || { log "RESCORE SKIPPED (no finished run) $1"; return 0; }
  [ -f "$1/fidelity.json" ] && return 0
  $U python scripts/fidelity.py rescore "$1" --split val test >> data/fidelity020_queue_rescore.log 2>&1 \
    && log "rescored $1" || log "RESCORE FAILED for $1"
}
mem_gate() {  # $1 run dir with run_info.json: stop the queue if its peak RSS > limit
  p=$($U python -c "import json;print(json.load(open('$1/run_info.json')).get('peak_rss_mb') or 0)" 2>/dev/null)
  log "peak_rss_mb $1 ${p:-?}"
  [ -n "$p" ] && $U python -c "import sys;sys.exit(0 if float('$p') > $MEM_LIMIT_MB else 1)" && memstop "$1 peak ${p} MB > $MEM_LIMIT_MB MB"
  return 0
}
train() {  # $1 config, $2 run_name prefix, $3 probe config
  d=$(latest "$2")
  if [ -n "$d" ] && [ -f "$d/aborted.json" ]; then log "ABORTED $d (see aborted.json); skipping"; return 0; fi
  if [ -z "$d" ]; then log "start $2"; $U strike-a-pose-train --config "$1"; rc=$?
  elif [ ! -f "$d/metrics.json" ]; then
    # OOM guard: a second failed attempt on the same boot without metrics -> terminal stop
    bid=$(cat /proc/sys/kernel/random/boot_id)
    if [ -f "$d/resume_tried" ] && [ "$(cat "$d/resume_tried")" = "$bid" ]; then memstop "$d unfinished after a resume (likely OOM)"; fi
    echo "$bid" > "$d/resume_tried"; log "resume $d"; $U strike-a-pose-train --resume "$d"; rc=$?
  else rc=0; fi
  # watchdog SIGTERM (143) is not OOM: drop the resume marker so the next pass does not MEMORY_STOP
  if [ "$rc" = 143 ]; then d=$(latest "$2"); [ -n "$d" ] && rm -f "$d/resume_tried"; log "trainer terminated by watchdog (exit 143); not counted as OOM"; fi
  [ "$rc" = 137 ] && memstop "$2 trainer killed (exit 137)"
  [ "$rc" != 0 ] && log "train exit $rc for $2"
  d=$(latest "$2"); [ -n "$d" ] && probe_run "$d" "$3" && rescore "$d"
}
[ -f data/generated/smpl_proc_v3/manifest.json ] && grep -q '"complete": true' data/generated/smpl_proc_v3/manifest.json && [ -f splits/smpl_proc_v3.json ] || { log "QUEUE_INCOMPLETE: smpl_proc_v3 data or split missing (run the 015 queue first)"; exit 1; }
[ -f scripts/fidelity.py ] || { log "QUEUE_INCOMPLETE: scripts/fidelity.py missing (implement slice 020 B1-B3 first)"; exit 1; }
REFS="ae-smpl_proc_v3 ae-seed1-smpl_proc_v3 ae-seed2-smpl_proc_v3 ae-pose-l0.03-smpl_proc_v3 ae-pose-l0.03-seed1-smpl_proc_v3"
TRAIN="ae-grad-smpl_proc_v3 ae-pose-l0.03-grad-smpl_proc_v3"
pending() {  # every reference rescored; every variant trained (or aborted), probed, rescored
  for n in $REFS; do d=$(latest $n); [ -n "$d" ] && [ -f "$d/fidelity.json" ] || return 0; done
  for n in $TRAIN; do d=$(latest $n); [ -z "$d" ] && return 0; [ -f "$d/aborted.json" ] && continue; [ ! -f "$d/metrics.json" ] && return 0; [ -f "$d/probed.txt" ] || return 0; [ -f "$d/fidelity.json" ] || return 0; done
  return 1
}
pass=0
while [ $pass -lt 5 ]; do
pass=$((pass+1)); log "pass $pass"
for n in $REFS; do rescore "$(latest $n)"; done
c=configs/smpl_proc_v3
train $c/ae_grad.yaml ae-grad-smpl_proc_v3 $c/probe.yaml
d=$(latest ae-grad-smpl_proc_v3); [ -n "$d" ] && [ -f "$d/run_info.json" ] && mem_gate "$d"
train $c/ae_pose_l0.03_grad.yaml ae-pose-l0.03-grad-smpl_proc_v3 $c/probe.yaml
pending || break
done
pending && { log "QUEUE_INCOMPLETE after $pass passes (see FAILED/exit lines)"; exit 1; }
log "QUEUE_DONE"
