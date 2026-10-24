#!/bin/sh
# Spec 018 campaign queue (operator decision 2026-10-24: pose-supervised AE; surrogate rulings
# Q3/Q4): on smpl_proc_v3, P1 pose_weight 0.003, P2 pose_weight 0.03, P3 pose_weight 0.03 +
# mask weight 4; probe (configs/smpl_proc_v3/probe.yaml) after each. ~5-6 h per run, ~17 h.
# Reuses the 015 / grill queue helpers: one per-machine lock, 12 GB memory gate after the
# first run, OOM guard with boot id, watchdog SIGTERM (exit 143) not counted as OOM.
# Copy to data/pose018_queue.sh; watchdog: LOG=data/pose018_queue.log QUEUE=data/pose018_queue.sh
# data/watchdog.sh (trainer output goes to $LOG, so the watchdog's loading-phase leniency on
# "run dir ..." / "resuming ..." lines applies).
# Idempotent: every step skips if done and resumes if unfinished. Re-run after any restart.
# Terminal lines: QUEUE_DONE, QUEUE_INCOMPLETE, MEMORY_STOP.
cd /home/user/strike-a-pose
# single instance per machine: the ONE campaign lock (analyze A2)
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
  out=$($U strike-a-pose-eval probe --config "$2" --checkpoint "$ck" 2>&1 | tee -a data/pose018_queue_probe.log | grep -o "runs/probe-[a-z0-9_.]*-[0-9TZ]*" | head -1)
  [ -n "$out" ] && { echo "$out ($ck)" > "$1/probed.txt"; log "probed $1 -> $out"; } || log "PROBE FAILED for $1"
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
    # OOM guard (analyze A2): a second failed attempt without peak_rss_mb -> terminal stop
    # marker = boot id: a differing boot id means a container restart killed the trainer (not OOM)
    bid=$(cat /proc/sys/kernel/random/boot_id)
    if [ -f "$d/resume_tried" ] && [ "$(cat "$d/resume_tried")" = "$bid" ]; then memstop "$d unfinished after a resume (likely OOM)"; fi
    echo "$bid" > "$d/resume_tried"; log "resume $d"; $U strike-a-pose-train --resume "$d"; rc=$?
  else rc=0; fi
  # watchdog SIGTERM (143) is not OOM: drop the resume marker so the next pass does not MEMORY_STOP
  if [ "$rc" = 143 ]; then d=$(latest "$2"); [ -n "$d" ] && rm -f "$d/resume_tried"; log "trainer terminated by watchdog (exit 143); not counted as OOM"; fi
  [ "$rc" = 137 ] && memstop "$2 trainer killed (exit 137)"
  [ "$rc" != 0 ] && log "train exit $rc for $2"
  d=$(latest "$2"); [ -n "$d" ] && probe_run "$d" "$3"
}
[ -f data/generated/smpl_proc_v3/manifest.json ] && grep -q '"complete": true' data/generated/smpl_proc_v3/manifest.json && [ -f splits/smpl_proc_v3.json ] || { log "QUEUE_INCOMPLETE: smpl_proc_v3 data or split missing (run the 015 queue first)"; exit 1; }
TRAIN="ae-pose-l0.003-smpl_proc_v3 ae-pose-l0.03-smpl_proc_v3 ae-pose-l0.03-mask-smpl_proc_v3"
pending() {  # every variant trained (or aborted) and probed
  for n in $TRAIN; do d=$(latest $n); [ -z "$d" ] && return 0; [ -f "$d/aborted.json" ] && continue; [ ! -f "$d/metrics.json" ] && return 0; [ -f "$d/probed.txt" ] || return 0; done
  return 1
}
pass=0
while [ $pass -lt 5 ]; do
pass=$((pass+1)); log "pass $pass"
c=configs/smpl_proc_v3
train $c/ae_pose_l0.003.yaml ae-pose-l0.003-smpl_proc_v3 $c/probe.yaml
d=$(latest ae-pose-l0.003-smpl_proc_v3); [ -n "$d" ] && [ -f "$d/run_info.json" ] && mem_gate "$d"
train $c/ae_pose_l0.03.yaml ae-pose-l0.03-smpl_proc_v3 $c/probe.yaml
train $c/ae_pose_l0.03_mask.yaml ae-pose-l0.03-mask-smpl_proc_v3 $c/probe.yaml
pending || break
done
pending && { log "QUEUE_INCOMPLETE after $pass passes (see FAILED/exit lines)"; exit 1; }
log "QUEUE_DONE"
