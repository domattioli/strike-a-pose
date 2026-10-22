#!/bin/sh
# Spec 015 campaign queue (operator 2026-10-22; order: surrogate ruling Q2 (a)):
#   R2b (VAE beta 0.1, v1) + probe -> sampling of R2b and R2 (v1) ->
#   smpl_proc_v3: generate, split, baselines [mem gate], R0-v3 [mem gate], R0b-v3, R2-v3,
#   R1-v3 (probe after each) -> sampling of R2-v3.
# Idempotent: every step skips if done and resumes if unfinished. Re-run after any restart.
# Terminal lines: QUEUE_DONE, QUEUE_INCOMPLETE, MEMORY_STOP (the watchdog stops on the first
# and the last; spec 015 FR-005, analyze A1/A2).
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
  out=$($U strike-a-pose-eval probe --config "$2" --checkpoint "$ck" 2>&1 | tee -a data/smpl015_queue_probe.log | grep -o "runs/probe-[a-z0-9_.]*-[0-9TZ]*" | head -1)
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
    [ -f "$d/resume_tried" ] && memstop "$d unfinished after a resume (likely OOM)"
    touch "$d/resume_tried"; log "resume $d"; $U strike-a-pose-train --resume "$d"; rc=$?
  else rc=0; fi
  [ "$rc" = 137 ] && memstop "$2 trainer killed (exit 137)"
  [ "$rc" != 0 ] && log "train exit $rc for $2"
  d=$(latest "$2"); [ -n "$d" ] && probe_run "$d" "$3"
}
sample_run() {  # $1 config, $2 VAE train prefix, $3 reference AE train prefix, $4 output prefix
  ls runs/$4-2026*/samples.json >/dev/null 2>&1 && return 0
  v=$(latest "$2"); r=$(latest "$3")
  [ -n "$v" ] && [ -f "$v/metrics.json" ] && [ -n "$r" ] && [ -f "$r/metrics.json" ] || { log "SAMPLE SKIPPED $4 (runs not finished)"; return 0; }
  log "sample $4"
  $U strike-a-pose-eval sample --config "$1" --checkpoint "$(selected "$v")" --reference-checkpoint "$(selected "$r")" >> data/smpl015_queue_sample.log 2>&1 || log "SAMPLE FAILED $4"
}
v3_data() {
  c=configs/smpl_proc_v3
  if [ ! -f data/generated/smpl_proc_v3/manifest.json ] || ! grep -q '"complete": true' data/generated/smpl_proc_v3/manifest.json; then
    rm -rf data/generated/smpl_proc_v3; log "generate v3"; $U strike-a-pose-data generate --config $c/generate.yaml --seed 0 || { log "GENERATE FAILED v3"; return 1; }
  fi
  [ -f splits/smpl_proc_v3.json ] || { $U strike-a-pose-data split data/generated/smpl_proc_v3 --seed 0 --out splits/smpl_proc_v3.json || return 1; log "split v3"; }
  [ -f data/v3_load_split.txt ] || { $U python -c "
import time
from strike_a_pose.data import PoseImageDataset
from strike_a_pose.train.trainer import load_split
ds = PoseImageDataset('data/generated/smpl_proc_v3', split='train', split_file='splits/smpl_proc_v3.json')
t = time.monotonic(); a = load_split(ds, 256); print(f'{a[\"n\"]} images {time.monotonic() - t:.0f}')
" > data/v3_load_split.txt 2>&1; log "load_split v3 $(tail -1 data/v3_load_split.txt) s"; }
  b=$(ls -d runs/baselines-smpl_proc_v3-2026* 2>/dev/null | sort | tail -1)
  if [ -z "$b" ] || [ ! -f "$b/metrics.json" ]; then log "baselines v3"; $U strike-a-pose-eval baseline --config $c/baselines.yaml; rc=$?
    [ "$rc" = 137 ] && memstop "baselines v3 killed (exit 137)"
    [ "$rc" != 0 ] && { log "BASELINES FAILED v3"; return 1; }
    b=$(ls -d runs/baselines-smpl_proc_v3-2026* | sort | tail -1)
  fi
  mem_gate "$b"
}
TRAIN="ae-vae-beta0.1-smpl_proc_v1 ae-smpl_proc_v3 ae-seed1-smpl_proc_v3 ae-vae-smpl_proc_v3 ae-mask-smpl_proc_v3"
SAMPLES="sample-vae-beta0.1-smpl_proc_v1 sample-vae-smpl_proc_v1 sample-vae-smpl_proc_v3"
pending() {  # exactly the five train prefixes and three sample outputs (analyze A3)
  for n in $TRAIN; do d=$(latest $n); [ -z "$d" ] && return 0; [ ! -f "$d/metrics.json" ] && [ ! -f "$d/aborted.json" ] && return 0; done
  for n in $SAMPLES; do ls runs/$n-2026*/samples.json >/dev/null 2>&1 || return 0; done
  return 1
}
pass=0
while [ $pass -lt 5 ]; do
pass=$((pass+1)); log "pass $pass"
train configs/ae_vae_beta0.1.yaml ae-vae-beta0.1-smpl_proc_v1 configs/probe.yaml
sample_run configs/sample_vae_beta0.1.yaml ae-vae-beta0.1-smpl_proc_v1 ae-smpl_proc_v1 sample-vae-beta0.1-smpl_proc_v1
sample_run configs/sample_vae.yaml ae-vae-smpl_proc_v1 ae-smpl_proc_v1 sample-vae-smpl_proc_v1
if v3_data; then
  c=configs/smpl_proc_v3
  train $c/ae.yaml ae-smpl_proc_v3 $c/probe.yaml
  d=$(latest ae-smpl_proc_v3); [ -n "$d" ] && [ -f "$d/metrics.json" ] && mem_gate "$d"
  train $c/ae_seed1.yaml ae-seed1-smpl_proc_v3 $c/probe.yaml
  train $c/ae_vae.yaml ae-vae-smpl_proc_v3 $c/probe.yaml
  train $c/ae_mask.yaml ae-mask-smpl_proc_v3 $c/probe.yaml
  sample_run $c/sample_vae.yaml ae-vae-smpl_proc_v3 ae-smpl_proc_v3 sample-vae-smpl_proc_v3
fi
pending || break
done
pending && { log "QUEUE_INCOMPLETE after $pass passes (see FAILED/exit lines)"; exit 1; }
log "QUEUE_DONE"
