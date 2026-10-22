#!/bin/sh
# SMPL campaign queue (012 procedure, operator 2026-10-22: all of R0/R0b, v2a-v2d, R1-R3).
# Idempotent: every step skips if done and resumes if unfinished. Re-run after any restart.
cd /home/user/strike-a-pose
U="uv run --locked"
latest() { ls -d runs/$1-2026* 2>/dev/null | sort | tail -1; }
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
probe_run() {  # $1 train run dir, $2 probe config
  [ -f "$1/metrics.json" ] || return 0; [ -f "$1/probed.txt" ] && return 0
  ck=$($U python -c "import json;print(json.load(open('$1/metrics.json'))['selection']['checkpoint'])" 2>/dev/null)
  out=$($U strike-a-pose-eval probe --config "$2" --checkpoint "$1/$ck" 2>&1 | tee -a data/smpl_queue_probe.log | grep -o "runs/probe-[a-z0-9_]*-[0-9TZ]*" | head -1)
  [ -n "$out" ] && { echo "$out ($ck)" > "$1/probed.txt"; log "probed $1 -> $out"; } || log "PROBE FAILED for $1"
}
train() {  # $1 config, $2 run_name prefix, $3 probe config
  d=$(latest "$2")
  if [ -n "$d" ] && [ -f "$d/aborted.json" ]; then log "ABORTED $d (see aborted.json); skipping"; return 0; fi
  if [ -z "$d" ]; then log "start $2"; $U strike-a-pose-train --config "$1" || log "train exit $? for $2"
  elif [ ! -f "$d/metrics.json" ]; then log "resume $d"; $U strike-a-pose-train --resume "$d" || log "resume exit $? for $d"; fi
  d=$(latest "$2"); [ -n "$d" ] && probe_run "$d" "$3"
}
dataset() {  # $1 v2x
  c=configs/smpl_proc_$1
  if [ ! -f data/generated/smpl_proc_$1/manifest.json ] || ! grep -q '"complete": true' data/generated/smpl_proc_$1/manifest.json; then
    rm -rf data/generated/smpl_proc_$1; log "generate $1"; $U strike-a-pose-data generate --config $c/generate.yaml --seed 0 || { log "GENERATE FAILED $1"; return 1; }
  fi
  [ -f splits/smpl_proc_$1.json ] || { $U strike-a-pose-data split data/generated/smpl_proc_$1 --seed 0 --out splits/smpl_proc_$1.json || return 1; log "split $1"; }
  ls -d runs/baselines-smpl_proc_$1-2026*/metrics.json >/dev/null 2>&1 || { log "baselines $1"; $U strike-a-pose-eval baseline --config $c/baselines.yaml || log "BASELINES FAILED $1"; }
  train $c/ae.yaml ae-smpl_proc_$1 $c/probe.yaml
}
log "queue start"
ls -d runs/baselines-smpl_proc_v1-2026*/metrics.json >/dev/null 2>&1 || $U strike-a-pose-eval baseline --config configs/baselines.yaml
train configs/ae.yaml ae-smpl_proc_v1 configs/probe.yaml
train configs/ae_seed1.yaml ae-seed1-smpl_proc_v1 configs/probe.yaml
# v2a gate: limb-torso intersection check (5 %) before generating
if [ ! -f data/v2a_gate.txt ]; then
  $U python scripts/intersection_check.py --config configs/smpl_proc_v2a/generate.yaml --seed 0 > data/v2a_gate.txt 2>&1; echo "exit $?" >> data/v2a_gate.txt; log "v2a gate: $(tail -1 data/v2a_gate.txt)"
fi
if grep -q "^exit 0" data/v2a_gate.txt; then dataset v2a; else log "v2a SKIPPED: intersection gate failed (data/v2a_gate.txt)"; fi
dataset v2b; dataset v2c; dataset v2d
train configs/ae_mask.yaml ae-mask-smpl_proc_v1 configs/probe.yaml
train configs/ae_vae.yaml ae-vae-smpl_proc_v1 configs/probe.yaml
train configs/ae_reg.yaml ae-reg-smpl_proc_v1 configs/probe.yaml
log "QUEUE_DONE"
