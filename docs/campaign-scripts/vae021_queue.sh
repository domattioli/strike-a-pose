#!/bin/sh
# Spec 021 campaign queue (operator scope 2026-11-03: generative pose VAE; surrogate rulings
# Q1-Q10): on smpl_proc_v3, V1 (128-d VAE beta 0.1 + pose head 0.03 on mu) then V2 (128-d VAE
# beta 0.1, control), each probed (configs/smpl_proc_v3/probe.yaml); then N(0, I) samples of
# both (reference L1), then probe round trips of V1, V2 and, retroactively, R2b-v3. ~15 h.
# Same per-machine lock data/smpl_queue.lock as every smpl queue (exits 0 while held; re-run).
# 12 GB memory gate after V1 (beta 0.1 on v1 once peaked 11.6 GB, unexplained), OOM guard with
# boot id, watchdog SIGTERM (exit 143) not counted as OOM (pattern of fidelity020_queue.sh).
# Copy to data/vae021_queue.sh; watchdog: LOG=data/vae021_queue.log
# QUEUE=data/vae021_queue.sh data/watchdog.sh
# Idempotent: every step skips if done and resumes if unfinished. Re-run after any restart.
# Terminal lines: the done marker (printed once, last line), QUEUE_INCOMPLETE, MEMORY_STOP.
cd /home/user/strike-a-pose
exec 9> data/smpl_queue.lock
flock -n 9 || { echo "[$(date -u +%H:%M:%S)] another queue holds the lock; exiting"; exit 0; }
U="uv run --locked"
MEM_LIMIT_MB=12288
c=configs/smpl_proc_v3
V1=ae-l128-vae-beta0.1-pose-l0.03-smpl_proc_v3
V2=ae-l128-vae-beta0.1-smpl_proc_v3
L1=ae-l128-smpl_proc_v3   # plain 128-d AE: reference_checkpoint of both sample runs
R2B=runs/sample-vae-beta0.1-smpl_proc_v3-20261024T143458Z   # R2b-v3 sample run (spec 015)
latest() { ls -d runs/$1-2026* 2>/dev/null | sort | tail -1; }
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
memstop() { log "MEMORY_STOP $*"; exit 0; }
selected() {  # $1 train run dir -> its scored checkpoint path
  ck=$($U python -c "import json;print(json.load(open('$1/metrics.json'))['selection']['checkpoint'])" 2>/dev/null)
  [ -n "$ck" ] && echo "$1/$ck"
}
probe_of() {  # $1 train run dir -> its probe run dir (first word of probed.txt)
  [ -f "$1/probed.txt" ] && cut -d' ' -f1 "$1/probed.txt"
}
probe_run() {  # $1 train run dir, $2 probe config
  [ -f "$1/metrics.json" ] || return 0; [ -f "$1/probed.txt" ] && return 0
  ck=$(selected "$1")
  out=$($U strike-a-pose-eval probe --config "$2" --checkpoint "$ck" 2>&1 | tee -a data/vae021_queue_probe.log | grep -o "runs/probe-[a-z0-9_.]*-[0-9TZ]*" | head -1)
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
    # OOM guard: a second failed attempt on the same boot without metrics -> terminal stop
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
sample() {  # $1 sample config, $2 train prefix, $3 sample run_name prefix
  v=$(latest "$2"); [ -n "$v" ] && [ -f "$v/metrics.json" ] || { log "SAMPLE SKIPPED (no finished $2)"; return 0; }
  s=$(latest "$3"); [ -n "$s" ] && [ -f "$s/samples.json" ] && return 0
  r=$(latest "$L1"); [ -n "$r" ] && [ -f "$r/metrics.json" ] || { log "SAMPLE FAILED $3 (no L1 reference)"; return 0; }
  log "sample $3"
  $U strike-a-pose-eval sample --config "$1" --checkpoint "$(selected "$v")" --reference-checkpoint "$(selected "$r")" >> data/vae021_queue_sample.log 2>&1 || log "SAMPLE FAILED $3"
}
roundtrip() {  # $1 round-trip config, $2 sample run dir, $3 probe run dir: marker roundtripped.txt in the sample run
  [ -n "$2" ] && [ -f "$2/samples.json" ] && [ -n "$3" ] || { log "ROUND-TRIP SKIPPED $1 (sample or probe missing)"; return 0; }
  [ -f "$2/roundtripped.txt" ] && return 0
  out=$($U strike-a-pose-eval round-trip --config "$1" --sample-run "$2" --probe-run "$3" 2>&1 | tee -a data/vae021_queue_roundtrip.log | grep -o "runs/roundtrip-[a-z0-9_.-]*-[0-9]\{8\}T[0-9]\{6\}Z" | head -1)
  [ -n "$out" ] && { echo "$out ($3)" > "$2/roundtripped.txt"; log "round trip $2 -> $out"; } || log "ROUND-TRIP FAILED for $2"
}
[ -f data/generated/smpl_proc_v3/manifest.json ] && grep -q '"complete": true' data/generated/smpl_proc_v3/manifest.json && [ -f splits/smpl_proc_v3.json ] || { log "QUEUE_INCOMPLETE: smpl_proc_v3 data or split missing (run the 015 queue first)"; exit 1; }
[ -f scripts/generative.py ] && $U python -c "from strike_a_pose.train.config import load_config as f;f('$c/ae_l128_vae_beta0.1_pose_l0.03.yaml')" >/dev/null 2>&1 || { log "QUEUE_INCOMPLETE: slice 021 code missing (implement B1-B2 first)"; exit 1; }
[ -f assets/smpl/SMPL_NEUTRAL.npz ] && [ -f "$R2B/samples.json" ] && [ -n "$(latest $L1)" ] || { log "QUEUE_INCOMPLETE: SMPL model, R2b-v3 sample run or L1 run missing"; exit 1; }
pending() {  # V1 and V2 trained (or aborted) and probed; both sampled; three round trips done
  for n in $V1 $V2; do d=$(latest $n); [ -z "$d" ] && return 0; [ -f "$d/aborted.json" ] && continue; [ -f "$d/metrics.json" ] || return 0; [ -f "$d/probed.txt" ] || return 0; done
  for n in $V1 $V2; do d=$(latest $n); [ -f "$d/aborted.json" ] && continue; s=$(latest sample-l128-vae-beta0.1${n#ae-l128-vae-beta0.1}); [ -n "$s" ] && [ -f "$s/roundtripped.txt" ] || return 0; done
  [ -f "$R2B/roundtripped.txt" ] || return 0
  return 1
}
pass=0
while [ $pass -lt 5 ]; do
pass=$((pass+1)); log "pass $pass"
train $c/ae_l128_vae_beta0.1_pose_l0.03.yaml $V1 $c/probe.yaml
d=$(latest $V1); [ -n "$d" ] && [ -f "$d/run_info.json" ] && mem_gate "$d"
train $c/ae_l128_vae_beta0.1.yaml $V2 $c/probe.yaml
sample $c/sample_l128_vae_beta0.1_pose_l0.03.yaml $V1 sample-l128-vae-beta0.1-pose-l0.03-smpl_proc_v3
sample $c/sample_l128_vae_beta0.1.yaml $V2 sample-l128-vae-beta0.1-smpl_proc_v3
roundtrip $c/roundtrip_l128_vae_beta0.1_pose_l0.03.yaml "$(latest sample-l128-vae-beta0.1-pose-l0.03-smpl_proc_v3)" "$(probe_of "$(latest $V1)")"
roundtrip $c/roundtrip_l128_vae_beta0.1.yaml "$(latest sample-l128-vae-beta0.1-smpl_proc_v3)" "$(probe_of "$(latest $V2)")"
roundtrip $c/roundtrip_vae_beta0.1.yaml "$R2B" runs/probe-smpl_proc_v3-20261024T134824Z
pending || break
done
pending && { log "QUEUE_INCOMPLETE after $pass passes (see FAILED/exit lines)"; exit 1; }
log "QUEUE_DONE"
