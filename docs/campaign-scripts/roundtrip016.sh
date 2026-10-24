#!/bin/sh
# Spec 016 round-trip campaign: waits for the grill queue to end, then runs the probe round trip
# on the three existing sample runs (R2 v1, R2b v1, R2 v3). Idempotent: a config whose
# runs/<run_name>-2026*/samples.json exists is skipped.
# Do not start while the queue runs; the lead or operator launches it:
#   nohup sh docs/campaign-scripts/roundtrip016.sh &
cd /home/user/strike-a-pose || exit 1
set -u
U="uv run --locked"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

while pgrep -f grill_queue.sh > /dev/null; do
  log "waiting for grill_queue.sh"
  sleep 300
done

for cfg in configs/roundtrip_vae.yaml configs/roundtrip_vae_beta0.1.yaml configs/roundtrip_vae_smpl_proc_v3.yaml; do
  name=$(sed -n 's/^run_name: *\([^ #]*\).*/\1/p' "$cfg")
  if ls runs/"$name"-2026*/samples.json > /dev/null 2>&1; then
    log "skip $cfg (done)"
    continue
  fi
  log "start $cfg"
  $U strike-a-pose-eval round-trip --config "$cfg" >> data/roundtrip016.log 2>&1 \
    || log "ROUND-TRIP FAILED $cfg"
done
# the beta 0.1 v3 sample run (grill queue) gets its own config once its probe run exists
log "BETA0.1-V3 PENDING"
log ROUNDTRIP_DONE
