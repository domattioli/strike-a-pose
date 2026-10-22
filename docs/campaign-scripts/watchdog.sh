#!/bin/sh
# Keep-alive + progress watchdog (run as a tracked background task, <= 110 min per arm).
# Every 5 min: if the queue is not running and not done -> relaunch it.
# If a train run's history.csv / log has not advanced for > max(3 x median epoch, 15 min)
# while its lock owner is alive -> STALL: kill the trainer (resume picks up from checkpoint).
cd /home/user/strike-a-pose
end=$(( $(date +%s) + ${1:-6600} ))
qrun() { ! flock -n data/smpl_queue.lock true; }  # lock held = a queue is running
while [ $(date +%s) -lt $end ]; do
  grep -q QUEUE_DONE data/smpl_queue.log 2>/dev/null && { echo "WD $(date -u +%H:%M) queue done"; exit 0; }
  if ! qrun; then echo "WD $(date -u +%H:%M) queue not running -> relaunch"; setsid nohup data/smpl_queue.sh >> data/smpl_queue.log 2>&1 < /dev/null & sleep 5; fi
  r=$(ls -td runs/ae-*smpl_proc_* 2>/dev/null | head -1)
  if [ -n "$r" ] && [ ! -f "$r/metrics.json" ] && [ -f "$r/history.csv" ]; then
    last=$(stat -c %Y "$r/history.csv"); [ -f "$r/lock" ] && lk=$(stat -c %Y "$r/lock") && [ "$lk" -gt "$last" ] && last=$lk
    age=$(( $(date +%s) - last ))
    med=$(awk -F, 'NR>1{print $5}' "$r/history.csv" | sort -n | awk '{a[NR]=$1}END{print (NR? int(a[int((NR+1)/2)]) : 0)}')
    lim=$(( med*3 > 900 ? med*3 : 900 )); n=$(( $(wc -l < "$r/history.csv") - 1 ))
    if [ "$n" -ge 1 ] && [ $age -gt $lim ]; then
      pid=$(python3 -c "import json;print(json.load(open('$r/lock'))['pid'])" 2>/dev/null)
      echo "WD $(date -u +%H:%M) STALL $r: no epoch for ${age}s (limit ${lim}s) -> kill $pid, queue will resume"
      [ -n "$pid" ] && kill "$pid" 2>/dev/null
    fi
  fi
  sleep 300
done
echo "WD $(date -u +%H:%M) arm expired; $(tail -1 data/smpl_queue.log)"
