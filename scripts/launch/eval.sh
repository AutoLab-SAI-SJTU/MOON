#!/bin/bash
# Closed-loop evaluation arms on the val_unseen split of DATASET.
#
#   bash scripts/launch/eval.sh ARM [run_harness_eval.py arguments]
#
#   base                       frozen policy, whole split; writes OUT_ROOT/episodes.json
#   clear_only                 judge + clear-history restart without the recovery adapter, whole split;
#                              without SUFFIX it writes the alarm set OUT_ROOT/alarms_a<ALPHA>.json
#   full                       full system on the alarm set; up to the alarm it is the base policy, so the
#                              comparison stitches it onto base
#   keep_history               as full, but the recovery adapter takes over with the history kept
#   pass8                      clear_only restarts decoded by sampling, DRAWS seeds, on the alarm set
#   random_trigger             as full, restarted at random episodes and depths matched to the alarm set
#   random_trigger_clear_only  as random_trigger, without the recovery adapter
#   oracle                     as full, restarted by the ground-truth route distance, whole split
#   oracle_clear_only          as oracle, without the recovery adapter
#   always                     recovery adapter from the first step, no monitor, whole split
#   every<k>                   full system with the judge queried every k-th decision point, whole split
#   timing                     latency of base and of the full system in its deployment build (adapter merged
#                              at the alarm, no judge calls after it), TIMING_EPISODES episodes, one process per GPU
#
# Run base first, then clear_only; the other arms read their outputs. Each arm ends with
# scripts/compare_arms.py against base (OUT_ROOT/compare_<arm>.json).
# Required: NAVIDA_ROOT, BASE_MODEL; JUDGE_LORA for judged arms; RECOVERY_LORA for arms with the adapter.
# Optional: DATASET (r2r | rxr), ALPHA (the dataset's deployed alpha), OUT_ROOT (outputs/eval_<DATASET>),
# SUFFIX (appended to the arm's directory, e.g. for another adapter), SHARDS_PER_GPU (3), DRAWS (8).
# Episodes are seeded individually, so results do not depend on the number of shards.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$LAUNCH_DIR/configs/eval.env"

ARM=${1:?usage: eval.sh ARM [run_harness_eval.py arguments]}
shift
EXTRA=("$@")
DATASET=${DATASET:-r2r}
OUT_ROOT=${OUT_ROOT:-outputs/eval_$DATASET}
SUFFIX=${SUFFIX:-}
SHARDS_PER_GPU=${SHARDS_PER_GPU:-3}
DRAWS=${DRAWS:-8}

NEEDS=()
case $ARM in
  base) ;;
  clear_only|pass8|random_trigger_clear_only|oracle_clear_only) NEEDS=(JUDGE_LORA) ;;
  full|keep_history|every[0-9]*|timing) NEEDS=(JUDGE_LORA RECOVERY_LORA) ;;
  random_trigger|oracle|always) NEEDS=(RECOVERY_LORA) ;;
  *) die "unknown arm $ARM" ;;
esac
require NAVIDA_ROOT BASE_MODEL "${NEEDS[@]}"
ALPHA=$(python -c 'import sys; from moon.monitor import DEPLOYED_ALPHA
print(f"{float(sys.argv[1] or DEPLOYED_ALPHA[sys.argv[2]]):g}")' "${ALPHA:-}" "$DATASET")
EPISODES=$OUT_ROOT/episodes.json
ALARMS=$OUT_ROOT/alarms_a$ALPHA.json
SCHEDULE=$OUT_ROOT/random_trigger_a$ALPHA.json
SHARDS=$(( $(ngpu) * SHARDS_PER_GPU ))

run_arm() {  # NAME SHARDS ARGS...: one arm into OUT_ROOT/NAME
  local name=$1 n=$2
  shift 2
  run_sharded "$n" "$OUT_ROOT/$name/shard_logs" python scripts/run_harness_eval.py --split val_unseen \
    --dataset "$DATASET" --model "$BASE_MODEL" --out-dir "$OUT_ROOT/$name" "$@" "${EXTRA[@]}"
}

compare_split() {  # ARM...: arms that ran the whole split, against base
  need "$EPISODES"
  python scripts/compare_arms.py --root "$OUT_ROOT" --arms base "$@" --control base \
    --episode-list "$EPISODES" --out "$OUT_ROOT/compare_$1.json"
}

compare_stitched() {  # IDS ARM...: arms that replayed IDS only, stitched onto base
  local ids=$1
  shift
  python scripts/compare_arms.py --root "$OUT_ROOT" --arms base "$@" --control base \
    --base-log "$OUT_ROOT/base/log" --alarm-ids "$ids" --out "$OUT_ROOT/compare_$1.json"
}

judge=(--arm c1 --judge-lora "${JUDGE_LORA:-}" --alpha "$ALPHA")
recovery=(--restart-lora "${RECOVERY_LORA:-}")

case $ARM in
  base)
    run_arm base "$SHARDS" --arm a0
    python - "$OUT_ROOT/base/log" "$EPISODES" <<'PY'
import json, sys
from pathlib import Path
ids = sorted((str(json.loads(p.read_text())["id"]) for p in Path(sys.argv[1]).glob("stats_*.json")),
             key=lambda e: (0, int(e)) if e.isdigit() else (1, e))
Path(sys.argv[2]).write_text(json.dumps({"episode_ids": ids}))
print(f"[episodes] {len(ids)} -> {sys.argv[2]}")
PY
    ;;
  clear_only)
    name=clear_only_a$ALPHA$SUFFIX
    run_arm "$name" "$SHARDS" "${judge[@]}"
    if [ -z "$SUFFIX" ]; then
      tau=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["drift_threshold"])' \
        "$OUT_ROOT/$name/run_config.json")
      python scripts/compare_arms.py --root "$OUT_ROOT" --arms "$name" --write-alarm-ids "$ALARMS" --tau "$tau"
    fi
    compare_split "$name"
    ;;
  full|keep_history)
    need "$ALARMS"
    name=${ARM}_a$ALPHA$SUFFIX
    mode=()
    [ "$ARM" = keep_history ] && mode=(--restart-keeps-history)
    run_arm "$name" "$SHARDS" "${judge[@]}" "${recovery[@]}" "${mode[@]}" --episode-list "$ALARMS"
    compare_stitched "$ALARMS" "$name"
    ;;
  pass8)
    need "$ALARMS"
    names=()
    for ((k = 0; k < DRAWS; k++)); do
      name=pass8_a$ALPHA${SUFFIX}_s$k
      run_arm "$name" "$SHARDS" "${judge[@]}" --episode-list "$ALARMS" --restart-decode sample \
        --seed-base $(( SEED_BASE + DRAW_SEED_STRIDE * k ))
      names+=("$name")
    done
    python - "$OUT_ROOT" "$ALARMS" "${names[@]}" <<'PY'
import json, sys
from pathlib import Path
root, ids = Path(sys.argv[1]), json.load(open(sys.argv[2]))["episode_ids"]
draws = []
for arm in sys.argv[3:]:
    stats = [json.loads(p.read_text()) for p in (root / arm / "log").glob("stats_*.json")]
    draws.append({str(s["id"]): s["success"] >= 0.5 for s in stats})
for k in (1, 2, 4, 8):
    if k <= len(draws):
        hit = sum(any(d.get(e, False) for d in draws[:k]) for e in ids)
        print(f"pass@{k}: {hit}/{len(ids)} = {100 * hit / len(ids):.1f}%")
PY
    ;;
  random_trigger|random_trigger_clear_only)
    need "$OUT_ROOT/clear_only_a$ALPHA"
    python scripts/compare_arms.py --root "$OUT_ROOT" --arms "clear_only_a$ALPHA" --base-arm base \
      --write-trigger-schedule "$SCHEDULE"
    name=${ARM}_a$ALPHA$SUFFIX
    adapter=()
    [ "$ARM" = random_trigger ] && adapter=("${recovery[@]}")
    run_arm "$name" "$SHARDS" --arm c1 --trigger-schedule "$SCHEDULE" "${adapter[@]}" --episode-list "$SCHEDULE"
    compare_stitched "$SCHEDULE" "$name"
    ;;
  oracle|oracle_clear_only)
    name=$ARM$SUFFIX
    adapter=()
    [ "$ARM" = oracle ] && adapter=("${recovery[@]}")
    run_arm "$name" "$SHARDS" --arm c1 --trigger oracle "${adapter[@]}"
    compare_split "$name"
    ;;
  always)
    name=always$SUFFIX
    run_arm "$name" "$SHARDS" --arm a1 "${recovery[@]}"
    compare_split "$name"
    ;;
  every[0-9]*)
    k=${ARM#every}
    name=every${k}_a$ALPHA$SUFFIX
    threshold=()
    if [ "$k" != 1 ]; then
      var=QUERY_INTERVAL_TAU_$k
      [ -n "${!var:-}" ] || die "no threshold for a query interval of $k in configs/eval.env"
      threshold=(--drift-threshold "${!var}")
    fi
    run_arm "$name" "$SHARDS" "${judge[@]}" "${recovery[@]}" --judge-every "$k" "${threshold[@]}"
    compare_split "$name"
    ;;
  timing)
    n=$(ngpu)
    (( TIMING_EPISODES % n == 0 )) || die "$n GPUs do not divide $TIMING_EPISODES timed episodes"
    run_arm "timing_base$SUFFIX" "$n" --arm a0 --episodes $(( TIMING_EPISODES / n ))
    run_arm "timing_full_a$ALPHA$SUFFIX" "$n" "${judge[@]}" "${recovery[@]}" --merge-restart --stop-judge-after-budget \
      --episodes $(( TIMING_EPISODES / n ))
    python - "$OUT_ROOT/timing_base$SUFFIX" "$OUT_ROOT/timing_full_a$ALPHA$SUFFIX" <<'PY'
import json, sys
from pathlib import Path
for run in sys.argv[1:]:
    timing = [json.loads(p.read_text())["timing"] for p in Path(run).glob("events/events_*.json")]
    wall = sum(t["wall_seconds"] for t in timing)
    steps = sum(t["sim_steps"] for t in timing)
    judge = [s for t in timing for s in t["judge_seconds"]]
    line = f"{Path(run).name}: {len(timing)} episodes, {1e3 * wall / max(steps, 1):.0f} ms per action"
    if judge:
        line += f", judge {1e3 * sum(judge) / len(judge):.0f} ms per call"
    print(line)
PY
    ;;
esac

