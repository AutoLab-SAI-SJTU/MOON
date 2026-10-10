# Sourced by every launcher: repository paths, offline Hugging Face, thread caps and shared helpers.
# Relative paths given to the launchers are taken from the repository root.
LAUNCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$LAUNCH_DIR/../.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# many processes per node: default thread pools would oversubscribe the cores
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

die() { echo "error: $*" >&2; exit 1; }

require() {
  local name
  for name in "$@"; do
    [ -n "${!name:-}" ] || die "set $name (see README.md)"
  done
}

need() { [ -e "$1" ] || die "$1 not found"; }

ngpu() {
  local n
  n=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || die "no GPU visible to nvidia-smi"
  echo "$n"
}

# run_sharded N LOG_DIR COMMAND...: N processes of COMMAND with --shard k --num-shards N, shard k pinned
# to GPU k mod the GPU count (habitat's EGL renderer always opens CUDA device 0 of its process).
run_sharded() {
  local n=$1 logdir=$2 gpus k failed=0
  local pids=()
  shift 2
  gpus=$(ngpu)
  mkdir -p "$logdir"
  for ((k = 0; k < n; k++)); do
    CUDA_VISIBLE_DEVICES=$((k % gpus)) "$@" --shard "$k" --num-shards "$n" > "$logdir/shard$k.log" 2>&1 &
    pids+=($!)
  done
  for k in "${pids[@]}"; do wait "$k" || failed=1; done
  [ "$failed" = 0 ] || die "a shard failed, see $logdir/shard*.log"
}

