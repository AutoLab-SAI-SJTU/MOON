# Inference and evaluation

Set `NAVIDA_ROOT`, `BASE_MODEL`, `JUDGE_LORA` and `RECOVERY_LORA` as in [models and data](CHECKPOINTS.md). Run commands from the repository root.

The commands use the default paper configuration. Prepare matching local checkpoints before running them.

## Direct inference

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_harness_eval.py \
  --arm c1 --dataset r2r --split val_unseen \
  --model "$BASE_MODEL" --judge-lora "$JUDGE_LORA" --restart-lora "$RECOVERY_LORA" \
  --episodes 1 --out-dir outputs/smoke_r2r \
  --stop-judge-after-budget
```

`a0`: frozen base; `c1`: judged restart; `a1`: recovery adapter from the first step. `c1` without recovery weights is clear-only. Defaults are greedy decoding and one recovery. Training splits are unsupported.

Omit `--episodes` for the full split. Direct `c1` needs no prior alarm file. `--episode-list` reads `{"episode_ids": ["1", "2"]}`. `--shard k --num-shards n` divides selected episodes deterministically.

## Optimized launcher

Run base, clear-only and full in sequence. The later arms use the outputs produced by the earlier arms.

```bash
DATASET=r2r bash scripts/launch/eval.sh base
DATASET=r2r bash scripts/launch/eval.sh clear_only
DATASET=r2r bash scripts/launch/eval.sh full

DATASET=rxr bash scripts/launch/eval.sh base
DATASET=rxr bash scripts/launch/eval.sh clear_only
DATASET=rxr bash scripts/launch/eval.sh full
```

Full replays alarmed episodes and combines the others from base. Match episode IDs and seeds; do not compare the replay-only average with the full base. `OUT_ROOT` defaults to `outputs/eval_<dataset>`; `SHARDS_PER_GPU=1` reduces concurrency; `SUFFIX` separates variants. Default alpha is 0.10 for R2R and 0.075 for RxR. Launchers use local/offline HF reads.

| Arm | Inference condition |
|---|---|
| `base`, `clear_only`, `full` | Core base/judge/clearing/recovery comparison |
| `keep_history` | Handover with retained visual history |
| `always` | Recovery from the first step |
| `random_trigger`, `random_trigger_clear_only` | Random matched schedules |
| `oracle`, `oracle_clear_only` | Offline ground-truth-route trigger, not deployment |
| `every1`, `every2`, `every4` | Query interval with matched thresholds |
| `pass8` | Sampled clear-only continuations |
| `timing` | Base/full timings; GPU count must divide episode count |

`full`, `keep_history` and `pass8` need the base/clear-only outputs. Oracle uses the simulator route; the deployed judge does not. `--merge-restart` uses a second backbone copy. `--stop-judge-after-budget` removes calls that cannot cause further recovery.

## Output and resume

Each run contains `run_config.json`, `log/stats_<id>.json` and `events/events_<id>.json`. Stats include available dataset metrics; events record executed actions, decision points, scores and handovers. Settings are pinned; changed conditions must use another output directory. Stats written last mark completed episodes. Do not edit them to make a run appear complete.

## RGB-stream API

`MoonAgent` runs the same loop without Habitat. Your environment provides RGB and reports actions that actually executed.

```python
from moon import MoonAgent

agent = MoonAgent(model=base_checkpoint, judge_lora=judge_checkpoint,
                  recovery_lora=recovery_checkpoint, dataset="r2r", device="cuda:0")
agent.begin_episode(instruction)
agent.observe(rgb)  # H x W x 3 RGB NumPy array
while not done:
    for action in agent.act():
        rgb, done = environment_step(action)
        agent.record_action(action)
        agent.observe(rgb)
        if done:
            break
```

The caller supplies checkpoint paths, `environment_step`, the initial RGB and `done`. IDs: 0 STOP, 1 FORWARD, 2 LEFT, 3 RIGHT. Call `act` once per decision point and record executed actions only. Begin each episode again to restore the base role and clear state.

`drift_threshold` overrides the dataset/alpha default when using a separately calibrated judge.

Clearing preserves the current frame and original instruction; visual history then re-accumulates. The driver's recent-action record is retained. In default one-recovery mode the judge stops after handover.
