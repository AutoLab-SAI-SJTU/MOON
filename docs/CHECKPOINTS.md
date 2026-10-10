# Models and data

## Project resources

| Resource | Location | Access |
|---|---|---|
| Base policy, judge and recovery checkpoints | [ModelScope weights](https://www.modelscope.cn/models/jmzhou/moon-iclr2027-weights) | Private; access required |
| Data and experiment archive | [ModelScope data](https://www.modelscope.cn/datasets/jmzhou/moon-iclr2027-data) | Private maintainer archive |

The public inference release requires matching local model files and the original benchmark inputs. Complete backup archives are not required to run inference.

## Local checkpoints

| Role | Variable | Directory in the weights repository | Required content |
|---|---|---|---|
| Frozen navigation base | `BASE_MODEL` | `base_policy/navida_qwen3vl_stage2_v2/` | `config.json`, tokenizer/processor and complete model weights/shards |
| Deviation judge | `JUDGE_LORA` | `judge_checkpoint-1500/` | `adapter_config.json` and adapter weights |
| Recovery policy | `RECOVERY_LORA` | `recovery_iter15/adapter/` | `adapter_config.json` and adapter weights |

The base is a trained NaVIDA navigation policy on Qwen3-VL-4B-Instruct. Vanilla Qwen3-VL and the upstream Qwen2.5-VL NaVIDA checkpoint are different models.

```bash
python scripts/check_setup.py \
  --model "$BASE_MODEL" --judge-lora "$JUDGE_LORA" --recovery-lora "$RECOVERY_LORA" \
  --navida-root "$NAVIDA_ROOT"
```

## Inference defaults

| Dataset | Alpha | Threshold | Score window |
|---|---:|---:|---:|
| R2R-CE | 0.10 | 2.54 | 3 |
| English RxR-CE | 0.075 | 3.33 | 3 |

These are the code's default paper settings. Use a judge's matching calibration; override a threshold with `--drift-threshold` or `MoonAgent(..., drift_threshold=...)` when needed. The score is `log p(lost) - log p(OK)`, averaged over the window. The checkpoint-compatible prompt retains the labels `OK`, `lost` and `amb`.

## Benchmark data

Obtain simulator scenes and evaluation episodes through [VLN-CE](https://github.com/jacobkrantz/VLN-CE), [RxR](https://github.com/google-research-datasets/RxR) and [NaVIDA](https://github.com/waynechu1021/NAVIDA#-data-preparation). Follow their licenses and access agreements, and use the [evaluation layout](../data/README.md).

```bash
python scripts/build_rxr_en_split.py \
  --rxr-root "$NAVIDA_ROOT/data/datasets/rxr" --splits val_unseen
```

The script follows the paper's Latin-script filtering on the guide export. The reported export yields 3,669 val-unseen episodes; check the printed count for your input. Ground-truth nDTW files remain in the original split directory.
