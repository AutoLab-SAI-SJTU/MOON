<div align="center">

# <img src="assets/moon_rabbit.png" width="42" alt="MOON rabbit"> MOON

**Learning When to Forget and How to Move On<br>in Vision-and-Language Navigation**

[![arXiv](assets/badges/paper.svg)](#-citation)
[![Code](assets/badges/code.svg)](https://github.com/AutoLab-SAI-SJTU/MOON)
[![Weights](assets/badges/weights.svg)](https://www.modelscope.cn/models/jmzhou/moon-iclr2027-weights)
[![Data](assets/badges/data.svg)](https://www.modelscope.cn/datasets/jmzhou/moon-iclr2027-data)

[![Training](assets/badges/training.svg)](#-todo)
[![License](assets/badges/license.svg)](LICENSE)

</div>

## 🔥 News

- **[2026.10]** Inference and evaluation code updated, with installation instructions and ModelScope resource links.

## 📝 TODO

- [x] Inference modules and RGB-stream interface
- [x] R2R-CE / English RxR-CE evaluation
- [x] Model and data resource links
- [ ] Paper and citation release
- [ ] Training code release **after paper acceptance**

## 🔎 Overview

![MOON: motivation, method, and benchmark results](assets/teaser.png)

MOON equips a frozen vision-and-language navigation policy with a **deviation judge** that decides when to forget and a **recovery policy** that learns how to move on. At an alarm, it clears the visual history while retaining the original instruction and current observation, then hands control to the recovery adapter.

The two modules add **36.0M parameters (0.81% of the base)**. They improve the frozen base's success rate by **5.2 points on R2R-CE** and **6.0 points on RxR-CE**, and transfer to open-vocabulary object-goal navigation without object-goal retraining.

## 🧠 Method

![MOON: deviation judging, history clearing, and recovery](assets/method.png)

Visual history helps an agent follow its route but can hinder recovery after a deviation. Paired interventions across four frozen VLN policies show that clearing improves recovery off route while hurting performance on route. MOON makes the intervention conditional on the agent's state.

1. **Judge:** score the instruction, sampled history, current frame, recent actions and proposed chunk; compare the rolling score with a calibrated threshold.
2. **Clear:** keep the instruction and current frame, clear visual history, and switch to the recovery adapter. The default permits at most one recovery per episode.
3. **Recover:** decode greedily on the frozen backbone plus the recovery adapter; visual history accumulates again after the handover.

<details>
<summary>🧩 Paper modules and implementation</summary>

| Component | Code | Responsibility |
|---|---|---|
| RGB-stream inference | [`inference.py`](moon/inference.py) | Public `MoonAgent` API and episode reset |
| Base and adapter switching | [`host.py`](moon/harness/host.py) | Load checkpoints and select/merge adapter roles |
| Deviation judge | [`judge.py`](moon/harness/judge.py) | Checkpoint-compatible prompt and action scoring |
| History selection | [`history.py`](moon/harness/history.py) | Sample judge frames |
| Score and thresholds | [`score.py`](moon/monitor/score.py) | Rolling log-odds and calibrated thresholds |
| Clear and handover | [`loop.py`](moon/harness/loop.py), [`restart.py`](moon/harness/restart.py) | Bound interventions and switch policy roles |
| Recovery execution | [`navida_driver.py`](moon/harness/navida_driver.py) | Greedy decoding and frame history |
| Image/action contracts | [`navida_io.py`](moon/navida_io.py), [`chunking.py`](moon/harness/chunking.py) | Backbone image sizes and atomic actions |
| Benchmark evaluation | [`run_harness_eval.py`](scripts/run_harness_eval.py) | Episode execution, sharding, resuming and metrics |
| Paired comparison | [`compare_arms.py`](scripts/compare_arms.py) | Complete/stitch outcomes and paired comparisons |

</details>

## 📊 Results

Paper-reported results on the full **Val-Unseen** splits, with **single forward-facing RGB input**. SR, SPL and nDTW use a 0–100 scale; higher is better.

| Method | R2R SR | R2R SPL | RxR SR | RxR SPL | RxR nDTW |
|---|---:|---:|---:|---:|---:|
| Frozen NaVIDA-Qwen3-VL base | 63.8 | 59.0 | 64.9 | 55.9 | — |
| Qwen-RobotNav 4B, single RGB | 66.9 | 60.5 | 71.3 | 61.5 | 68.6 |
| Qwen-RobotNav 8B, single RGB | 65.7 | 59.6 | 73.4 | 63.5 | 69.9 |
| **MOON** | **69.0** | **62.3** | **70.9** | **60.8** | **72.4** |

Qwen rows use its single-camera results. Panoramic policies and systems with external planners use different inputs/resources and should be compared separately.

### Zero-shot open-vocabulary object navigation

| Method | HM3D-OVON Unseen SR | SPL |
|---|---:|---:|
| Frozen base | 40.7 | 21.7 |
| Qwen-RobotNav 4B | 53.1 | 20.9 |
| Qwen-RobotNav 8B | 51.2 | 24.0 |
| **MOON** | **57.2** | **35.4** |

The adapters and R2R threshold are unchanged for this transfer experiment. An HM3D-OVON launcher is not included in this release; the current benchmark launcher covers R2R-CE and English RxR-CE. OVON is distinct from closed-vocabulary HM3D v2 ObjectNav.

## 📋 Requirements

| Component | Version / requirement |
|---|---|
| OS | Linux |
| Python | 3.10 |
| PyTorch | 2.7.0, CUDA 12.6 build |
| Simulator | Habitat-Sim / Habitat-Lab 0.2.4 |

See the [installation guide](docs/INSTALL.md) for FlashAttention and the complete environment.

## 🛠️ Installation

```bash
git clone https://github.com/AutoLab-SAI-SJTU/MOON.git
cd MOON
conda env create -f environment.yml
conda activate moon
python -m pip install -e . --no-deps
```

Follow the [installation guide](docs/INSTALL.md) to add Habitat and FlashAttention.

## 📦 Models and Data

| Resource | Link |
|---|---|
| Frozen base policy, deviation judge and recovery adapter | [🪄 ModelScope weights](https://www.modelscope.cn/models/jmzhou/moon-iclr2027-weights) |
| Data resources | [📦 ModelScope data](https://www.modelscope.cn/datasets/jmzhou/moon-iclr2027-data) |

The ModelScope resources are currently private and require access. See [checkpoint setup](docs/CHECKPOINTS.md) and [benchmark data preparation](data/README.md).

## 🚀 Quick Start

Prepare [checkpoint files](docs/CHECKPOINTS.md) and [benchmark data](data/README.md), then set the local paths:

```bash
export NAVIDA_ROOT=/path/to/NAVIDA
export BASE_MODEL=/path/to/navida_qwen3vl_stage2_v2
export JUDGE_LORA=/path/to/judge_checkpoint-1500
export RECOVERY_LORA=/path/to/recovery_iter15/adapter

python scripts/check_setup.py \
  --model "$BASE_MODEL" --judge-lora "$JUDGE_LORA" \
  --recovery-lora "$RECOVERY_LORA" --navida-root "$NAVIDA_ROOT"
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_harness_eval.py \
  --arm c1 --dataset r2r --split val_unseen \
  --model "$BASE_MODEL" --judge-lora "$JUDGE_LORA" \
  --restart-lora "$RECOVERY_LORA" \
  --episodes 1 --out-dir outputs/smoke_r2r \
  --stop-judge-after-budget
```

This runs the complete loop directly and saves metrics and decisions under the output directory.

## 🔬 Evaluation

```bash
# Run base, clear-only, then full; the full arm stitches unalarmed base outcomes.
DATASET=r2r bash scripts/launch/eval.sh base
DATASET=r2r bash scripts/launch/eval.sh clear_only
DATASET=r2r bash scripts/launch/eval.sh full

DATASET=rxr bash scripts/launch/eval.sh base
DATASET=rxr bash scripts/launch/eval.sh clear_only
DATASET=rxr bash scripts/launch/eval.sh full
```

See the [inference guide](docs/INFERENCE.md) for metrics, ablations, sharding, resuming and the RGB-stream API.

## 📂 Repository Layout

```text
moon/inference.py      RGB-stream API
moon/navida_io.py      image contract
moon/harness/          model host, judge, driver and control loop
moon/monitor/          score and inference thresholds
configs/navida/        English RxR evaluation config
scripts/               setup checks, inference, evaluation preparation and metrics
scripts/launch/        evaluation launcher and configuration
docs/                  installation, resources, inference and release scope
assets/                paper figures and icon
tests/                 model-free / CPU inference tests
```

## 🧪 Tests

```bash
python -m pip install -e '.[test]'
python -m pytest tests -q
```

GitHub Actions checks the CPU inference tests and builds the package.

## ❤️ Acknowledgements

Built on [NaVIDA](https://github.com/waynechu1021/NAVIDA), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [Habitat](https://github.com/facebookresearch/habitat-lab) and [VLN-CE](https://github.com/jacobkrantz/VLN-CE). Qwen comparison values come from its [technical report](https://arxiv.org/abs/2606.18112).

## 📄 License

Code uses [Apache-2.0](LICENSE). Third-party models, benchmark datasets and upstream code retain their own licenses/access conditions. Resource files do not automatically inherit the code license.

## 📚 Citation

The paper link, author list and BibTeX entry will be added with the paper release. Training code is planned for release after acceptance.
