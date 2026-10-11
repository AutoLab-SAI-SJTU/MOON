<div align="center">

# <img src="assets/moon_rabbit.png" width="48" alt="MOON rabbit"> MOON

**Learning When to Forget and How to Move On<br>in Vision-and-Language Navigation**

<p>
  <a href="#citation"><img src="assets/badges/paper.svg" alt="arXiv: Coming Soon"></a>
  <a href="https://github.com/AutoLab-SAI-SJTU/MOON"><img src="assets/badges/code.svg" alt="GitHub: Inference Code"></a>
  <a href="https://www.modelscope.cn/models/jmzhou/moon-iclr2027-weights"><img src="assets/badges/weights.svg" alt="ModelScope weights (private)"></a>
  <a href="https://www.modelscope.cn/datasets/jmzhou/moon-iclr2027-data"><img src="assets/badges/data.svg" alt="ModelScope data (private)"></a>
  <a href="#training"><img src="assets/badges/training.svg" alt="Training: Coming Soon"></a>
  <a href="LICENSE"><img src="assets/badges/license.svg" alt="License: Apache-2.0"></a>
</p>

</div>

![MOON overview](assets/teaser.png)

<a id="news"></a>

## 🔥 News

- **[2026.10.10]** Inference and evaluation code is available for R2R-CE and English RxR-CE.

## 📋 Contents

[Overview](#overview) · [Method](#method) · [Results](#results) · [Installation](#installation)<br>[Models and Data](#models-and-data) · [Quick Start](#quick-start) · [Evaluation](#evaluation) · [Code Structure](#code-structure)<br>[Training](#training) · [Citation](#citation) · [Acknowledgements](#acknowledgements) · [License](#license)

<a id="overview"></a>

## 🔎 Overview

**MOON** learns when to forget visual history and how to recover from a deviation in vision-and-language navigation. It adds a deviation judge and a recovery adapter to a frozen navigation policy: the judge detects a deviation, the agent clears its visual history, and the recovery adapter takes over while keeping the original instruction and current observation.

The two modules add **36.0M parameters (0.81% of the base)** and improve the base policy's success rate by **5.2 points on R2R-CE** and **6.0 points on English RxR-CE**. The same adapters transfer to open-vocabulary object navigation without object-goal retraining.

This repository provides inference and evaluation. Model weights and the experiment archive are hosted on ModelScope and currently require private access.

<a id="method"></a>

## 🧠 Method

![Deviation judging, history clearing, and recovery](assets/method.png)

Visual history supports route following but can hinder recovery after a deviation. MOON makes history clearing conditional on the agent's state:

1. **Judge:** monitor the instruction, observations, and action context for route deviation.
2. **Clear:** retain the original instruction and current frame, then clear the accumulated visual history.
3. **Recover:** switch to the recovery adapter and build a new visual history as navigation continues.

The default configuration permits one recovery per episode. The [inference guide](docs/INFERENCE.md) describes the control loop and RGB-stream API.

<a id="results"></a>

## 📊 Main Results

Paper-reported results on the full **Val-Unseen** splits with a single forward-facing RGB camera. RxR uses English instructions. SR, SPL, and nDTW are reported on a 0–100 scale.

| Method | R2R SR ↑ | R2R SPL ↑ | RxR SR ↑ | RxR SPL ↑ | RxR nDTW ↑ |
|---|---:|---:|---:|---:|---:|
| Frozen NaVIDA-Qwen3-VL base | 63.8 | 59.0 | 64.9 | 55.9 | — |
| Qwen-RobotNav 4B | 66.9 | 60.5 | 71.3 | 61.5 | 68.6 |
| Qwen-RobotNav 8B | 65.7 | 59.6 | 73.4 | 63.5 | 69.9 |
| **MOON** | **69.0** | **62.3** | **70.9** | **60.8** | **72.4** |

Qwen-RobotNav values are the single-camera results from its [technical report](https://arxiv.org/abs/2606.18112).

### Zero-shot Open-Vocabulary Object Navigation

| Method | HM3D-OVON Unseen SR ↑ | SPL ↑ |
|---|---:|---:|
| Frozen base | 40.7 | 21.7 |
| Qwen-RobotNav 4B | 53.1 | 20.9 |
| Qwen-RobotNav 8B | 51.2 | 24.0 |
| **MOON** | **57.2** | **35.4** |

This transfer experiment uses the same adapters and R2R threshold, without object-goal retraining. The current evaluation launcher supports R2R-CE and English RxR-CE; the HM3D-OVON launcher is not included.

<a id="installation"></a>

## 🛠️ Installation

Inference and benchmark evaluation use **Linux, Python 3.10, an NVIDIA GPU, PyTorch 2.7.0 (CUDA 12.6), and Habitat 0.2.4**. FlashAttention requires a compatible CUDA compiler; headless Habitat requires NVIDIA EGL libraries.

### 1. Install MOON

```bash
git clone https://github.com/AutoLab-SAI-SJTU/MOON.git
cd MOON
conda env create -f environment.yml
conda activate moon
python -m pip install -e . --no-deps
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

### 2. Install Habitat and prepare the NaVIDA checkout

```bash
conda install -c conda-forge -c aihabitat habitat-sim=0.2.4 withbullet headless
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git external/habitat-lab
python -m pip install -e external/habitat-lab/habitat-lab -e external/habitat-lab/habitat-baselines
python -m pip install dtw fastdtw
git clone https://github.com/waynechu1021/NAVIDA.git external/NAVIDA
```

MOON uses NaVIDA's Habitat configuration, benchmark extensions, and data layout. See the [installation guide](docs/INSTALL.md) for environment checks and troubleshooting.

<a id="models-and-data"></a>

## 📦 Models and Data

Download the base policy and both adapters from [ModelScope weights](https://www.modelscope.cn/models/jmzhou/moon-iclr2027-weights). **The weights repository is private; use an account with access.**

| Checkpoint | Directory in the weights repository | Path variable |
|---|---|---|
| Frozen NaVIDA-Qwen3-VL base | `base_policy/navida_qwen3vl_stage2_v2/` | `BASE_MODEL` |
| Deviation judge | `judge_checkpoint-1500/` | `JUDGE_LORA` |
| Recovery adapter | `recovery_iter15/adapter/` | `RECOVERY_LORA` |

The base is a trained Qwen3-VL-4B-Instruct navigation policy. Use it with the matching judge and recovery adapters. Calibration settings are in the [checkpoint guide](docs/CHECKPOINTS.md).

<details>
<summary><b>Download the three checkpoints with the ModelScope SDK</b></summary>

Get an SDK token from your authorized [ModelScope account](https://modelscope.cn/my/myaccesstoken). The command below prompts for it without displaying it and downloads the checkpoints to `checkpoints/moon/`.

```bash
python -m pip install modelscope
python - <<'PY'
from getpass import getpass
from modelscope import snapshot_download
from modelscope.hub.api import HubApi

HubApi().login(getpass("ModelScope SDK token: "))
snapshot_download(
    "jmzhou/moon-iclr2027-weights",
    local_dir="checkpoints/moon",
    allow_patterns=[
        "base_policy/navida_qwen3vl_stage2_v2/*",
        "judge_checkpoint-1500/*",
        "recovery_iter15/adapter/*",
    ],
)
PY
```

</details>

Prepare Matterport3D scenes and R2R-CE / RxR-CE validation episodes following the [data guide](data/README.md). The private [ModelScope data archive](https://www.modelscope.cn/datasets/jmzhou/moon-iclr2027-data) also stores environments and experiment records; the full archive is not needed for inference.

<a id="quick-start"></a>

## 🚀 Quick Start

After installing the environment and preparing checkpoints and benchmark data, run one R2R-CE episode. All commands below run from the **MOON repository root** in the activated `moon` environment.

### 1. Set local paths

These paths match the installation and download commands above. Change `NAVIDA_ROOT` or `CHECKPOINT_ROOT` if your files are elsewhere.

```bash
export NAVIDA_ROOT="$(pwd)/external/NAVIDA"
export CHECKPOINT_ROOT="$(pwd)/checkpoints/moon"
export BASE_MODEL="$CHECKPOINT_ROOT/base_policy/navida_qwen3vl_stage2_v2"
export JUDGE_LORA="$CHECKPOINT_ROOT/judge_checkpoint-1500"
export RECOVERY_LORA="$CHECKPOINT_ROOT/recovery_iter15/adapter"
```

### 2. Check the setup

```bash
python scripts/check_setup.py \
  --model "$BASE_MODEL" --judge-lora "$JUDGE_LORA" \
  --recovery-lora "$RECOVERY_LORA" --navida-root "$NAVIDA_ROOT"
```

This checks checkpoint metadata and simulator prerequisites without loading the model weights.

### 3. Run a single episode

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_harness_eval.py \
  --arm c1 --dataset r2r --split val_unseen \
  --model "$BASE_MODEL" --judge-lora "$JUDGE_LORA" \
  --restart-lora "$RECOVERY_LORA" \
  --episodes 1 --out-dir outputs/smoke_r2r \
  --stop-judge-after-budget
```

The run saves `run_config.json`, per-episode metrics in `log/stats_<id>.json`, and decisions in `events/events_<id>.json` under `outputs/smoke_r2r/`. For integration with your own environment, see the [RGB-stream API](docs/INFERENCE.md#rgb-stream-api).

<a id="evaluation"></a>

## 🔬 Evaluation

Run the frozen base, clearing alone, and MOON **in this order**, using the same local paths as Quick Start.

### R2R-CE

```bash
DATASET=r2r SHARDS_PER_GPU=1 bash scripts/launch/eval.sh base
DATASET=r2r SHARDS_PER_GPU=1 bash scripts/launch/eval.sh clear_only
DATASET=r2r SHARDS_PER_GPU=1 bash scripts/launch/eval.sh full
```

### English RxR-CE

```bash
DATASET=rxr SHARDS_PER_GPU=1 bash scripts/launch/eval.sh base
DATASET=rxr SHARDS_PER_GPU=1 bash scripts/launch/eval.sh clear_only
DATASET=rxr SHARDS_PER_GPU=1 bash scripts/launch/eval.sh full
```

These commands use one process per GPU reported by `nvidia-smi`. The full arm replays alarmed episodes and combines them with the remaining base outcomes. Results are saved under `outputs/eval_r2r/` and `outputs/eval_rxr/`.

See the [evaluation guide](docs/INFERENCE.md) for aggregation, ablations, sharding, resuming, and CPU tests.

<a id="code-structure"></a>

## 📂 Code Structure

```text
MOON/
├── moon/
│   ├── inference.py               # MoonAgent API for an RGB stream
│   ├── navida_io.py               # image preprocessing
│   ├── harness/
│   │   ├── host.py                # frozen backbone and adapter loading
│   │   ├── judge.py               # deviation judge
│   │   ├── history.py             # history sampling for the judge
│   │   ├── loop.py                # judge, clearing, and recovery loop
│   │   ├── restart.py             # original instruction and recovery handover
│   │   ├── chunking.py            # action chunks and atomic actions
│   │   └── navida_driver.py       # navigation decoding and visual history
│   └── monitor/
│       └── score.py               # rolling score and calibrated thresholds
├── scripts/
│   ├── check_setup.py             # checkpoint metadata and dependency checks
│   ├── run_harness_eval.py        # Habitat evaluation
│   ├── compare_arms.py            # aggregate and compare evaluation runs
│   ├── build_rxr_en_split.py      # prepare English RxR episodes
│   └── launch/
│       ├── eval.sh                # benchmark and ablation launcher
│       └── configs/eval.env       # evaluation defaults
├── configs/navida/                # English RxR Habitat configuration
├── data/README.md                 # benchmark downloads and directory layout
├── docs/                          # installation, checkpoints, and inference
├── assets/                        # figures and badges
├── tests/                         # CPU tests
└── environment.yml                # Python dependencies
```

<a id="training"></a>

## 🏋️ Training

Coming soon after acceptance.

<a id="citation"></a>

## 📚 Citation

The paper link and BibTeX entry will be added with the paper release.

<a id="acknowledgements"></a>

## ❤️ Acknowledgements

MOON builds on [NaVIDA](https://github.com/waynechu1021/NAVIDA), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [Habitat](https://github.com/facebookresearch/habitat-lab), and [VLN-CE](https://github.com/jacobkrantz/VLN-CE).

<a id="license"></a>

## 📄 License

The code is released under [Apache-2.0](LICENSE). Model weights, benchmark datasets, and upstream code remain subject to their respective licenses and access conditions.
