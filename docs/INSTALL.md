# Installation

## Supported environment

Benchmark inference targets Linux, Python 3.10, an NVIDIA driver compatible with PyTorch's CUDA 12.6 build, and a CUDA compiler for FlashAttention. Habitat needs NVIDIA EGL libraries. Windows supports the CPU tests, not the simulator installation.

[`environment.yml`](../environment.yml) pins PyTorch 2.7.0, torchvision 0.22.0, Transformers 5.15.0, PEFT 0.20.0 and the core dependencies. NumPy 1.26.4 satisfies numpy-quaternion 2024.0.8's requirement of NumPy >=1.25; the old NumPy 1.24 pin did not.

## Core packages

```bash
git clone https://github.com/AutoLab-SAI-SJTU/MOON.git
cd MOON
conda env create -f environment.yml
conda activate moon
python -m pip install -e . --no-deps
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

The host uses bf16 and `flash_attention_2`; a CPU-only installation cannot run the model path.

## Habitat and NaVIDA

```bash
conda install -c conda-forge -c aihabitat habitat-sim=0.2.4 withbullet headless
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git external/habitat-lab
python -m pip install -e external/habitat-lab/habitat-lab -e external/habitat-lab/habitat-baselines
python -m pip install dtw fastdtw
git clone https://github.com/waynechu1021/NAVIDA.git external/NAVIDA
export NAVIDA_ROOT="$(pwd)/external/NAVIDA"
```

Use Habitat-Sim and Habitat-Lab 0.2.4 together. NaVIDA's metric extension imports both `dtw` and `fastdtw`. MOON uses NaVIDA's `config/`, `habitat_extensions/` and benchmark layout; no upstream training launcher is needed. Avoid installing NaVIDA's older training dependency recipe over this Qwen3-VL inference environment.

Prepare [data](../data/README.md) and [checkpoints](CHECKPOINTS.md), then verify:

```bash
python -c "import torch, transformers, peft; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(transformers.__version__, peft.__version__)"
python -c "import habitat, habitat_sim, habitat_baselines, dtw, fastdtw; print('Habitat imports OK')"
python scripts/run_harness_eval.py --help
```

Finally run the checkpoint check and single-episode command in the README. Checking metadata does not load or validate model predictions.

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing `NAVIDA_ROOT` | Point it to the checkout with `config/`, `habitat_extensions/` and `data/`. |
| EGL failure | Verify GPU visibility and NVIDIA driver/EGL libraries. |
| Adapter shape mismatch | Use the matching trained base, judge and recovery adapters. |
| Tiny weight files | Download the Git LFS objects; pointers are not weights. |
| Empty RxR split | Prepare the supplied English guide evaluation files. |
| Out of memory | Reduce `SHARDS_PER_GPU`; merging loads a second backbone copy. |
| Hub download attempted | Complete local downloads; launchers default to offline HF mode. |

VRAM depends on checkpoint, history, concurrent processes and merge mode. The paper's training GPU memory does not establish a measured inference minimum.
