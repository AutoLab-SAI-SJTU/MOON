# Installation

## Supported environment

Use Linux and an NVIDIA GPU for inference and benchmark evaluation. FlashAttention requires a CUDA compiler; headless Habitat requires NVIDIA EGL libraries. The CPU tests also run on Windows.

[`environment.yml`](../environment.yml) specifies Python 3.10, PyTorch 2.7.0 with CUDA 12.6, torchvision 0.22.0, Transformers 5.15.0, PEFT 0.20.0 and NumPy 1.26.4. Use an NVIDIA driver compatible with CUDA 12.6.

## Core packages

```bash
git clone https://github.com/AutoLab-SAI-SJTU/MOON.git
cd MOON
conda env create -f environment.yml
conda activate moon
python -m pip install -e . --no-deps
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Inference uses bf16 and FlashAttention 2.

## Habitat and NaVIDA

```bash
conda install -c conda-forge -c aihabitat habitat-sim=0.2.4 withbullet headless
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git external/habitat-lab
python -m pip install -e external/habitat-lab/habitat-lab -e external/habitat-lab/habitat-baselines
python -m pip install dtw fastdtw
git clone https://github.com/waynechu1021/NAVIDA.git external/NAVIDA
export NAVIDA_ROOT="$(pwd)/external/NAVIDA"
```

Use Habitat-Sim and Habitat-Lab 0.2.4 together. MOON uses NaVIDA's `config/`, `habitat_extensions/` and data layout; its metric extension requires `dtw` and `fastdtw`.

Prepare the [evaluation data](../data/README.md) and [checkpoints](CHECKPOINTS.md). Check the installation with:

```bash
python -c "import torch, transformers, peft; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(transformers.__version__, peft.__version__)"
python -c "import habitat, habitat_sim, habitat_baselines, dtw, fastdtw; print('Habitat imports OK')"
python scripts/run_harness_eval.py --help
```

Continue with the [single-episode example](../README.md#quick-start).

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing `NAVIDA_ROOT` | Point it to the checkout with `config/`, `habitat_extensions/` and `data/`. |
| EGL failure | Verify GPU visibility and NVIDIA driver/EGL libraries. |
| Adapter shape mismatch | Use the matching trained base, judge and recovery adapters. |
| Tiny weight files | Download the Git LFS objects; pointers are not weights. |
| Empty RxR split | Generate the English guide episodes with `scripts/build_rxr_en_split.py`. |
| Out of memory | Reduce `SHARDS_PER_GPU`; merging loads a second backbone copy. |
| Hub download attempted | Complete local downloads; launchers default to offline HF mode. |

VRAM use depends on history length, concurrent evaluation processes and adapter merge mode.
