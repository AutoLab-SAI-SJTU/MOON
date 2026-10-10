# Evaluation data

Prepare the original benchmark scenes and validation episodes separately. Original benchmark licenses apply; see [models and data](../docs/CHECKPOINTS.md) for resource links.

Under the checkout selected by `NAVIDA_ROOT`:

```text
config/vln_r2r.yaml
habitat_extensions/
data/scene_dataset/mp3d/<scene>/<scene>.glb
data/scene_dataset/mp3d/<scene>/<scene>.navmesh
data/datasets/r2r/                              # upstream config paths
data/datasets/rxr/val_unseen/val_unseen_guide.json.gz
data/datasets/rxr/val_unseen/val_unseen_guide_gt.json.gz
data/datasets/rxr/val_unseen_en/val_unseen_en_guide.json.gz
```

```bash
python scripts/build_rxr_en_split.py \
  --rxr-root "$NAVIDA_ROOT/data/datasets/rxr" --splits val_unseen
```

The config reads filtered guide episodes and the original metric ground truth. The paper uses 1,839 R2R and 3,669 English RxR val-unseen episodes; verify counts against the actual export. No judge training rows, ScaleVLN inputs, GRPO state pools or expert demonstrations are required. Evaluation episode/alarm lists are created under `outputs/eval_<dataset>/`.
