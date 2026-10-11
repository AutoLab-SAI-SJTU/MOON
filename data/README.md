# Evaluation data

Download the validation episodes and Matterport3D scenes following [VLN-CE](https://github.com/jacobkrantz/VLN-CE), [RxR](https://github.com/google-research-datasets/RxR) and [NaVIDA](https://github.com/waynechu1021/NAVIDA#-data-preparation). Follow the original datasets' licenses and access agreements. Our [ModelScope resources](../docs/CHECKPOINTS.md) are linked separately.

Place the files under the NaVIDA checkout selected by `NAVIDA_ROOT`:

```text
NAVIDA/
├── config/
│   └── vln_r2r.yaml
├── habitat_extensions/
└── data/
    ├── scene_dataset/
    │   └── mp3d/
    │       └── <scene>/
    │           ├── <scene>.glb                  # Matterport3D scene
    │           └── <scene>.navmesh
    ├── R2R_VLNCE_v1-3_preprocessed/
    │   └── <split>/
    │       └── <split>.json.gz                  # val_unseen or val_seen
    └── datasets/
        └── rxr/
            ├── val_unseen/
            │   ├── val_unseen_guide.json.gz     # original guide episodes
            │   └── val_unseen_guide_gt.json.gz  # ground truth for nDTW
            └── val_unseen_en/
                └── val_unseen_en_guide.json.gz # generated English subset
```

Prepare the English RxR split from the original guide episodes:

```bash
python scripts/build_rxr_en_split.py \
  --rxr-root "$NAVIDA_ROOT/data/datasets/rxr" --splits val_unseen
```

The script selects episodes whose instructions contain no Devanagari or Telugu characters. Evaluation reads this English subset and uses the ground truth in the original `val_unseen/` directory. The paper evaluates 1,839 R2R and 3,669 English RxR val-unseen episodes.
