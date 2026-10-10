"""Habitat evaluation environments for R2R-CE and English RxR-CE.

Benchmark configs and data live in the NaVIDA checkout selected by NAVIDA_ROOT.
The English RxR evaluation config is supplied in configs/navida/.
"""
import os
import sys
from pathlib import Path

REPO_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "navida"

# Upstream eval.py forces STOP after this many rotation-only steps, or steps in total.
SPIN_LIMIT = 25
MAX_TOTAL_STEPS = 400


# Evaluation only. Training splits/configs are withheld from this release.
DATASET_CONFIGS = {"r2r": "vln_r2r.yaml", "rxr": "vln_rxr_en.yaml"}


def set_navida_root(path) -> None:
    """Point this process, and the workers it starts afterwards, at a NaVIDA checkout."""
    if path is not None:
        os.environ["NAVIDA_ROOT"] = str(Path(path).resolve())


def navida_root() -> Path:
    root = os.environ.get("NAVIDA_ROOT")
    if not root:
        raise RuntimeError("NAVIDA_ROOT is not set: point it (or --navida-root) at the NaVIDA "
                           "checkout that holds config/ and data/")
    root = Path(root)
    if not (root / "config").is_dir():
        raise RuntimeError(f"NAVIDA_ROOT={root} has no config/ directory")
    return root


def config_path(split: str, dataset: str = "r2r") -> Path:
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"unknown dataset {dataset!r}, known: {sorted(DATASET_CONFIGS)}")
    if split not in ("val_seen", "val_unseen"):
        raise ValueError("this release supports evaluation splits only")
    name = DATASET_CONFIGS[dataset]
    local = REPO_CONFIG_DIR / name
    return local if local.exists() else navida_root() / "config" / name


def navida_config(split: str, dataset: str = "r2r"):
    """Habitat config for ``split``; its data paths are relative to the NaVIDA root."""
    root = navida_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import habitat
    from habitat_baselines.config.default import get_config
    from habitat_extensions import measures, task  # noqa: F401  (registers R2R-CE)

    os.chdir(root)
    cfg = get_config(str(config_path(split, dataset)))
    with habitat.config.read_write(cfg):
        cfg.habitat.dataset.split = split
    return cfg


def build_env(split: str, dataset: str = "r2r"):
    from habitat import Env
    return Env(navida_config(split, dataset).habitat)


def reset_to(env, episode):
    """Land on one specific episode.

    Assigning current_episode is the supported way in: its setter makes reset()
    keep the episode instead of pulling the next one. Writing _episode_iterator
    directly leaves the first reset on habitat's own pre-fetched episode, in a
    different scene, so the check stays.
    """
    env.current_episode = episode
    obs = env.reset()
    got = str(env.current_episode.episode_id)
    if got != str(episode.episode_id):
        raise RuntimeError(f"reset landed on episode {got}, wanted {episode.episode_id}")
    return obs


def seed_all(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

