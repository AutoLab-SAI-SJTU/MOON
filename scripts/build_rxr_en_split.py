"""Rebuild the English RxR-CE episode files that configs/navida/vln_rxr_en.yaml
loads.

RxR ships one episode per instruction in English, Hindi and Telugu, and the
VLN-CE export the loader reads may carry no per-episode language, so the
English split is selected by script: an episode is kept when its instruction
text contains no Devanagari or Telugu character. Episodes and the file's other
top-level keys are copied unchanged, in source order. On the RxR-CE guide files
the reported val_unseen evaluation uses 3669 English episodes. This script
prepares evaluation episodes only; it does not generate training examples.

Usage, from the NaVIDA checkout:
  python <this repo>/scripts/build_rxr_en_split.py --rxr-root data/datasets/rxr
"""
import argparse
import gzip
import json
import re
from pathlib import Path

NON_LATIN_SCRIPT = re.compile(r"[ऀ-ॿఀ-౿]")   # Devanagari, Telugu

# Output layout relative to --rxr-root, as the data_path of the two configs spells it.
EVAL_PATTERN = "{split}_en/{split}_en_{role}.json.gz"        # vln_rxr_en.yaml
SOURCE_PATTERN = "{split}/{split}_{role}.json.gz"


def is_english(episode) -> bool:
    return not NON_LATIN_SCRIPT.search(episode["instruction"]["instruction_text"])


def english_path(root: Path, split: str, role: str) -> Path:
    if split not in ("val_seen", "val_unseen"):
        raise ValueError("only val_seen and val_unseen are supported in this release")
    pattern = EVAL_PATTERN
    return Path(root) / pattern.format(split=split, role=role)


def english_subset(doc):
    """The same document with only the English episodes."""
    return dict(doc, episodes=[e for e in doc["episodes"] if is_english(e)])


def build_split(rxr_root: Path, split: str, role: str, out_root: Path) -> Path:
    src = Path(rxr_root) / SOURCE_PATTERN.format(split=split, role=role)
    with gzip.open(src, "rt") as f:
        doc = json.load(f)
    sub = english_subset(doc)
    out = english_path(out_root, split, role)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump(sub, f)
    print(f"[rxr-en] {split}: {len(sub['episodes'])} of {len(doc['episodes'])} episodes -> {out}")
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rxr-root", required=True, type=Path,
                    help="RxR-CE directory holding <split>/<split>_<role>.json.gz")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="where the English files go; default --rxr-root")
    ap.add_argument("--splits", nargs="+", choices=["val_seen", "val_unseen"], default=["val_unseen"])
    ap.add_argument("--role", default="guide")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for split in args.splits:
        build_split(args.rxr_root, split, args.role, args.out_root or args.rxr_root)


if __name__ == "__main__":
    main()

