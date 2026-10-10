import gzip
import importlib.util
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_rxr_en_split",
                                               REPO / "scripts" / "build_rxr_en_split.py")
rxr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rxr)


def _episode(ep_id, text):
    return {"episode_id": ep_id, "trajectory_id": ep_id, "scene_id": "mp3d/x/x.glb",
            "instruction": {"instruction_text": text}, "info": {"role": "guide"},
            "goals": [{"position": [0, 0, 0], "radius": 3.0}], "reference_path": [[0, 0, 0]],
            "start_position": [0, 0, 0], "start_rotation": [0, 0, 0, 1]}


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(doc, f)


def test_keeps_latin_script_instructions_in_order_with_the_other_keys(tmp_path):
    vocab = {"word_list": ["<pad>", "go"], "num_vocab": 2}
    episodes = [_episode(1, "Walk past the sofa and stop at the door."),
                _episode(2, "सीढ़ियों से नीचे जाइए"),
                _episode(3, "మెట్ల పైకి వెళ్ళండి"),
                _episode(4, "Ikada talupu vaipu velandi"),
                _episode(5, "Turn right ’til the hallway.")]
    for split in ("val_seen", "val_unseen"):
        _write(tmp_path / split / f"{split}_guide.json.gz",
               {"episodes": episodes, "instruction_vocab": vocab})
    rxr.main(["--rxr-root", str(tmp_path), "--splits", "val_seen", "val_unseen"])
    for rel in ("val_seen_en/val_seen_en_guide.json.gz", "val_unseen_en/val_unseen_en_guide.json.gz"):
        with gzip.open(tmp_path / rel, "rt") as f:
            got = json.load(f)
        assert [e["episode_id"] for e in got["episodes"]] == [1, 4, 5]
        assert got["episodes"][0] == episodes[0]
        assert got["instruction_vocab"] == vocab


def test_output_paths_follow_the_config_data_paths():
    for name, split in (("vln_rxr_en.yaml", "val_seen"), ("vln_rxr_en.yaml", "val_unseen")):
        text = (REPO / "configs" / "navida" / name).read_text()
        pattern = re.search(r"data_path:\s*data/datasets/rxr/(\S+)", text).group(1)
        expected = Path("root") / pattern.format(split=split, role="guide")
        assert rxr.english_path(Path("root"), split, "guide") == expected

