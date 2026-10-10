import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_setup", Path(__file__).resolve().parents[1] / "scripts/check_setup.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def test_checkpoint_check_detects_lfs_pointer_instead_of_weights(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    assert "Git LFS pointer" in setup.check_checkpoint(tmp_path, "judge")[0]


def test_checkpoint_check_rejects_vanilla_unsupported_architecture(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}))
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    assert "unsupported model_type" in setup.check_checkpoint(tmp_path, "base")[0]
