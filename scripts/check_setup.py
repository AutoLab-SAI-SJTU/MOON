"""Inspect local checkpoints and simulator prerequisites without loading models."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path


def check_checkpoint(path, kind):
    root = Path(path)
    config_name = "config.json" if kind == "base" else "adapter_config.json"
    config = root / config_name
    errors = []
    if not config.is_file():
        return [f"{kind}: missing {config}"]
    try:
        metadata = json.loads(config.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        return [f"{kind}: invalid {config_name}: {exc}"]
    if kind == "base" and metadata.get("model_type") not in ("qwen3_vl", "qwen2_5_vl"):
        errors.append(f"base: unsupported model_type {metadata.get('model_type')!r}")
    weights = list(root.glob("*.safetensors")) + list(root.glob("*.bin"))
    if not weights:
        errors.append(f"{kind}: no model/adapter weight file found in {root}")
    for weight in weights:
        with weight.open("rb") as stream:
            if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                errors.append(f"{kind}: {weight.name} is a Git LFS pointer, not downloaded weights")
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--judge-lora", required=True, type=Path)
    ap.add_argument("--recovery-lora", required=True, type=Path)
    ap.add_argument("--navida-root", type=Path,
                    help="optional: also inspect Habitat benchmark prerequisites")
    args = ap.parse_args(argv)
    errors = []
    for kind, path in (("base", args.model), ("judge", args.judge_lora),
                       ("recovery", args.recovery_lora)):
        errors.extend(check_checkpoint(path, kind))
    modules = ["torch", "transformers", "peft", "PIL", "numpy"]
    if args.navida_root:
        modules += ["habitat", "habitat_sim", "habitat_baselines", "dtw", "fastdtw"]
        for relative in ["config", "habitat_extensions", "data"]:
            if not (args.navida_root / relative).is_dir():
                errors.append(f"NAVIDA_ROOT: missing {relative}/")
    for module in modules:
        if importlib.util.find_spec(module) is None:
            errors.append(f"dependency: {module} is not installed")
    gpu_available = False
    if importlib.util.find_spec("torch") is not None:
        import torch
        gpu_available = torch.cuda.is_available()
    report = {"checkpoint_errors": errors, "cuda_available": gpu_available,
              "model_loaded": False,
              "note": "Metadata checks do not replace a GPU inference smoke test."}
    print(json.dumps(report, indent=2))
    return 1 if errors or not gpu_available else 0


if __name__ == "__main__":
    sys.exit(main())
