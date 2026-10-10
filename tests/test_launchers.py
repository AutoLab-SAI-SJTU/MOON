"""The shell launchers: syntax, required inputs, and the arguments they hand to the scripts.

GPU tools and the heavy scripts are replaced by stubs on PATH that record their arguments.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCH = REPO / "scripts" / "launch"
LAUNCHERS = sorted(LAUNCH.glob("*.sh")) + sorted(LAUNCH.glob("configs/*.env"))
RECORDED = ("scripts/run_harness_eval.py", "scripts/compare_arms.py")

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                               reason="launcher execution needs POSIX Bash; syntax is checked separately")


def stub_bin(tmp_path, gpus=8, mib=81920):
    """nvidia-smi, torchrun and a python that records the recorded scripts and runs everything else."""
    stub = Path(tempfile.mkdtemp(dir=tmp_path))
    bin_dir = stub / "bin"
    bin_dir.mkdir()
    log = stub / "calls.jsonl"
    (bin_dir / "nvidia-smi").write_text(
        "#!/bin/bash\n"
        f'if [[ "$*" == *memory.total* ]]; then echo {mib}; exit 0; fi\n'
        f"for i in $(seq 0 {gpus - 1}); do echo \"GPU $i: stub\"; done\n")
    record = (f"import json, sys; open({str(log)!r}, 'a').write(json.dumps("
              "{'cmd': sys.argv[1], 'args': sys.argv[2:], 'cuda': __import__('os').environ.get('CUDA_VISIBLE_DEVICES')})"
              " + '\\n')")
    (bin_dir / "torchrun").write_text(f"#!/bin/bash\n{sys.executable} -c \"{record}\" torchrun \"$@\"\n")
    cases = "|".join(RECORDED)
    (bin_dir / "python").write_text(
        "#!/bin/bash\n"
        f'case "$1" in {cases}) exec {sys.executable} -c "{record}" "$@" ;; esac\n'
        f'exec {sys.executable} "$@"\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    return bin_dir, log


def run(script, tmp_path, env=None, args=(), gpus=8, mib=81920):
    bin_dir, log = stub_bin(tmp_path, gpus, mib)
    base = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(["bash", str(LAUNCH / script), *args], env={**base, **(env or {})},
                          capture_output=True, text=True, timeout=120)
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return proc, calls


def flag(args, name):
    return args[args.index(name) + 1]


@pytest.mark.parametrize("script", LAUNCHERS, ids=lambda p: p.name)
def test_launchers_parse(script):
    subprocess.run(["bash", "-n", str(script)], check=True)



def test_eval_rejects_unknown_arms_and_missing_adapters(tmp_path):
    proc, _ = run("eval.sh", tmp_path, args=["bogus"])
    assert proc.returncode != 0 and "unknown arm bogus" in proc.stderr
    env = {"NAVIDA_ROOT": str(tmp_path), "BASE_MODEL": "m", "JUDGE_LORA": "j"}
    proc, calls = run("eval.sh", tmp_path, env=env, args=["full"])
    assert proc.returncode != 0 and "set RECOVERY_LORA" in proc.stderr
    assert not calls


def eval_env(tmp_path, **extra):
    return {"NAVIDA_ROOT": str(tmp_path), "BASE_MODEL": "ckpt/base", "JUDGE_LORA": "ckpt/judge",
            "RECOVERY_LORA": "ckpt/recovery", "OUT_ROOT": str(tmp_path / "eval"), **extra}


def harness_calls(calls):
    return [c["args"] for c in calls if c["cmd"] == "scripts/run_harness_eval.py"]


def test_eval_base_shards_the_frozen_policy_over_the_gpus(tmp_path):
    proc, calls = run("eval.sh", tmp_path, env=eval_env(tmp_path), args=["base"], gpus=2)
    assert proc.returncode == 0, proc.stderr
    runs = harness_calls(calls)
    assert len(runs) == 6
    assert sorted(int(flag(a, "--shard")) for a in runs) == list(range(6))
    assert {flag(a, "--num-shards") for a in runs} == {"6"}
    assert all(flag(a, "--arm") == "a0" and flag(a, "--split") == "val_unseen" for a in runs)
    assert sorted(c["cuda"] for c in calls if c["cmd"] == "scripts/run_harness_eval.py") == ["0", "0", "0",
                                                                                           "1", "1", "1"]
    assert json.loads((tmp_path / "eval" / "episodes.json").read_text()) == {"episode_ids": []}


def test_eval_full_replays_the_alarm_set_at_the_deployed_alpha(tmp_path):
    for dataset, alpha in (("r2r", "0.1"), ("rxr", "0.075")):
        out = tmp_path / dataset
        (out / "eval").mkdir(parents=True)
        alarms = out / "eval" / f"alarms_a{alpha}.json"
        alarms.write_text(json.dumps({"episode_ids": ["1"]}))
        proc, calls = run("eval.sh", out, env=eval_env(out, DATASET=dataset), args=["full", "--max-recoveries", "2"],
                          gpus=1)
        assert proc.returncode == 0, proc.stderr
        runs = harness_calls(calls)
        assert len(runs) == 3
        args = runs[0]
        assert flag(args, "--arm") == "c1" and flag(args, "--dataset") == dataset
        assert flag(args, "--alpha") == alpha
        assert flag(args, "--judge-lora") == "ckpt/judge" and flag(args, "--restart-lora") == "ckpt/recovery"
        assert flag(args, "--episode-list") == str(alarms)
        assert flag(args, "--max-recoveries") == "2"
        assert flag(args, "--out-dir") == str(out / "eval" / f"full_a{alpha}")
        compare = [c["args"] for c in calls if c["cmd"] == "scripts/compare_arms.py"][0]
        assert flag(compare, "--alarm-ids") == str(alarms)
        assert flag(compare, "--base-log") == str(out / "eval" / "base" / "log")


def test_eval_full_needs_the_alarm_set(tmp_path):
    proc, calls = run("eval.sh", tmp_path, env=eval_env(tmp_path), args=["full"])
    assert proc.returncode != 0 and "alarms_a0.1.json not found" in proc.stderr
    assert not calls


def test_eval_pass8_draws_seeded_sampled_restarts_without_the_adapter(tmp_path):
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "alarms_a0.1.json").write_text(json.dumps({"episode_ids": ["1"]}))
    proc, calls = run("eval.sh", tmp_path, env=eval_env(tmp_path, DRAWS="2"), args=["pass8"], gpus=1)
    assert proc.returncode == 0, proc.stderr
    runs = harness_calls(calls)
    assert len(runs) == 6
    assert all(flag(a, "--restart-decode") == "sample" and "--restart-lora" not in a for a in runs)
    assert sorted({flag(a, "--seed-base") for a in runs}) == ["1041", "41"]


def test_eval_query_interval_uses_the_matched_threshold(tmp_path):
    proc, calls = run("eval.sh", tmp_path, env=eval_env(tmp_path), args=["every4"], gpus=1)
    runs = harness_calls(calls)
    assert runs and flag(runs[0], "--judge-every") == "4" and flag(runs[0], "--drift-threshold") == "1.00"
    proc, calls = run("eval.sh", tmp_path,
                      env=eval_env(tmp_path), args=["every1"], gpus=1)
    runs = harness_calls(calls)
    assert runs and flag(runs[0], "--judge-every") == "1" and "--drift-threshold" not in runs[0]
    proc, calls = run("eval.sh", tmp_path,
                      env=eval_env(tmp_path), args=["every3"], gpus=1)
    assert proc.returncode != 0 and "no threshold for a query interval of 3" in proc.stderr

