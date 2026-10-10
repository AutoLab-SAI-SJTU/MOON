import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("run_harness_eval", REPO / "scripts" / "run_harness_eval.py")
rhe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rhe)

from moon.harness import HarnessConfig, HarnessLoop  # noqa: E402
from moon.harness.judge import NoJudge  # noqa: E402


class FakeEnv:
    def __init__(self, n_steps):
        self.left = n_steps
        self.current_episode = None
        self.dtg = 10.0

    @property
    def episode_over(self):
        return self.left <= 0

    def reset(self):
        return {"rgb": None}

    def step(self, action):
        self.left -= 1
        self.dtg -= 0.25
        return {"rgb": None}

    def get_metrics(self):
        return {"distance_to_goal": self.dtg, "success": 0.0, "spl": 0.0}


class FakeDriver:
    instruction, judge_rgb_list, recent_actions = "go", [], []

    def reset(self, instruction): pass
    def observe(self, rgb): pass
    def note_action(self, name): pass
    def propose_chunk(self, state): return [1, 1]


def test_run_episode_reports_simulator_time_next_to_the_loop_timings(monkeypatch):
    monkeypatch.setattr(rhe, "reset_to", lambda env, ep: env.reset())
    env = FakeEnv(6)
    env.current_episode = SimpleNamespace(instruction=SimpleNamespace(instruction_text="go"))
    base = FakeDriver()
    loop = HarnessLoop(HarnessConfig(drift_threshold=float("inf"), risk_window=1), base, NoJudge())
    _, st = rhe.run_episode(env, loop, base, "1", env.current_episode)
    assert st.step == 6
    assert len(st.policy_seconds) == 3 == len(st.judge_seconds)
    assert st.sim_seconds >= 0 and st.sim_steps == 6


def test_run_config_pins_the_decode_mode(tmp_path):
    args = SimpleNamespace(arm="c1", split="val_unseen", seed_base=41, decode_mode="official")
    cfg = rhe.write_run_config(tmp_path, args, {})
    assert cfg["decode_mode"] == "official"
    assert "decode_mode" in rhe.PINNED_KEYS
    args.decode_mode = "greedy"
    with pytest.raises(SystemExit):
        rhe.write_run_config(tmp_path, args, {})


def test_the_events_payload_carries_what_an_offline_replay_needs(monkeypatch):
    monkeypatch.setattr(rhe, "reset_to", lambda env, ep: env.reset())
    env = FakeEnv(6)
    env.current_episode = SimpleNamespace(instruction=SimpleNamespace(instruction_text="go"))
    base = FakeDriver()
    loop = HarnessLoop(HarnessConfig(drift_threshold=float("inf"), risk_window=1), base, NoJudge())
    _, st = rhe.run_episode(env, loop, base, "1", env.current_episode)
    payload = rhe.events_payload("1", "a0", st, {"arm": "a0"}, 1.5)
    assert payload["executed_atoms"] == [1, 1, 1, 1, 1, 1] == st.executed_atoms
    assert payload["decision_steps"] == [0, 2, 4] == st.decision_steps
    assert len(payload["executed_atoms"]) == payload["steps"] == st.step
    assert payload["timing"]["sim_steps"] == 6 and payload["config"] == {"arm": "a0"}
    json.dumps(payload)      # the runner writes it straight to disk


def test_run_config_pins_the_fast_deployment_switches(tmp_path):
    args = SimpleNamespace(arm="c1", split="val_unseen", seed_base=41, decode_mode="greedy",
                           merge_restart=True, stop_judge_after_budget=True)
    cfg = rhe.write_run_config(tmp_path, args, {})
    assert cfg["merge_restart"] is True and cfg["stop_judge_after_budget"] is True
    args.merge_restart = False
    with pytest.raises(SystemExit):
        rhe.write_run_config(tmp_path, args, {})


def _args(*extra):
    return rhe.parse_args(["--arm", "c1", "--model", "m", "--judge-lora", "j",
                           "--out-dir", "o", *extra])


def test_defaults_are_the_reported_configuration():
    args = _args()
    assert args.split == "val_unseen" and args.dataset == "r2r"
    assert args.decode_mode == "greedy" and args.resolution_ratio == 1.0
    assert args.risk_window == 3 and args.judge_every == 1 and args.max_recoveries == 1
    assert args.seed_mode == "paired" and args.seed_base == 41
    assert rhe.resolve_threshold(args) == 2.54
    assert rhe.resolve_threshold(_args("--dataset", "rxr")) == 3.33


def test_alpha_selects_the_calibrated_threshold_and_an_explicit_one_wins():
    assert rhe.resolve_threshold(_args("--alpha", "0.05")) == 3.83
    assert rhe.resolve_threshold(_args("--dataset", "rxr", "--alpha", "0.025")) == 4.37
    assert rhe.resolve_threshold(_args("--alpha", "0.05", "--drift-threshold", "1.7")) == 1.7
    with pytest.raises(SystemExit):
        _args("--alpha", "0.2")
    with pytest.raises(SystemExit):
        rhe.resolve_threshold(_args("--dataset", "scalevln", "--split", "train"))
    with pytest.raises(SystemExit):
        _args("--split", "train")


def test_non_judge_arms_carry_no_threshold():
    a0 = rhe.parse_args(["--arm", "a0", "--model", "m", "--out-dir", "o"])
    assert rhe.resolve_threshold(a0) is None
    periodic = rhe.parse_args(["--arm", "c1", "--model", "m", "--out-dir", "o",
                               "--trigger", "periodic"])
    assert rhe.resolve_threshold(periodic) is None


def test_decode_and_ratio_are_explicit_and_validated():
    args = _args("--decode", "official", "--resolution-ratio", "0.5")
    assert args.decode_mode == "official" and args.resolution_ratio == 0.5
    with pytest.raises(SystemExit):
        _args("--decode", "beam")


def test_run_config_records_paths_as_given(tmp_path):
    args = _args("--episode-list", "ids/a10.json", "--restart-lora", "runs/iter15/adapter")
    cfg = rhe.write_run_config(tmp_path, args, {"drift_threshold": 2.54})
    written = json.loads((tmp_path / "run_config.json").read_text())
    assert written["model"] == "m" and written["out_dir"] == "o"
    assert written["episode_list"] == "ids/a10.json"
    assert written["restart_lora"] == "runs/iter15/adapter"
    assert cfg["drift_threshold"] == 2.54 and cfg["alpha"] is None

