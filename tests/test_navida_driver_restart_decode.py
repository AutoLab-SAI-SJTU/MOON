import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("transformers")

from moon.harness.navida_driver import NavidaDriver  # noqa: E402


class FakeHost:
    processor = None
    device = "cpu"


def test_sampled_decoding_applies_only_after_a_restart():
    d = NavidaDriver(FakeHost(), restart_gen_cfg="SAMPLE")
    d.reset("go")
    assert d.active_gen_cfg() is d.gen_cfg
    d.restart_fresh("go", step=12)
    assert d.active_gen_cfg() == "SAMPLE"
    d.reset("next episode")
    assert d.active_gen_cfg() is d.gen_cfg


def test_without_a_restart_config_the_restart_keeps_the_decoder():
    d = NavidaDriver(FakeHost())
    d.restart_fresh("go", step=3)
    assert d.restarted and d.active_gen_cfg() is d.gen_cfg


def test_decoding_is_greedy_unless_asked_and_rejects_unknown_modes():
    from moon.harness.navida_driver import decode_config
    assert NavidaDriver(FakeHost()).gen_cfg.do_sample is False
    official = NavidaDriver(FakeHost(), decode="official").gen_cfg
    assert official.do_sample is True and official.temperature == 0.2 and official.top_k == 50
    assert decode_config().do_sample is False
    with pytest.raises(ValueError):
        NavidaDriver(FakeHost(), decode="beam")


def test_the_policy_frame_uses_the_driver_resolution_ratio(monkeypatch):
    import numpy as np
    from moon.harness import navida_driver
    seen = []
    monkeypatch.setattr(navida_driver, "frame_from_sensor", lambda rgb, r: seen.append(r) or "p")
    monkeypatch.setattr(navida_driver, "judge_frame_from_sensor", lambda rgb: "j")
    NavidaDriver(FakeHost()).observe(np.zeros((4, 4, 3)))
    NavidaDriver(FakeHost(), resolution_ratio=0.5).observe(np.zeros((4, 4, 3)))
    assert seen == [1.0, 0.5]


def test_a_restart_keeps_the_executed_action_record():
    d = NavidaDriver(FakeHost())
    d.reset("go")
    d.history.append("f0")
    d.judge_history.append("j0")
    d.note_action("MOVE_FORWARD")
    d.restart_fresh("go", step=1)
    assert d.recent_actions == ["MOVE_FORWARD"] and d.history == ["f0"]

