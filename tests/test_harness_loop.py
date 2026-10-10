import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moon.harness import HarnessConfig, HarnessLoop  # noqa: E402
from moon.harness.judge import ScheduledJudge  # noqa: E402


class FakeBase:
    def __init__(self):
        self.restarts = []

    def propose_chunk(self, state):
        return [1, 1]

    def restart_fresh(self, instruction, step):
        self.restarts.append(step)


class FakeRestart:
    def __init__(self):
        self.proposals = 0

    def propose(self, state):
        self.proposals += 1
        return {"text": "go", "verdict": "identity"}


def run(loop, base, n_decisions):
    st = loop.reset()
    for _ in range(n_decisions):
        atoms = loop.decide(st)
        loop.record_executed(st, atoms)
    return st


def scheduled_loop(trigger_steps, **cfg):
    judge = ScheduledJudge(trigger_steps)
    base, restart = FakeBase(), FakeRestart()
    loop = HarnessLoop(HarnessConfig(drift_threshold=0.0, risk_window=1, max_recoveries=1, **cfg),
                       base, judge, restart)
    return loop, base, restart, judge


def test_scheduled_judge_fires_once_at_the_first_decision_on_or_after_its_step():
    loop, base, restart, judge = scheduled_loop({"7": 5})
    judge.begin("7")
    st = run(loop, base, 6)
    replans = [e for e in st.events if e["kind"] == "replan"]
    assert [e["step"] for e in replans] == [6]
    assert base.restarts == [6]
    assert restart.proposals == 1


def test_scheduled_judge_stays_quiet_for_an_unscheduled_episode():
    loop, base, restart, judge = scheduled_loop({"7": 5})
    judge.begin("8")
    st = run(loop, base, 6)
    assert not [e for e in st.events if e["kind"] == "replan"]
    assert base.restarts == []


def test_scheduled_judge_resets_between_episodes():
    loop, base, restart, judge = scheduled_loop({"7": 0})
    judge.begin("7")
    run(loop, base, 3)
    judge.begin("7")
    run(loop, base, 3)
    assert base.restarts == [0, 0]


def test_keeping_history_mounts_the_adapter_without_restarting_the_base():
    loop, base, restart, judge = scheduled_loop({"7": 0}, restart_clears_history=False)
    judge.begin("7")
    st = run(loop, base, 3)
    assert [e["kind"] for e in st.events].count("replan") == 1
    assert restart.proposals == 1
    assert base.restarts == []


def cheap_loop(judge, **cfg):
    base, restart = FakeBase(), FakeRestart()
    loop = HarnessLoop(HarnessConfig(drift_threshold=0.0, risk_window=1, max_recoveries=1, **cfg),
                       base, judge, restart)
    return loop, base, restart


def test_periodic_trigger_fires_once_the_episode_reaches_its_interval():
    from moon.harness.judge import PeriodicJudge
    loop, base, _ = cheap_loop(PeriodicJudge(every_steps=5))
    st = run(loop, base, 6)
    assert [e["step"] for e in st.events if e["kind"] == "replan"] == [6]
    assert base.restarts == [6]      # max_recoveries caps it at one


def test_spin_trigger_fires_only_after_a_window_without_a_forward_move():
    from moon.harness.judge import SpinJudge
    judge = SpinJudge(window=3)
    loop, base, _ = cheap_loop(judge)
    st = loop.reset()
    st.judge_ctx = {"recent": ["MOVE_FORWARD", "TURN_LEFT", "TURN_LEFT"]}
    loop.decide(st)
    assert not [e for e in st.events if e["kind"] == "replan"]
    st.judge_ctx = {"recent": ["TURN_LEFT", "TURN_RIGHT", "TURN_LEFT"]}
    loop.decide(st)
    assert [e["kind"] for e in st.events].count("replan") == 1


def test_oracle_trigger_fires_beyond_the_route_margin_and_never_without_a_distance():
    from moon.harness.judge import OracleJudge
    loop, base, _ = cheap_loop(OracleJudge(threshold=2.0))
    st = loop.reset()
    for d in (0.4, 1.9, None):
        st.judge_ctx = {"route_dist": d}
        loop.decide(st)
    assert not [e for e in st.events if e["kind"] == "replan"]
    st.judge_ctx = {"route_dist": 2.4}
    loop.decide(st)
    assert [e["kind"] for e in st.events].count("replan") == 1


def test_more_recoveries_allow_a_second_restart_after_the_refractory_period():
    from moon.harness.judge import PeriodicJudge
    for cap, want in ((1, 1), (2, 2), (3, 3)):
        judge = PeriodicJudge(every_steps=1)      # every decision would fire
        base, restart = FakeBase(), FakeRestart()
        loop = HarnessLoop(HarnessConfig(drift_threshold=0.0, risk_window=1,
                                         max_recoveries=cap, refractory_chunks=4),
                           base, judge, restart)
        st = run(loop, base, 30)
        assert [e["kind"] for e in st.events].count("replan") == want
        assert len(base.restarts) == want


def test_loop_times_every_policy_call_and_only_the_judge_calls_it_makes():
    from moon.harness.judge import NoJudge
    base = FakeBase()
    loop = HarnessLoop(HarnessConfig(drift_threshold=float("inf"), risk_window=3, judge_every=2),
                       base, NoJudge(), FakeRestart())
    st = run(loop, base, 5)
    assert len(st.policy_seconds) == 5
    assert len(st.judge_seconds) == 3 == len(st.mid_scores)
    assert all(t >= 0 for t in st.policy_seconds + st.judge_seconds)


def test_a_restart_times_the_second_policy_call_too():
    loop, base, restart, judge = scheduled_loop({"7": 0})
    judge.begin("7")
    st = run(loop, base, 3)
    assert len(st.policy_seconds) == 3 + 1


class CountingJudge:
    """Scores every decision as lost, so the first judged decision alarms."""

    def __init__(self):
        self.calls = 0

    def score_mid(self, state, proposed_atoms):
        self.calls += 1
        return {"OK": 0.0, "lost": 1.0, "amb": 0.0}


def test_judge_stops_once_the_recovery_budget_is_spent_without_changing_the_actions():
    runs = {}
    for after in (True, False):
        judge, base, restart = CountingJudge(), FakeBase(), FakeRestart()
        loop = HarnessLoop(HarnessConfig(drift_threshold=0.0, risk_window=1, max_recoveries=1,
                                         judge_after_budget=after), base, judge, restart)
        st = run(loop, base, 10)
        runs[after] = (judge.calls, st.executed_atoms, [e["kind"] for e in st.events], base.restarts)
    assert runs[True][0] == 10
    assert runs[False][0] == 1
    assert runs[True][1:] == runs[False][1:]


def test_judge_keeps_scoring_without_a_restart_even_when_told_to_stop_after_the_budget():
    judge, base = CountingJudge(), FakeBase()
    loop = HarnessLoop(HarnessConfig(drift_threshold=0.0, risk_window=1, max_recoveries=1,
                                     judge_after_budget=False), base, judge, None)
    run(loop, base, 5)
    assert judge.calls == 5


def test_library_defaults_are_the_deployed_monitor():
    cfg = HarnessConfig()
    assert (cfg.drift_threshold, cfg.risk_window, cfg.judge_every, cfg.max_recoveries) == \
        (2.54, 3, 1, 1)


def test_calibrated_thresholds_are_looked_up_by_dataset_and_alpha():
    import pytest
    from moon.monitor import calibrated_tau
    assert calibrated_tau("r2r") == 2.54 and calibrated_tau("rxr") == 3.33
    assert calibrated_tau("r2r", 0.025) == 4.67 and calibrated_tau("rxr", 0.125) == 2.71
    with pytest.raises(ValueError):
        calibrated_tau("r2r", 0.2)
    with pytest.raises(ValueError):
        calibrated_tau("scalevln")


def test_navida_root_must_be_set_and_the_extra_configs_come_from_the_repo(tmp_path, monkeypatch):
    import pytest
    from moon.harness import env
    monkeypatch.setenv("NAVIDA_ROOT", "unset-below")   # registers the variable for restore
    monkeypatch.delenv("NAVIDA_ROOT")
    with pytest.raises(RuntimeError, match="NAVIDA_ROOT"):
        env.navida_root()
    with pytest.raises(RuntimeError, match="NAVIDA_ROOT"):
        env.config_path("val_unseen", "r2r")
    for split in ("val_seen", "val_unseen"):
        assert env.config_path(split, "rxr").parent == env.REPO_CONFIG_DIR
        assert env.config_path(split, "rxr").exists()
    with pytest.raises(ValueError, match="evaluation"):
        env.config_path("train", "rxr")
    (tmp_path / "config").mkdir()
    env.set_navida_root(tmp_path)
    assert env.config_path("val_unseen", "r2r") == tmp_path.resolve() / "config" / "vln_r2r.yaml"
    with pytest.raises(ValueError):
        env.config_path("train", "ovon")


def test_the_frame_contract_must_be_selected_before_use(monkeypatch):
    import pytest
    from moon import navida_io
    monkeypatch.setattr(navida_io, "_active_model_type", None)
    with pytest.raises(RuntimeError, match="configure"):
        navida_io.img_size()
    navida_io.configure("qwen3_vl")
    assert navida_io.img_size() == (352, 288)

