"""End-to-end state and handover tests with explicit model-free components."""
import pytest

from moon import MoonAgent


class Driver:
    def reset(self, instruction):
        self.instruction = instruction
        self.judge_rgb_list = []
        self.recent_actions = []
    def observe(self, frame):
        self.judge_rgb_list.append(frame)
    def note_action(self, action):
        self.recent_actions.append(action)
    def propose_chunk(self, state):
        return [2] if self.policy_role == "restart" else [1]
    def restart_fresh(self, instruction, step):
        self.judge_rgb_list = self.judge_rgb_list[-1:]


class Judge:
    def score_mid(self, state, atoms):
        return {"lost": .99, "OK": .01, "amb": 0.}


def test_alarm_clears_history_preserves_current_frame_and_hands_over():
    agent = MoonAgent.from_components(Driver(), Judge())
    agent.begin_episode("turn left")
    agent.observe("old")
    agent.observe("current")
    assert agent.act() == [2]
    assert agent.driver.judge_rgb_list == ["current"]
    agent.record_action(2)
    agent.observe("next")
    assert agent.act() == [2]
    assert agent.state.recoveries_used == 1
    assert agent.state.executed_atoms == [2]
    assert agent.state.step == 1


def test_new_episode_restores_base_role_and_clears_episode_state():
    agent = MoonAgent.from_components(Driver(), Judge())
    agent.begin_episode("first")
    agent.observe("rgb")
    agent.act()
    agent.record_action(2)
    agent.begin_episode("second")
    assert agent.driver.policy_role is None
    assert agent.driver.judge_rgb_list == []
    assert agent.state.step == 0 and agent.state.recoveries_used == 0
    assert len(agent.loop.risk) == 0


def test_agent_requires_episode_and_current_observation():
    agent = MoonAgent.from_components(Driver(), Judge())
    with pytest.raises(RuntimeError, match="begin_episode"):
        agent.act()
    agent.begin_episode("go")
    with pytest.raises(RuntimeError, match="observe"):
        agent.act()
    with pytest.raises(ValueError, match="unknown action"):
        agent.record_action(99)


def test_checkpoint_threshold_override_changes_alarm_behavior():
    class BorderlineJudge:
        def score_mid(self, state, atoms):
            return {"lost": .95, "OK": .05, "amb": 0.}

    default = MoonAgent.from_components(Driver(), BorderlineJudge())
    available = MoonAgent.from_components(Driver(), BorderlineJudge(),
                                           drift_threshold=3.96)
    for agent in (default, available):
        agent.begin_episode("turn left")
        agent.observe("rgb")
    assert default.act() == [2]
    assert default.state.recoveries_used == 1
    assert available.act() == [1]
    assert available.state.recoveries_used == 0
