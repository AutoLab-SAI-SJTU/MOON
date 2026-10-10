"""Simulator-independent MOON inference on a stream of RGB observations."""
from typing import Optional

from moon.harness import HarnessConfig, HarnessLoop
from moon.harness.chunking import ACTION_NAMES
from moon.harness.judge import NavidaJudge
from moon.harness.restart import IdentityRestart
from moon.monitor import calibrated_tau


class MoonAgent:
    """Wrap the released backbone and adapters without loading Habitat.

    Call begin_episode once, observe after every environment step, act once per
    decision point, and record_action for each action the environment executes.
    A returned chunk may end early; only executed actions enter the state.
    """

    def __init__(self, model: str, judge_lora: str, recovery_lora: str,
                 dataset: str = "r2r", device: str = "cuda:0",
                 alpha: Optional[float] = None, merge_recovery: bool = False,
                 drift_threshold: Optional[float] = None):
        from moon.harness.host import NavidaHost
        from moon.harness.navida_driver import NavidaDriver

        host = NavidaHost(model, device=device,
                          adapters={"judge": judge_lora, "restart": recovery_lora},
                          merge_adapters=("restart",) if merge_recovery else ())
        self._initialize(NavidaDriver(host, decode="greedy"), NavidaJudge(host),
                         dataset, alpha, drift_threshold)
        self.host = host

    @classmethod
    def from_components(cls, driver, judge, dataset: str = "r2r",
                        alpha: Optional[float] = None,
                        drift_threshold: Optional[float] = None):
        """Use a compatible observation driver and judge, e.g. for smoke tests."""
        agent = cls.__new__(cls)
        agent._initialize(driver, judge, dataset, alpha, drift_threshold)
        agent.host = None
        return agent

    def _initialize(self, driver, judge, dataset, alpha, drift_threshold):
        self.driver = driver
        threshold = (calibrated_tau(dataset, alpha) if drift_threshold is None
                     else float(drift_threshold))
        self.loop = HarnessLoop(
            HarnessConfig(drift_threshold=threshold,
                          max_recoveries=1, judge_after_budget=False), driver, judge)
        self.state = None

    def begin_episode(self, instruction: str) -> None:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        self.driver.policy_role = None
        self.driver.reset(instruction)
        self.state = self.loop.reset()
        self.loop.restart = IdentityRestart(instruction, self.driver)

    def _require_episode(self):
        if self.state is None:
            raise RuntimeError("call begin_episode before inference")

    def observe(self, rgb) -> None:
        """Add the current H x W x 3 RGB frame, with no privileged route state."""
        self._require_episode()
        self.driver.observe(rgb)

    def act(self):
        """Return an action chunk: 0 STOP, 1 FORWARD, 2 LEFT, 3 RIGHT."""
        self._require_episode()
        frames = self.driver.judge_rgb_list
        if not frames:
            raise RuntimeError("observe the current RGB frame before act")
        self.state.judge_ctx = {
            "frames": frames,
            "recent": self.driver.recent_actions,
            "instruction": self.driver.instruction,
        }
        return self.loop.decide(self.state)

    def record_action(self, action: int) -> None:
        """Record one executed atomic action, not merely a proposed action."""
        self._require_episode()
        if action not in ACTION_NAMES:
            raise ValueError(f"unknown action {action!r}; expected 0, 1, 2 or 3")
        self.driver.note_action(ACTION_NAMES[action])
        self.loop.record_executed(self.state, [action])
