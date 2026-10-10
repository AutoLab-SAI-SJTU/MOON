"""Harness main loop: monitor the frozen policy and restart it when it drifts.

The base policy proposes an action chunk at every decision point. The judge
shadow-scores the chunk, a rolling window aggregates the scores, and a score
above the calibrated threshold restarts the policy on the current frame with
its history cleared. A per-episode cap and a refractory period bound the
interventions. A proposed STOP ends the episode: the stop decision belongs to
the policy, never to the monitor.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from moon.harness.chunking import STOP
from moon.monitor import RollingRisk, calibrated_tau, lost_margin


@dataclass
class HarnessConfig:
    drift_threshold: float = calibrated_tau("r2r")   # tau on the rolled lost-margin scale
    risk_window: int = 3
    risk_mode: str = "mean"
    judge_every: int = 1             # one judge forward per N decision points
    refractory_chunks: int = 4       # decision points without a trigger after a restart
    max_recoveries: int = 1          # restarts per episode
    restart_clears_history: bool = True   # False keeps the frame history across a restart
    judge_after_budget: bool = True  # False: no judge calls once no restart is left

    def __post_init__(self):
        if self.risk_window < 1 or self.judge_every < 1:
            raise ValueError("invalid HarnessConfig")


class BasePolicy(Protocol):
    def propose_chunk(self, state: "EpisodeState") -> List[int]: ...
    def restart_fresh(self, instruction: str, step: int) -> None: ...


class Judge(Protocol):
    def score_mid(self, state: "EpisodeState", proposed_atoms: List[int]) -> Dict[str, float]: ...


class Restart(Protocol):
    def propose(self, state: "EpisodeState") -> Dict: ...


@dataclass
class EpisodeState:
    """Observable episode state; the loop never touches simulator ground truth."""
    step: int = 0
    executed_atoms: List[int] = field(default_factory=list)
    decision_steps: List[int] = field(default_factory=list)
    mid_scores: List[float] = field(default_factory=list)
    recoveries_used: int = 0
    refractory_left: int = 0
    judge_ctx: Dict = field(default_factory=dict)   # driver-owned observations for the judge
    events: List[Dict] = field(default_factory=list)
    policy_seconds: List[float] = field(default_factory=list)   # wall time per policy call
    judge_seconds: List[float] = field(default_factory=list)    # wall time per judge call
    sim_seconds: float = 0.0                                     # simulator time, set by the runner
    sim_steps: int = 0

    def log(self, kind: str, **kw) -> None:
        self.events.append({"step": self.step, "kind": kind, **kw})


class HarnessLoop:
    def __init__(self, cfg: HarnessConfig, base: BasePolicy, judge: Judge,
                 restart: Optional[Restart] = None):
        self.cfg = cfg
        self.base = base
        self.judge = judge
        self.restart = restart
        self.risk = RollingRisk(cfg.risk_window, cfg.risk_mode)

    def reset(self) -> EpisodeState:
        self.risk.reset()
        return EpisodeState()

    @staticmethod
    def _normalize(atoms: List[int]) -> List[int]:
        """Upstream executes a chunk through its first STOP and the episode ends there."""
        if STOP in atoms:
            return atoms[: atoms.index(STOP) + 1]
        return atoms

    def _propose(self, state: EpisodeState) -> List[int]:
        t0 = time.perf_counter()
        atoms = self.base.propose_chunk(state)
        state.policy_seconds.append(time.perf_counter() - t0)
        return atoms

    def decide(self, state: EpisodeState) -> List[int]:
        atoms = self._normalize(self._propose(state))
        state.decision_steps.append(state.step)
        if STOP in atoms:
            state.log("stop_accept")
            return atoms

        if (not self.cfg.judge_after_budget and self.restart is not None
                and state.recoveries_used >= self.cfg.max_recoveries):
            return atoms

        # Decision points the judge skips still spend the refractory period.
        if (len(state.decision_steps) - 1) % self.cfg.judge_every:
            if state.refractory_left > 0:
                state.refractory_left -= 1
            return atoms

        t0 = time.perf_counter()
        probs = self.judge.score_mid(state, atoms)
        state.judge_seconds.append(time.perf_counter() - t0)
        score = self.risk.update(lost_margin(probs))
        state.mid_scores.append(score)

        if state.refractory_left > 0:
            state.refractory_left -= 1
        elif (self.restart is not None and score > self.cfg.drift_threshold
              and state.recoveries_used < self.cfg.max_recoveries):
            return self._restart(state, score)
        return atoms

    def _restart(self, state: EpisodeState, score: float) -> List[int]:
        res = self.restart.propose(state)
        state.recoveries_used += 1
        state.refractory_left = self.cfg.refractory_chunks
        self.risk.reset()
        if self.cfg.restart_clears_history:
            self.base.restart_fresh(res["text"], state.step)
        state.log("replan", score=score, text=res["text"], verdict=res.get("verdict"))
        return self._normalize(self._propose(state))

    def record_executed(self, state: EpisodeState, executed: List[int]) -> None:
        """The runner reports what actually executed; only that enters the history."""
        state.executed_atoms.extend(executed)
        state.step += len(executed)

