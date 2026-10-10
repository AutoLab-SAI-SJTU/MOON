"""The judge: prompt contract and the client that scores decision points.

The deployed prompt matches the released checkpoint contract. The answer is one of three single-token labels; the loop turns
the label distribution into a score.
"""
from typing import Dict, List, Sequence

from PIL import Image

from moon.harness.chunking import ACTION_NAMES
from moon.harness.history import NUM_HISTORY, sample_history_indices
from moon.harness.settings import ORACLE_ROUTE_MARGIN

SYSTEM_PROMPT = "You are a helpful assistant."
LABELS = ("OK", "lost", "amb")
RECENT_WINDOW = 12

# Label distributions of the model-free triggers: below and above any threshold.
QUIET = {"OK": 0.98, "lost": 0.01, "amb": 0.01}
ALARM = {"OK": 0.01, "lost": 0.98, "amb": 0.01}

JUDGE_HEADER = (
    "You are a trajectory health critic for a vision-language navigation agent. "
    "Given the recent visual history, the instruction, and recent actions, "
    "judge whether the agent is on-track or off-track."
)


def build_judge_prompt(n_images: int, instruction: str,
                       recent_actions: Sequence[str],
                       proposed_chunk: Sequence[str]) -> str:
    """One string with <image> placeholders: n_images - 1 history frames, then
    the current frame, the instruction, the last executed actions and the
    chunk the base policy proposes next."""
    if not proposed_chunk:
        raise ValueError("the judge scores a proposed chunk; it cannot be empty")
    his = "<image>" * (n_images - 1)
    cur = "<image>"
    recent = ", ".join(recent_actions[-RECENT_WINDOW:]) if recent_actions else "(none)"
    chunk_str = ", ".join(proposed_chunk)
    question = (
        f"The base policy proposes the next action chunk: [{chunk_str}]. "
        "Is the agent off-track or losing progress? Answer one word from {OK, lost, amb}, "
        "where 'OK' = making progress, 'lost' = off-track or stagnating, "
        "'amb' = uncertain."
    )
    return (
        JUDGE_HEADER + "\n"
        + f" This is your historical observation:{his}\n"
        + f" This is your current observation:{cur}\n"
        + f" Your task is to {instruction.strip()}\n"
        + f" Recent executed actions: [{recent}]\n"
        + f" {question}"
    )


def judge_messages(prompt: str, images: Sequence) -> List[Dict]:
    """Splice the prompt and frames into the inference chat turns."""
    parts = prompt.split("<image>")
    content: List[Dict] = []
    for j, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if j < len(images):
            content.append({"type": "image", "image": images[j]})
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}]
    return msgs


class NavidaJudge:
    """Scores the proposed chunk on the shared host with the judge adapter."""

    def __init__(self, host, adapter: str = "judge", n_history: int = NUM_HISTORY):
        self.host = host
        self.adapter = adapter
        self.n_history = n_history

    def _frames(self, rgb_list: List[Image.Image]) -> List[Image.Image]:
        idx = sample_history_indices(len(rgb_list) - 1, self.n_history)
        return [rgb_list[i] for i in idx]

    def score_mid(self, state, proposed_atoms: List[int]) -> Dict[str, float]:
        ctx = state.judge_ctx
        frames = self._frames(ctx["frames"])
        prompt = build_judge_prompt(len(frames), ctx["instruction"], ctx["recent"],
                                    [ACTION_NAMES[a] for a in proposed_atoms])
        text = self.host.processor.apply_chat_template(
            judge_messages(prompt, frames), tokenize=False, add_generation_prompt=True)
        return self.host.label_probs(text, frames, self.adapter)


class NoJudge:
    """Constant scores; with an infinite threshold the loop never restarts."""

    def score_mid(self, state, proposed_atoms: List[int]) -> Dict[str, float]:
        return dict(QUIET)


class PeriodicJudge:
    """Fires once the episode has executed ``every_steps`` steps; needs no model."""

    def __init__(self, every_steps: int):
        if every_steps < 1:
            raise ValueError(f"every_steps must be positive, got {every_steps}")
        self.every_steps = int(every_steps)

    def begin(self, episode_id) -> None:
        pass

    def score_mid(self, state, proposed_atoms: List[int]) -> Dict[str, float]:
        return dict(ALARM if state.step >= self.every_steps else QUIET)


class SpinJudge:
    """Fires when the last ``window`` executed actions contain no forward move:
    a stuck/oscillation detector over the driver's own action record."""

    def __init__(self, window: int):
        if window < 1:
            raise ValueError(f"window must be positive, got {window}")
        self.window = int(window)

    def begin(self, episode_id) -> None:
        pass

    def score_mid(self, state, proposed_atoms: List[int]) -> Dict[str, float]:
        recent = state.judge_ctx.get("recent") or []
        stuck = (len(recent) >= self.window
                 and "MOVE_FORWARD" not in recent[-self.window:])
        return dict(ALARM if stuck else QUIET)


class OracleJudge:
    """Fires at the first decision point whose geodesic distance to the
    reference path exceeds ``threshold``; reads ground truth unavailable at
    deployment."""

    def __init__(self, threshold: float = ORACLE_ROUTE_MARGIN):
        self.threshold = float(threshold)

    def begin(self, episode_id) -> None:
        pass

    def score_mid(self, state, proposed_atoms: List[int]) -> Dict[str, float]:
        d = state.judge_ctx.get("route_dist")
        off = d is not None and d > self.threshold
        return dict(ALARM if off else QUIET)


class ScheduledJudge:
    """Fires once per episode, at the first decision point on or after the
    scheduled step, and scores like NoJudge everywhere else. Run it with
    risk_window 1: a rolling mean would dilute the single spike."""

    def __init__(self, trigger_steps: Dict[str, int]):
        self.trigger_steps = {str(k): int(v) for k, v in trigger_steps.items()}
        self.episode = None
        self.fired = False

    def begin(self, episode_id) -> None:
        self.episode = str(episode_id)
        self.fired = False

    def score_mid(self, state, proposed_atoms: List[int]) -> Dict[str, float]:
        at = self.trigger_steps.get(self.episode)
        if at is not None and not self.fired and state.step >= at:
            self.fired = True
            return dict(ALARM)
        return dict(QUIET)

