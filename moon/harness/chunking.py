"""NaVIDA action-chunk text, parsed exactly as the upstream evaluator does.

A chunk splits on ", "; only the first two sub-actions are kept, and each
expands to min(3, round(amount / step)) atomic actions.
"""
import math
import re
from typing import List

STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = {STOP: "STOP", FORWARD: "MOVE_FORWARD", LEFT: "TURN_LEFT", RIGHT: "TURN_RIGHT"}

# "stop" is a single token of the Qwen2.5-VL / Qwen3-VL tokenizer; NavidaHost checks it on load.
STOP_TOKEN_ID = 9495

FORWARD_CM = 25
TURN_DEG = 15
MAX_ATOMS_PER_SUB = 3
MAX_SUBS = 2

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>")
_NUM_RE = re.compile(r"-?\d+")
_STEP = {FORWARD: FORWARD_CM, LEFT: TURN_DEG, RIGHT: TURN_DEG}


def _parse_sub(sub: str):
    """Upstream extract_result: strip <answer> tags, keyword order
    stop > forward > left > right, first integer anywhere in the fragment."""
    m = _ANSWER_RE.search(sub)
    s = (m.group(1) if m else sub).strip().lower()
    if "stop" in s:
        return STOP, None
    for a, kw in ((FORWARD, "forward"), (LEFT, "left"), (RIGHT, "right")):
        if kw in s:
            n = _NUM_RE.search(s)
            return a, float(n.group()) if n else float(_STEP[a])
    return None, None


def _expand(sub: str) -> List[int]:
    a, n = _parse_sub(sub)
    if a == STOP:
        return [STOP]
    if a is None:
        return []
    scaled = n / _STEP[a]
    # A degenerate digit run parses to inf, which round() cannot take.
    count = round(scaled) if math.isfinite(scaled) else MAX_ATOMS_PER_SUB
    return [a] * min(MAX_ATOMS_PER_SUB, count)


def sub_action_atoms(text: str) -> List[List[int]]:
    """Atoms of each kept sub-action; an empty list is a sub-action the upstream
    evaluator replaces with a random fallback action."""
    return [_expand(sub) for sub in text.split(", ")[:MAX_SUBS]]


def chunk_text_to_atoms(text: str, rng=None) -> List[int]:
    """Decode chunk text as the upstream parser and expansion would.

    ``rng`` (the random module or a Random) reproduces the upstream fallback: an
    unparseable or zero-length sub-action becomes rng.randint(1, 3). Without it
    such sub-actions contribute nothing.
    """
    atoms: List[int] = []
    for got in sub_action_atoms(text):
        if not got and rng is not None:
            got = [rng.randint(1, 3)]
        atoms.extend(got)
    return atoms

