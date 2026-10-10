"""Deployment score: the lost-vs-OK margin, aggregated over a rolling window.

The aggregation is part of the score the conformal threshold is calibrated on,
so it is fixed before calibration.
"""
import math
from collections import deque
from typing import Dict, Optional

# Conformal thresholds of the released judge adapter on the rolled lost margin, per
# dataset and target false-alarm rate alpha: the split-conformal order statistic over
# successful calibration episodes (window 3; one query per decision point),
# rounded to two decimals. Calibration/training code is not part of this release.
CALIBRATED_TAU = {
    "r2r": {0.025: 4.67, 0.05: 3.83, 0.075: 3.08, 0.10: 2.54, 0.125: 2.29},
    "rxr": {0.025: 4.37, 0.05: 3.71, 0.075: 3.33, 0.10: 3.00, 0.125: 2.71},
}
ALPHAS = (0.025, 0.05, 0.075, 0.10, 0.125)
DEPLOYED_ALPHA = {"r2r": 0.10, "rxr": 0.075}


def calibrated_tau(dataset: str, alpha: Optional[float] = None) -> float:
    """tau for ``dataset`` at ``alpha``; None takes the deployed alpha of that dataset."""
    if dataset not in CALIBRATED_TAU:
        raise ValueError(f"no calibrated threshold for dataset {dataset!r}; "
                         f"known: {sorted(CALIBRATED_TAU)}")
    if alpha is None:
        if dataset not in DEPLOYED_ALPHA:
            raise ValueError(f"no deployed alpha for dataset {dataset!r}; pass one of {ALPHAS}")
        alpha = DEPLOYED_ALPHA[dataset]
    table = CALIBRATED_TAU[dataset]
    match = [a for a in table if math.isclose(a, float(alpha))]
    if not match:
        raise ValueError(f"alpha {alpha} is not calibrated; choose from {ALPHAS}")
    return table[match[0]]


def lost_margin(probs: Dict[str, float], eps: float = 1e-9) -> float:
    """logit(lost) - logit(OK); recomputable offline from the class probabilities."""
    return math.log(probs["lost"] + eps) - math.log(probs["OK"] + eps)


class RollingRisk:
    """Rolling aggregate over decision-point scores; reset() clears it after a restart."""

    MODES = ("mean", "max")

    def __init__(self, window: int = 3, mode: str = "mean"):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self.mode = mode
        self._buf: deque = deque(maxlen=window)

    def update(self, score: float) -> float:
        self._buf.append(float(score))
        if self.mode == "mean":
            return sum(self._buf) / len(self._buf)
        return max(self._buf)

    def reset(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

