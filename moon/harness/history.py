from typing import List

NUM_HISTORY = 8   # NUM_HISTORY + 1 frames per sample, the deployed judge window


def sample_history_indices(current_step: int, num_history: int = NUM_HISTORY) -> List[int]:
    """num_history + 1 frame indices spanning [0, current_step]; the last is current."""
    import numpy as np
    if current_step <= num_history:
        return list(range(current_step)) + [current_step]
    idx = np.linspace(0, current_step, num_history + 1, dtype=int).tolist()
    idx[-1] = current_step
    return idx

