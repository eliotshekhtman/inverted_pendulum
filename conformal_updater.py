"""SR-CR trajectory scoring and episodic margin updates."""

from __future__ import annotations

import numpy as np


def compute_trajectory_score(residuals: np.ndarray) -> float:
    """Return S = max_t ||epsilon_t||_2 for one trajectory."""
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.size == 0:
        return 0.0
    norms = np.linalg.norm(residuals, axis=1)
    return float(np.max(norms))


def compute_quantile(scores: list[float], delta: float) -> float:
    """Conservative conformal-style empirical (1-delta)-quantile."""
    if len(scores) == 0:
        return 0.0

    delta = float(delta)
    sorted_scores = np.sort(np.asarray(scores, dtype=np.float64))
    n = sorted_scores.size

    # Conformal index: k = ceil((n + 1) * (1 - delta)), clipped to [1, n].
    k = int(np.ceil((n + 1) * (1.0 - delta)))
    k = int(np.clip(k, 1, n))
    return float(sorted_scores[k - 1])


def update_margin(q_j: float, r_j: float, kappa: float) -> float:
    """Piecewise affine SR-CR update for next episode margin."""
    q_j = float(q_j)
    r_j = float(r_j)
    kappa = float(kappa)

    if q_j >= r_j:
        r_next = (q_j - kappa * r_j) / (1.0 - kappa)
    else:
        r_next = (q_j + kappa * r_j) / (1.0 + kappa)
        r_next = max(0.0, r_next)

    return float(r_next)
