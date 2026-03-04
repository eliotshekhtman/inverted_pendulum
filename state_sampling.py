"""Shared state-initialization samplers for rollouts and model fitting."""

from __future__ import annotations

import numpy as np


def sample_state_with_target_v(
    P: np.ndarray,
    rng: np.random.Generator,
    v_target: float = 1.3,
    theta_bound: float = np.pi / 2.0,
    thetad_bound: float = 10.0,
    max_tries: int = 20000,
) -> np.ndarray:
    """Sample x0 inside bounds while satisfying x0^T P x0 = v_target.

    With x=[theta, theta_dot] and quadratic CLF V=x^T P x, fixing theta turns
    V=v_target into a quadratic in theta_dot, which is solved analytically.
    """
    P = np.asarray(P, dtype=np.float64)
    p11 = float(P[0, 0])
    p12 = float(P[0, 1])
    p22 = float(P[1, 1])
    v_target = float(v_target)

    if p22 <= 0.0:
        raise RuntimeError("Invalid CLF matrix: P[1,1] must be positive.")

    for _ in range(max_tries):
        theta = float(rng.uniform(-theta_bound, theta_bound))
        # p22*d^2 + 2*p12*theta*d + (p11*theta^2 - v_target) = 0
        A = p22
        B = 2.0 * p12 * theta
        C = p11 * theta * theta - v_target
        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            continue

        sqrt_disc = float(np.sqrt(max(disc, 0.0)))
        roots = [(-B + sqrt_disc) / (2.0 * A), (-B - sqrt_disc) / (2.0 * A)]
        if rng.uniform() < 0.5:
            roots.reverse()

        for theta_dot in roots:
            if abs(theta_dot) > thetad_bound:
                continue
            x = np.array([theta, theta_dot], dtype=np.float64)
            vx = float(x.T @ P @ x)
            if vx > 0.0:
                # Tiny correction for floating-point drift only.
                x *= np.sqrt(v_target / vx)
            if abs(x[0]) <= theta_bound + 1e-10 and abs(x[1]) <= thetad_bound + 1e-10:
                return x

    raise RuntimeError(
        "Failed to sample x0 with V(x0)=target inside bounds. "
        "Try relaxing bounds or changing target level."
    )


def sample_state_for_training(
    P: np.ndarray,
    rng: np.random.Generator,
    theta_bound: float = np.pi / 2.0,
    thetad_bound: float = 10.0,
    max_tries: int = 20000,
) -> np.ndarray:
    """Sample x0 for model fitting using random CLF level v_target in [0, 1.3]."""
    v_target = float(rng.uniform(0.0, 1.3))
    return sample_state_with_target_v(
        P=P,
        rng=rng,
        v_target=v_target,
        theta_bound=theta_bound,
        thetad_bound=thetad_bound,
        max_tries=max_tries,
    )
