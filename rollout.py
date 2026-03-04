"""Trajectory collection utilities for episodic SR-CR updates."""

from __future__ import annotations

import numpy as np

from clf_controller import RobustCLFController
from pendulum_env import InvertedPendulum


def collect_trajectories(
    env: InvertedPendulum,
    controller: RobustCLFController,
    r_j: float,
    num_trajs: int,
    steps: int,
    dt: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Collect trajectory residuals under robust CLF-QP control at margin r_j."""
    all_residuals: list[np.ndarray] = []

    for _ in range(int(num_trajs)):
        theta0 = rng.uniform(-np.pi / 4.0, np.pi / 4.0)
        theta_dot0 = rng.uniform(-1.0, 1.0)
        x = np.array([theta0, theta_dot0], dtype=np.float64)

        traj_residuals = []
        t = 0.0

        for _ in range(int(steps)):
            u = controller.compute_control(x, r_j)
            eps = env.compute_residual(x, u, t)
            traj_residuals.append(eps)

            x = env.step(x, u, t, dt)
            t += dt

        all_residuals.append(np.asarray(traj_residuals, dtype=np.float64))

    return all_residuals
