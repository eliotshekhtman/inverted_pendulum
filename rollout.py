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
    use_random_init: bool = False,
) -> list[np.ndarray]:
    """Run multiple closed-loop rollouts and collect residual traces.

    Each returned entry is an array of shape (steps, 2) containing
    epsilon_t = f_true(x_t,u_t,t) - (f_drift(x_t) + g_ctrl(x_t)u_t).
    """
    all_residuals: list[np.ndarray] = []

    for _ in range(int(num_trajs)):
        if use_random_init:
            theta0 = rng.uniform(-np.pi / 4.0, np.pi / 4.0)
            theta_dot0 = rng.uniform(-1.0, 1.0)
        else:
            theta0, theta_dot0 = 0.76, 0.05
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
