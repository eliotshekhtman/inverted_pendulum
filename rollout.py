"""Trajectory collection utilities for episodic SR-CR updates."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from clf_controller import RobustCLFController
from pendulum_env import InvertedPendulum
from state_sampling import sample_state_with_target_v


def collect_trajectories(
    env: InvertedPendulum,
    controller: RobustCLFController,
    r_j: float,
    num_trajs: int,
    steps: int,
    dt: float,
    rng: np.random.Generator,
    use_random_init: bool = False,
    show_progress: bool = True,
    progress_desc: str = "Collecting trajectories",
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Run multiple closed-loop rollouts and collect residual traces.

    Each returned entry is an array of shape (steps, 2) containing
    epsilon_t = f_true(x_t,u_t,t) - (f_drift(x_t) + g_ctrl(x_t)u_t).

    Returns:
        residuals_all: list of (steps, 2) residual arrays.
        thetas_all: list of (steps,) theta trajectories.
        thetads_all: list of (steps,) theta_dot trajectories.
        v_all: list of (steps,) CLF value trajectories V(x_t).
    """
    all_residuals: list[np.ndarray] = []
    all_thetas: list[np.ndarray] = []
    all_thetads: list[np.ndarray] = []
    all_v: list[np.ndarray] = []

    traj_iter = tqdm(
        range(int(num_trajs)),
        desc=progress_desc,
        leave=False,
        disable=not show_progress,
    )
    for _ in traj_iter:
        if use_random_init:
            x = sample_state_with_target_v(
                P=controller.P,
                rng=rng,
                v_target=1.3,
                theta_bound=np.pi / 2.0,
                thetad_bound=10.0,
            )
        else:
            x = np.array([0.76, 0.05], dtype=np.float64)

        traj_residuals = []
        traj_theta = []
        traj_thetad = []
        traj_v = []
        t = 0.0

        for _ in range(int(steps)):
            traj_theta.append(float(x[0]))
            traj_thetad.append(float(x[1]))
            traj_v.append(float(controller.V(x)))

            u = controller.compute_control(x, r_j)
            eps = env.compute_residual(x, u, t)
            traj_residuals.append(eps)

            x = env.step(x, u, t, dt)
            t += dt

        all_residuals.append(np.asarray(traj_residuals, dtype=np.float64))
        all_thetas.append(np.asarray(traj_theta, dtype=np.float64))
        all_thetads.append(np.asarray(traj_thetad, dtype=np.float64))
        all_v.append(np.asarray(traj_v, dtype=np.float64))

    return all_residuals, all_thetas, all_thetads, all_v
