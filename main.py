"""Run episodic Shift-Robust Conformal Robustness (SR-CR) updates for a CLF-QP pendulum controller."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from clf_controller import RobustCLFController
from conformal_updater import compute_quantile, compute_trajectory_score, get_alpha_bar, update_margin
from pendulum_env import InvertedPendulum
from rollout import collect_trajectories


def main() -> None:
    """Run episodic SR-CR loop and plot margin/quantile evolution."""
    alpha = 0.10
    delta = 0.10
    kappa = 0.8
    r_0 = 2.0
    num_episodes = 3
    num_trajs_per_episode = 200
    dt = 0.02
    steps = int(5.0 / dt)

    seed = 42
    rng = np.random.default_rng(seed)
    weights_path = "nominal_model_weights.npz"
    results_path = "srcr_rollout_data.npz"
    alpha_bar = get_alpha_bar(alpha, delta, num_trajs_per_episode)
    alpha_bar = float(np.clip(alpha_bar, 1e-6, 1.0 - 1e-6))
    print(f"Using alpha={alpha:.4f}, delta={delta:.4f}, alpha_bar={alpha_bar:.4f}")

    env = InvertedPendulum(
        g=9.81,
        m=1.0,
        l=1.0,
        b=0.01,
        m_hat=1.0,
        l_hat=1.0,
        b_hat=0.01,
        u_min=-7.0,
        u_max=7.0,
        disturbance_amp=0.0,
    )
    if os.path.exists(weights_path):
        env.load_nominal_weights(weights_path)
        print(f"Loaded nominal model weights from {weights_path}")
    else:
        print(
            f"Nominal weights file '{weights_path}' not found. "
            "Using bootstrap nominal weights."
        )

    controller = RobustCLFController(
        env=env,
        k_fb=(8.0, 5.0),
        c3=0.5,
        weight_input=1.0,
        weight_slack=100000.0,
        u_ref=0.0,
        use_robust_term=True,
        auto_select_k=True,
        stability_margin=1e-3,
        max_p_condition=1e7,
    )
    print(f"CLF linearization gain K used: {controller.k_fb.tolist()}")

    r_j = float(r_0)
    r_history = [r_j]
    q_history: list[float] = []
    used_r_history: list[float] = []

    all_episode_residuals: list[np.ndarray] = []
    all_episode_theta: list[np.ndarray] = []
    all_episode_thetad: list[np.ndarray] = []
    all_episode_v: list[np.ndarray] = []

    baseline_seed = int(rng.integers(0, 2**32 - 1))
    rng_baseline = np.random.default_rng(baseline_seed)
    (
        traj_residuals_baseline,
        traj_theta_baseline,
        traj_thetad_baseline,
        traj_v_baseline,
    ) = collect_trajectories(
        env=env,
        controller=controller,
        r_j=0.0,
        num_trajs=num_trajs_per_episode,
        steps=steps,
        dt=dt,
        rng=rng_baseline,
        use_random_init=True,
        show_progress=True,
        progress_desc="Baseline (r=0) trajectories",
    )
    baseline_residuals = np.asarray(traj_residuals_baseline, dtype=np.float64)
    baseline_theta = np.asarray(traj_theta_baseline, dtype=np.float64)
    baseline_thetad = np.asarray(traj_thetad_baseline, dtype=np.float64)
    baseline_v = np.asarray(traj_v_baseline, dtype=np.float64)

    for episode in range(num_episodes):
        used_r_history.append(r_j)
        episode_seed = int(rng.integers(0, 2**32 - 1))
        rng_robust = np.random.default_rng(episode_seed)

        traj_residuals, traj_theta, traj_thetad, traj_v = collect_trajectories(
            env=env,
            controller=controller,
            r_j=r_j,
            num_trajs=num_trajs_per_episode,
            steps=steps,
            dt=dt,
            rng=rng_robust,
            use_random_init=True,
            show_progress=True,
            progress_desc=f"Episode {episode:02d} robust",
        )
        all_episode_residuals.append(np.asarray(traj_residuals, dtype=np.float64))
        all_episode_theta.append(np.asarray(traj_theta, dtype=np.float64))
        all_episode_thetad.append(np.asarray(traj_thetad, dtype=np.float64))
        all_episode_v.append(np.asarray(traj_v, dtype=np.float64))

        scores = [compute_trajectory_score(residuals) for residuals in traj_residuals]
        q_j = compute_quantile(scores, alpha_bar)
        r_next = update_margin(q_j=q_j, r_j=r_j, kappa=kappa)

        q_history.append(q_j)
        r_j = r_next
        r_history.append(r_j)

        print(
            f"Episode {episode:02d} | "
            f"r_j={r_history[-2]:.4f}, q_j={q_j:.4f}, r_next={r_j:.4f}"
        )

    np.savez(
        results_path,
        r_j=np.asarray(used_r_history, dtype=np.float64),
        q_j=np.asarray(q_history, dtype=np.float64),
        r_full=np.asarray(r_history, dtype=np.float64),
        theta=np.asarray(all_episode_theta, dtype=np.float64),
        thetad=np.asarray(all_episode_thetad, dtype=np.float64),
        v=np.asarray(all_episode_v, dtype=np.float64),
        residuals=np.asarray(all_episode_residuals, dtype=np.float64),
        theta_baseline=baseline_theta,
        thetad_baseline=baseline_thetad,
        v_baseline=baseline_v,
        residuals_baseline=baseline_residuals,
        dt=float(dt),
        steps=int(steps),
        num_episodes=int(num_episodes),
        num_trajs_per_episode=int(num_trajs_per_episode),
        seed=int(seed),
        baseline_seed=int(baseline_seed),
        k_fb=np.asarray(controller.k_fb, dtype=np.float64),
        c3=float(controller.c3),
        delta=float(delta),
        alpha=float(alpha),
        alpha_bar=float(alpha_bar),
        kappa=float(kappa),
    )
    print(f"Saved rollout data to {results_path}")

    episode_idx = np.arange(num_episodes)


if __name__ == "__main__":
    main()
