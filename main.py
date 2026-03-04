"""Run episodic Shift-Robust Conformal Robustness (SR-CR) updates for a CLF-QP pendulum controller."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from clf_controller import RobustCLFController
from conformal_updater import compute_quantile, compute_trajectory_score, update_margin
from pendulum_env import InvertedPendulum
from rollout import collect_trajectories


def main() -> None:
    """Run episodic SR-CR loop and plot margin/quantile evolution."""
    delta = 0.10
    kappa = 0.8
    r_0 = 2.0
    num_episodes = 5
    num_trajs_per_episode = 200
    dt = 0.02
    steps = int(20.0 / dt)

    seed = 42
    rng = np.random.default_rng(seed)
    weights_path = "nominal_model_weights.npz"
    results_path = "srcr_rollout_data.npz"

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
        use_robust_term=False,
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

    for episode in range(num_episodes):
        used_r_history.append(r_j)
        traj_residuals, traj_theta, traj_thetad, traj_v = collect_trajectories(
            env=env,
            controller=controller,
            r_j=r_j,
            num_trajs=num_trajs_per_episode,
            steps=steps,
            dt=dt,
            rng=rng,
            use_random_init=True,
        )
        all_episode_residuals.append(np.asarray(traj_residuals, dtype=np.float64))
        all_episode_theta.append(np.asarray(traj_theta, dtype=np.float64))
        all_episode_thetad.append(np.asarray(traj_thetad, dtype=np.float64))
        all_episode_v.append(np.asarray(traj_v, dtype=np.float64))

        scores = [compute_trajectory_score(residuals) for residuals in traj_residuals]
        q_j = compute_quantile(scores, delta)
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
        dt=float(dt),
        steps=int(steps),
        num_episodes=int(num_episodes),
        num_trajs_per_episode=int(num_trajs_per_episode),
        seed=int(seed),
        k_fb=np.asarray(controller.k_fb, dtype=np.float64),
        c3=float(controller.c3),
        delta=float(delta),
        kappa=float(kappa),
    )
    print(f"Saved rollout data to {results_path}")

    episode_idx = np.arange(num_episodes)

    plt.figure(figsize=(8, 5))
    plt.plot(episode_idx, r_history[:-1], marker="o", linewidth=2.0, label="r_j (margin)")
    plt.plot(episode_idx, q_history, marker="s", linewidth=2.0, label="q_j (quantile score)")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("SR-CR Episodic Margin Adaptation for Robust CLF-QP")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
