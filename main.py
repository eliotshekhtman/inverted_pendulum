"""Run episodic Shift-Robust Conformal Robustness (SR-CR) updates for a CLF-QP pendulum controller."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from clf_controller import RobustCLFController
from conformal_updater import compute_quantile, compute_trajectory_score, update_margin
from pendulum_env import InvertedPendulum
from rollout import collect_trajectories


def main() -> None:
    delta = 0.10
    kappa = 0.8
    r_0 = 0.0
    num_episodes = 15
    num_trajs_per_episode = 20
    dt = 0.05
    steps = 100

    seed = 42
    rng = np.random.default_rng(seed)

    env = InvertedPendulum()
    controller = RobustCLFController(env=env)

    r_j = float(r_0)
    r_history = [r_j]
    q_history: list[float] = []

    for episode in range(num_episodes):
        traj_residuals = collect_trajectories(
            env=env,
            controller=controller,
            r_j=r_j,
            num_trajs=num_trajs_per_episode,
            steps=steps,
            dt=dt,
            rng=rng,
        )

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
