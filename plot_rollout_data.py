"""Plot saved SR-CR rollout data from .npz file.

Expected arrays in the file:
- r_j: (num_episodes,)
- q_j: (num_episodes,)
- theta: (num_episodes, num_trajs, steps) robust trajectories
- thetad: (num_episodes, num_trajs, steps) robust trajectories
- v: (num_episodes, num_trajs, steps) robust trajectories
- theta_baseline/thetad_baseline/v_baseline: baseline trajectories (r=0)
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot episodic pendulum rollout traces from npz data.")
    parser.add_argument("--input", type=str, default="srcr_rollout_data.npz", help="Path to saved rollout data.")
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory where all output figures will be written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive windows. By default, figures are saved and closed.",
    )
    return parser.parse_args()


def _save_or_show(fig: plt.Figure, path: str, show: bool) -> None:
    fig.tight_layout()
    if show:
        plt.show()
    else:
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_rq(r_j: np.ndarray, q_j: np.ndarray, out_path: str, show: bool) -> None:
    episodes = np.arange(r_j.size)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episodes, r_j, marker="o", linewidth=2.0, label="r_j")
    ax.plot(episodes, q_j, marker="s", linewidth=2.0, label="q_j")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.set_title("SR-CR: r_j and q_j over Episodes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_or_show(fig, out_path, show)


def plot_episode_comparison(
    robust_data: np.ndarray,
    baseline_data: np.ndarray,
    dt: float,
    y_label: str,
    title_prefix: str,
    out_dir: str,
    file_stem: str,
    show: bool,
) -> None:
    """Create one figure per episode, overlaying robust and baseline trajectories."""
    num_episodes, _, steps = robust_data.shape
    t = np.arange(steps) * dt

    for ep in range(num_episodes):
        fig, ax = plt.subplots(figsize=(8, 5))
        robust_handle = None
        baseline_handle = None
        for traj in range(robust_data.shape[1]):
            line = ax.plot(t, robust_data[ep, traj], linewidth=1.0, alpha=0.35, color="tab:orange")[0]
            if robust_handle is None:
                robust_handle = line
        for traj in range(baseline_data.shape[1]):
            line = ax.plot(t, baseline_data[ep, traj], linewidth=1.0, alpha=0.35, color="tab:blue")[0]
            if baseline_handle is None:
                baseline_handle = line
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(y_label)
        ax.set_title(f"{title_prefix} | Episode {ep:02d}")
        ax.grid(True, alpha=0.3)
        if robust_handle is not None and baseline_handle is not None:
            ax.legend(
                [robust_handle, baseline_handle],
                ["Robust (r_j)", "Baseline (r=0)"],
                loc="best",
            )
        out_path = os.path.join(out_dir, f"{file_stem}_ep_{ep:02d}.png")
        _save_or_show(fig, out_path, show)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = np.load(args.input)

    r_j = np.asarray(data["r_j"], dtype=np.float64)
    q_j = np.asarray(data["q_j"], dtype=np.float64)
    theta = np.asarray(data["theta"], dtype=np.float64)
    thetad = np.asarray(data["thetad"], dtype=np.float64)
    v = np.asarray(data["v"], dtype=np.float64)
    theta_baseline = np.asarray(data["theta_baseline"], dtype=np.float64)
    thetad_baseline = np.asarray(data["thetad_baseline"], dtype=np.float64)
    v_baseline = np.asarray(data["v_baseline"], dtype=np.float64)
    dt = float(data["dt"])

    plot_rq(r_j, q_j, out_path=os.path.join(args.out_dir, "rj_qj.png"), show=args.show)
    plot_episode_comparison(
        robust_data=theta,
        baseline_data=theta_baseline,
        dt=dt,
        y_label="theta (rad)",
        title_prefix="Theta vs Time",
        out_dir=args.out_dir,
        file_stem="theta",
        show=args.show,
    )
    plot_episode_comparison(
        robust_data=thetad,
        baseline_data=thetad_baseline,
        dt=dt,
        y_label="theta_dot (rad/s)",
        title_prefix="Theta_dot vs Time",
        out_dir=args.out_dir,
        file_stem="thetad",
        show=args.show,
    )
    plot_episode_comparison(
        robust_data=v,
        baseline_data=v_baseline,
        dt=dt,
        y_label="V(x_t)",
        title_prefix="CLF Value vs Time",
        out_dir=args.out_dir,
        file_stem="V",
        show=args.show,
    )

    if not args.show:
        print(f"Saved all plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
