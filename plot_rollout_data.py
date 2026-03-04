"""Plot saved SR-CR rollout data from .npz file.

Expected arrays in the file:
- r_j: (num_episodes,)
- q_j: (num_episodes,)
- theta: (num_episodes, num_trajs, steps) robust trajectories
- thetad: (num_episodes, num_trajs, steps) robust trajectories
- v: (num_episodes, num_trajs, steps) robust trajectories
- theta_baseline/thetad_baseline/v_baseline: baseline trajectories (r=0),
  either shape (num_trajs, steps) or (num_episodes, num_trajs, steps)
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
    exp_decay_c3: float | None = None,
) -> None:
    """Create one figure per episode, overlaying robust and baseline trajectories."""
    num_episodes, _, steps = robust_data.shape
    t = np.arange(steps) * dt

    if baseline_data.ndim not in (2, 3):
        raise ValueError("baseline_data must have shape (num_trajs, steps) or (num_episodes, num_trajs, steps)")

    def baseline_for_episode(ep: int) -> np.ndarray:
        if baseline_data.ndim == 2:
            return baseline_data
        return baseline_data[ep]

    for ep in range(num_episodes):
        baseline_ep = baseline_for_episode(ep)
        if baseline_ep.shape[1] != steps:
            raise ValueError("Baseline and robust data must have the same number of timesteps.")

        fig, ax = plt.subplots(figsize=(8, 5))
        robust_handle = None
        baseline_handle = None
        exp_handle = None
        for traj in range(robust_data.shape[1]):
            line = ax.plot(t, robust_data[ep, traj], linewidth=1.0, alpha=0.35, color="tab:orange")[0]
            if robust_handle is None:
                robust_handle = line
        for traj in range(baseline_ep.shape[0]):
            line = ax.plot(t, baseline_ep[traj], linewidth=1.0, alpha=0.35, color="tab:blue")[0]
            if baseline_handle is None:
                baseline_handle = line
        if exp_decay_c3 is not None:
            v0 = float(robust_data[ep, 0, 0])
            v_ref = v0 * np.exp(-float(exp_decay_c3) * t)
            exp_handle = ax.plot(t, v_ref, "k:", linewidth=2.0)[0]
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(y_label)
        ax.set_title(f"{title_prefix} | Episode {ep:02d}")
        ax.grid(True, alpha=0.3)
        if robust_handle is not None and baseline_handle is not None and exp_handle is None:
            ax.legend(
                [robust_handle, baseline_handle],
                ["Robust (r_j)", "Baseline (r=0)"],
                loc="best",
            )
        if robust_handle is not None and baseline_handle is not None and exp_handle is not None:
            ax.legend(
                [robust_handle, baseline_handle, exp_handle],
                ["Robust (r_j)", "Baseline (r=0)", r"$V(x_0)e^{-c_3 t}$"],
                loc="best",
            )
        out_path = os.path.join(out_dir, f"{file_stem}_ep_{ep:02d}.png")
        _save_or_show(fig, out_path, show)


def _final_state_norm_metric(theta: np.ndarray, thetad: np.ndarray) -> np.ndarray:
    """Average final ||[theta, theta_dot]|| across trajectories."""
    norms = np.sqrt(theta**2 + thetad**2)
    return np.mean(norms[:, :, -1], axis=1)


def _final_v_metric(v: np.ndarray) -> np.ndarray:
    """Average final V(x_T) across trajectories."""
    return np.mean(v[:, :, -1], axis=1)


def _progress_metric(theta: np.ndarray, thetad: np.ndarray) -> np.ndarray:
    """Average progress per episode: ||x_{t-1}|| - ||x_t||, averaged in time and trajectories."""
    norms = np.sqrt(theta**2 + thetad**2)  # (E, N, T)
    delta = norms[:, :, :-1] - norms[:, :, 1:]  # (E, N, T-1)
    per_traj = np.mean(delta, axis=2)  # (E, N)
    return np.mean(per_traj, axis=1)  # (E,)


def _baseline_metric(
    theta_baseline: np.ndarray,
    thetad_baseline: np.ndarray,
    v_baseline: np.ndarray,
    metric: str,
) -> np.ndarray | float:
    """Compute baseline metric for either single-run baseline or legacy per-episode baseline."""
    if theta_baseline.ndim == 2:
        # (N, T)
        if metric == "final_state_norm":
            norms = np.sqrt(theta_baseline**2 + thetad_baseline**2)
            return float(np.mean(norms[:, -1]))
        if metric == "final_v":
            return float(np.mean(v_baseline[:, -1]))
        if metric == "progress":
            norms = np.sqrt(theta_baseline**2 + thetad_baseline**2)
            delta = norms[:, :-1] - norms[:, 1:]
            return float(np.mean(np.mean(delta, axis=1)))
        raise ValueError(f"Unknown metric: {metric}")

    # Legacy shape (E, N, T)
    if metric == "final_state_norm":
        return _final_state_norm_metric(theta_baseline, thetad_baseline)
    if metric == "final_v":
        return _final_v_metric(v_baseline)
    if metric == "progress":
        return _progress_metric(theta_baseline, thetad_baseline)
    raise ValueError(f"Unknown metric: {metric}")


def plot_episode_metric(
    robust_metric: np.ndarray,
    baseline_metric: np.ndarray | float,
    y_label: str,
    title: str,
    out_path: str,
    show: bool,
) -> None:
    """Plot one per-episode scalar metric for robust and baseline controllers."""
    episodes = np.arange(robust_metric.size)
    fig, ax = plt.subplots(figsize=(8, 5))
    robust_handle = ax.plot(episodes, robust_metric, marker="o", linewidth=2.0, color="tab:orange", label="Robust")[0]

    if np.isscalar(baseline_metric):
        baseline_handle = ax.axhline(
            float(baseline_metric),
            linestyle="--",
            linewidth=2.0,
            color="tab:blue",
            label="Baseline (r=0)",
        )
    else:
        baseline_metric = np.asarray(baseline_metric, dtype=np.float64)
        baseline_handle = ax.plot(
            episodes,
            baseline_metric,
            marker="s",
            linewidth=2.0,
            color="tab:blue",
            label="Baseline (r=0)",
        )[0]

    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[robust_handle, baseline_handle], loc="best")
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
    c3 = float(data["c3"]) if "c3" in data.files else 0.5

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
        exp_decay_c3=c3,
    )

    robust_final_norm = _final_state_norm_metric(theta, thetad)
    robust_final_v = _final_v_metric(v)
    robust_progress = _progress_metric(theta, thetad)

    baseline_final_norm = _baseline_metric(theta_baseline, thetad_baseline, v_baseline, "final_state_norm")
    baseline_final_v = _baseline_metric(theta_baseline, thetad_baseline, v_baseline, "final_v")
    baseline_progress = _baseline_metric(theta_baseline, thetad_baseline, v_baseline, "progress")

    plot_episode_metric(
        robust_metric=robust_final_norm,
        baseline_metric=baseline_final_norm,
        y_label=r"Avg final ||[theta, \dot{theta}]||",
        title="Average Final State Norm vs Episode",
        out_path=os.path.join(args.out_dir, "final_state_norm_vs_episode.png"),
        show=args.show,
    )
    plot_episode_metric(
        robust_metric=robust_final_v,
        baseline_metric=baseline_final_v,
        y_label="Avg final V(x_T)",
        title="Average Final CLF Value vs Episode",
        out_path=os.path.join(args.out_dir, "final_v_vs_episode.png"),
        show=args.show,
    )
    plot_episode_metric(
        robust_metric=robust_progress,
        baseline_metric=baseline_progress,
        y_label=r"Avg progress: ||x_{t-1}|| - ||x_t||",
        title="Average Progress Metric vs Episode",
        out_path=os.path.join(args.out_dir, "progress_vs_episode.png"),
        show=args.show,
    )

    if not args.show:
        print(f"Saved all plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
