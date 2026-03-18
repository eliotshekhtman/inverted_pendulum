"""Combined episode-level plots from robust and naive SR-CR runs.

This script loads two rollout .npz files:
- robust run from main.py
- naive run from main_naive.py

It plots episode-level comparisons for:
1) r_j and q_j
2) average final ||[theta, theta_dot]||
3) average final V(x_T)
4) average progress metric: ||x_{t-1}|| - ||x_t||
5) trajectory-wise overlays for theta, thetad, V, and ||x|| with decay envelopes.

Nonrobust line is built by averaging baseline (r=0) statistics from both files.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot combined robust/naive/nonrobust episode metrics.")
    parser.add_argument("--robust_input", type=str, default="srcr_rollout_data.npz")
    parser.add_argument("--naive_input", type=str, default="srcr_rollout_data_naive.npz")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _save_or_show(fig: plt.Figure, path: str, show: bool) -> None:
    fig.tight_layout()
    if show:
        plt.show()
    else:
        base, _ = os.path.splitext(path)
        fig.savefig(base + ".pdf", dpi=150)
        plt.close(fig)


def _set_integer_episode_ticks(ax: plt.Axes) -> None:
    """Force x-axis ticks to be integer episode indices."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _trajectory_scores_from_residuals(residuals: np.ndarray) -> np.ndarray:
    """Return per-trajectory max residual norm scores."""
    # residuals shape can be (N, T, 2) or (E, N, T, 2)
    if residuals.ndim == 4:
        residuals = residuals.reshape(-1, residuals.shape[2], residuals.shape[3])
    norms = np.linalg.norm(residuals, axis=2)  # (N, T)
    return np.max(norms, axis=1)


def _as_episode_data(data: np.ndarray, name: str, ndim_expected: int = 3) -> np.ndarray:
    """Normalize trajectory arrays to shape (E, N, T[, ...])."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == ndim_expected:
        return arr[None, ...]
    if arr.ndim == ndim_expected + 1:
        return arr
    raise ValueError(f"{name} must have shape (N,T) or (E,N,T); got {arr.shape}")


def _as_episode_residual_data(data: np.ndarray, name: str) -> np.ndarray:
    """Normalize residual arrays to shape (E, N, T, 2)."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 3:
        return arr[None, ...]
    if arr.ndim == 4:
        return arr
    raise ValueError(f"{name} must have shape (N,T,2) or (E,N,T,2); got {arr.shape}")


def _align_baseline_series(series: np.ndarray, n: int) -> np.ndarray:
    """Repeat scalar baseline series to per-episode length if needed."""
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    if series.size == 1:
        return np.full(n, float(series[0]), dtype=np.float64)
    if series.size < n:
        raise ValueError(f"Baseline series has only {series.size} episodes, expected at least {n}.")
    return series[:n]


def _baseline_episode(
    baseline_data: np.ndarray,
    episode_idx: int,
) -> np.ndarray:
    """Return baseline trajectories for a given episode.

    Baseline may either be stored once as (N, T) (single non-robust rollout)
    or per episode as (E, N, T).
    """
    if baseline_data.ndim == 2:
        return baseline_data
    if baseline_data.ndim == 3:
        if baseline_data.shape[0] == 0:
            raise ValueError("baseline data has zero episodes.")
        ep = int(min(max(episode_idx, 0), baseline_data.shape[0] - 1))
        return baseline_data[ep]
    raise ValueError("baseline data must be shape (N,T) or (E,N,T)")


def _calibrate_once_series(series: np.ndarray, n: int) -> np.ndarray:
    """Create a calibrate-once per-episode series.

    Episode t uses:
    - episode 0 from the raw series when t==0
    - episode 1 from the raw series when t>=1

    If only one episode is available, it is used for all t.
    """
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("Cannot calibrate series with no dimension.")
    if n <= 0:
        raise ValueError("n must be a positive number of episodes.")

    if arr.ndim == 1:
        if arr.size == 0:
            raise ValueError("Cannot calibrate from empty series.")
        first = float(arr[0])
        second = float(arr[1]) if arr.size >= 2 else first
        out = np.full(n, second, dtype=np.float64)
        out[0] = first
        return out

    # Trajectory array style (E, N, T).
    if arr.ndim == 3:
        if arr.shape[0] == 0:
            raise ValueError("Cannot calibrate from empty trajectory series.")
        second = arr[1] if arr.shape[0] >= 2 else arr[0]
        out = np.repeat(np.expand_dims(second, axis=0), repeats=n, axis=0).copy()
        out[0] = arr[0]
        return out
    # Residual trajectory style (E, N, T, D).
    if arr.ndim == 4:
        if arr.shape[0] == 0:
            raise ValueError("Cannot calibrate from empty residual series.")
        second = arr[1] if arr.shape[0] >= 2 else arr[0]
        out = np.repeat(np.expand_dims(second, axis=0), repeats=n, axis=0).copy()
        out[0] = arr[0]
        return out

    raise ValueError(f"Unsupported series shape for calibration: {arr.shape}")


def _plot_episode_comparison(
    robust_data: np.ndarray,
    naive_data: np.ndarray,
    baseline_data_robust: np.ndarray,
    baseline_data_naive: np.ndarray,
    y_label: str,
    file_stem: str,
    out_dir: str,
    show: bool,
    dt: float,
    exp_decay_c3: float | None = None,
    calibrate_data: np.ndarray | None = None,
) -> None:
    """Plot one figure per episode for robust/naive trajectories.

    This mirrors the per-episode plot in plot_rollout_data.py, extended to include
    both robust and naive runs plus a pooled non-robust baseline overlay.
    """
    if robust_data.ndim != 3 or naive_data.ndim != 3:
        raise ValueError("robust_data and naive_data must have shape (E, N, T).")
    num_episodes, num_trajs, steps = robust_data.shape
    if naive_data.shape != (num_episodes, num_trajs, steps):
        num_episodes = min(num_episodes, naive_data.shape[0])
        num_trajs = min(num_trajs, naive_data.shape[1])
        steps = min(steps, naive_data.shape[2])
        robust_data = robust_data[:num_episodes, :num_trajs, :steps]
        naive_data = naive_data[:num_episodes, :num_trajs, :steps]

    if calibrate_data is not None:
        calibrate_data = np.asarray(calibrate_data, dtype=np.float64)
        calibrate_data = calibrate_data[:num_episodes, :num_trajs, :steps]

    t = np.arange(steps) * float(dt)

    for ep in range(num_episodes):
        robust_ep = robust_data[ep]
        naive_ep = naive_data[ep]
        baseline_ep = np.concatenate(
            [
                _baseline_episode(baseline_data_robust, ep),
                _baseline_episode(baseline_data_naive, ep),
            ],
            axis=0,
        )
        if baseline_ep.shape[1] != steps:
            raise ValueError("Baseline and robust/naive data must have matching timesteps.")

        fig, ax = plt.subplots(figsize=(8, 5))
        robust_handle = None
        naive_handle = None
        baseline_handle = None
        robust_bound_handle = None
        naive_bound_handle = None
        baseline_bound_handle = None

        for traj in range(robust_ep.shape[0]):
            line = ax.plot(t, robust_ep[traj], linewidth=1.0, alpha=0.35, color="tab:orange")[0]
            if robust_handle is None:
                robust_handle = line
        for traj in range(naive_ep.shape[0]):
            line = ax.plot(t, naive_ep[traj], linewidth=1.0, alpha=0.35, color="tab:green")[0]
            if naive_handle is None:
                naive_handle = line
        for traj in range(baseline_ep.shape[0]):
            line = ax.plot(t, baseline_ep[traj], linewidth=1.0, alpha=0.35, color="tab:blue")[0]
            if baseline_handle is None:
                baseline_handle = line

        if calibrate_data is not None:
            cal_ep = calibrate_data[ep]
            cal_handle = None
            for traj in range(cal_ep.shape[0]):
                line = ax.plot(t, cal_ep[traj], linewidth=1.0, alpha=0.40, color="tab:purple")[0]
                if cal_handle is None:
                    cal_handle = line
        else:
            cal_handle = None

        if exp_decay_c3 is not None:
            decay = np.exp(-float(exp_decay_c3) * t)
            for traj in range(robust_ep.shape[0]):
                v0 = float(robust_ep[traj, 0])
                line = ax.plot(t, v0 * decay, ":", linewidth=1.0, alpha=0.20, color="tab:orange")[0]
                if robust_bound_handle is None:
                    robust_bound_handle = line
            for traj in range(naive_ep.shape[0]):
                v0 = float(naive_ep[traj, 0])
                line = ax.plot(t, v0 * decay, ":", linewidth=1.0, alpha=0.20, color="tab:green")[0]
                if naive_bound_handle is None:
                    naive_bound_handle = line
            for traj in range(baseline_ep.shape[0]):
                v0 = float(baseline_ep[traj, 0])
                line = ax.plot(t, v0 * decay, ":", linewidth=1.0, alpha=0.20, color="tab:blue")[0]
                if baseline_bound_handle is None:
                    baseline_bound_handle = line

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} | Episode {ep:02d}")
        ax.grid(True, alpha=0.3)

        if exp_decay_c3 is None or (
            robust_bound_handle is None and naive_bound_handle is None and baseline_bound_handle is None
        ):
            ax.legend(
                [h for h in (robust_handle, naive_handle, baseline_handle, cal_handle) if h is not None],
                [lbl for h, lbl in (
                    (robust_handle, "Robust (r_j)"),
                    (naive_handle, "Naive (r_j)"),
                    (baseline_handle, "Nonrobust"),
                    (cal_handle, "Calibrate once"),
                ) if h is not None],
                loc="best",
            )
        else:
            handles = [h for h in (robust_handle, naive_handle, baseline_handle,
                                   robust_bound_handle, naive_bound_handle, baseline_bound_handle, cal_handle) if h is not None]
            labels = [lbl for h, lbl in (
                (robust_handle, "Robust (r_j)"),
                (naive_handle, "Naive (r_j)"),
                    (baseline_handle, "Nonrobust"),
                (cal_handle, "Calibrate once"),
                (robust_bound_handle, r"Robust bound: $V(x_{0,i})e^{-c_3 t}$"),
                (naive_bound_handle, r"Naive bound: $V(x_{0,i})e^{-c_3 t}$"),
                (baseline_bound_handle, r"Nonrobust bound: $V(x_{0,i})e^{-c_3 t}$"),
            ) if h is not None]
            ax.legend(handles, labels, loc="best")
        out_path = os.path.join(out_dir, f"{file_stem}_ep_{ep:02d}_combined.png")
        _save_or_show(fig, out_path, show)


def _plot_norm_episode_comparison(
    robust_theta: np.ndarray,
    robust_thetad: np.ndarray,
    naive_theta: np.ndarray,
    naive_thetad: np.ndarray,
    baseline_theta_robust: np.ndarray,
    baseline_thetad_robust: np.ndarray,
    baseline_theta_naive: np.ndarray,
    baseline_thetad_naive: np.ndarray,
    dt: float,
    out_dir: str,
    show: bool,
    calibrate_norm_robust: np.ndarray | None = None,
    calibrate_norm_naive: np.ndarray | None = None,
    c1: float | None = None,
    c2: float | None = None,
    c3: float | None = None,
) -> None:
    """Plot ||x_t|| over time per episode for robust/naive/non-robust data."""
    robust_norm = np.sqrt(robust_theta**2 + robust_thetad**2)
    naive_norm = np.sqrt(naive_theta**2 + naive_thetad**2)
    if robust_norm.shape != naive_norm.shape:
        raise ValueError("robust and naive state tensors must have matching shape.")

    num_episodes, _, steps = robust_norm.shape
    t = np.arange(steps) * float(dt)

    baseline_norm_robust = np.sqrt(baseline_theta_robust**2 + baseline_thetad_robust**2)
    baseline_norm_naive = np.sqrt(baseline_theta_naive**2 + baseline_thetad_naive**2)

    if calibrate_norm_robust is not None:
        calibrate_norm_robust = np.asarray(calibrate_norm_robust, dtype=np.float64)
        calibrate_norm_robust = calibrate_norm_robust[:num_episodes, :, :steps]
    if calibrate_norm_naive is not None:
        calibrate_norm_naive = np.asarray(calibrate_norm_naive, dtype=np.float64)
        calibrate_norm_naive = calibrate_norm_naive[:num_episodes, :, :steps]

    for ep in range(num_episodes):
        baseline_ep = np.concatenate(
            [
                _baseline_episode(baseline_norm_robust, ep),
                _baseline_episode(baseline_norm_naive, ep),
            ],
            axis=0,
        )
        robust_ep = robust_norm[ep]
        naive_ep = naive_norm[ep]
        if baseline_ep.shape[1] != steps:
            raise ValueError("Baseline and robust/naive norm traces must match timesteps.")

        fig, ax = plt.subplots(figsize=(8, 5))
        robust_handle = None
        naive_handle = None
        baseline_handle = None
        robust_bound_handle = None
        naive_bound_handle = None
        baseline_bound_handle = None

        for traj in range(robust_ep.shape[0]):
            line = ax.plot(t, robust_ep[traj], linewidth=1.0, alpha=0.35, color="tab:orange")[0]
            if robust_handle is None:
                robust_handle = line
        for traj in range(naive_ep.shape[0]):
            line = ax.plot(t, naive_ep[traj], linewidth=1.0, alpha=0.35, color="tab:green")[0]
            if naive_handle is None:
                naive_handle = line
        for traj in range(baseline_ep.shape[0]):
            line = ax.plot(t, baseline_ep[traj], linewidth=1.0, alpha=0.35, color="tab:blue")[0]
            if baseline_handle is None:
                baseline_handle = line

        cal_handle = None
        if calibrate_norm_robust is not None:
            cal_ep = calibrate_norm_robust[ep]
            for traj in range(cal_ep.shape[0]):
                line = ax.plot(t, cal_ep[traj], linewidth=1.0, alpha=0.40, color="tab:purple")[0]
                if cal_handle is None:
                    cal_handle = line
        elif calibrate_norm_naive is not None:
            cal_ep = calibrate_norm_naive[ep]
            for traj in range(cal_ep.shape[0]):
                line = ax.plot(t, cal_ep[traj], linewidth=1.0, alpha=0.40, color="tab:purple")[0]
                if cal_handle is None:
                    cal_handle = line

        if (
            c1 is not None
            and c2 is not None
            and c3 is not None
            and float(c1) > 0.0
            and np.isfinite(float(c1))
            and np.isfinite(float(c2))
        ):
            decay = np.exp(-0.5 * float(c3) * t)
            gain = np.sqrt(float(c2) / float(c1))
            for traj in range(robust_ep.shape[0]):
                coeff = gain * float(robust_ep[traj, 0])
                line = ax.plot(t, coeff * decay, ":", linewidth=1.0, alpha=0.20, color="tab:orange")[0]
                if robust_bound_handle is None:
                    robust_bound_handle = line
            for traj in range(naive_ep.shape[0]):
                coeff = gain * float(naive_ep[traj, 0])
                line = ax.plot(t, coeff * decay, ":", linewidth=1.0, alpha=0.20, color="tab:green")[0]
                if naive_bound_handle is None:
                    naive_bound_handle = line
            for traj in range(baseline_ep.shape[0]):
                coeff = gain * float(baseline_ep[traj, 0])
                line = ax.plot(t, coeff * decay, ":", linewidth=1.0, alpha=0.20, color="tab:blue")[0]
                if baseline_bound_handle is None:
                    baseline_bound_handle = line

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$||x_t||_2$")
        ax.set_title(f"State Norm vs Time | Episode {ep:02d}")
        ax.grid(True, alpha=0.3)

        if robust_bound_handle is None and naive_bound_handle is None and baseline_bound_handle is None:
            ax.legend(
                [h for h in (robust_handle, naive_handle, baseline_handle, cal_handle) if h is not None],
                [lbl for h, lbl in (
                    (robust_handle, "Robust (r_j)"),
                    (naive_handle, "Naive (r_j)"),
                    (baseline_handle, "Nonrobust"),
                    (cal_handle, "Calibrate once"),
                ) if h is not None],
                loc="best",
            )
        else:
            handles = [h for h in (robust_handle, naive_handle, baseline_handle,
                                   robust_bound_handle, naive_bound_handle, baseline_bound_handle, cal_handle) if h is not None]
            labels = [lbl for h, lbl in (
                (robust_handle, "Robust (r_j)"),
                (naive_handle, "Naive (r_j)"),
                (baseline_handle, "Nonrobust"),
                (cal_handle, "Calibrate once"),
                (robust_bound_handle, r"Robust bound: $\sqrt{c_2/c_1}\|x_{0,i}\|e^{-c_3 t/2}$"),
                (naive_bound_handle, r"Naive bound: $\sqrt{c_2/c_1}\|x_{0,i}\|e^{-c_3 t/2}$"),
                (baseline_bound_handle, r"Nonrobust bound: $\sqrt{c_2/c_1}\|x_{0,i}\|e^{-c_3 t/2}$"),
            ) if h is not None]
            ax.legend(handles, labels, loc="best")

        out_path = os.path.join(out_dir, f"norm_ep_{ep:02d}_combined.png")
        _save_or_show(fig, out_path, show)


def _compute_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conservative conformal quantile at level (1-alpha)."""
    if scores.size == 0:
        return 0.0
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = s.size
    k = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    k = int(np.clip(k, 1, n))
    return float(s[k - 1])


def _final_state_norm_metric(theta: np.ndarray, thetad: np.ndarray) -> np.ndarray:
    norms = np.sqrt(theta**2 + thetad**2)  # (E, N, T)
    return np.mean(norms[:, :, -1], axis=1)


def _final_state_norm_metric_stats(
    theta: np.ndarray,
    thetad: np.ndarray,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Final norm per episode: mean, q_low, q_high."""
    norms = np.sqrt(theta**2 + thetad**2)  # (E, N, T)
    final = norms[:, :, -1]  # (E, N)
    return (
        np.mean(final, axis=1),
        np.quantile(final, q_low, axis=1),
        np.quantile(final, q_high, axis=1),
    )


def _final_v_metric(v: np.ndarray) -> np.ndarray:
    return np.mean(v[:, :, -1], axis=1)


def _final_v_metric_stats(
    v: np.ndarray,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Final V per episode: mean, q_low, q_high."""
    final = v[:, :, -1]  # (E, N)
    return (
        np.mean(final, axis=1),
        np.quantile(final, q_low, axis=1),
        np.quantile(final, q_high, axis=1),
    )


def _progress_metric(theta: np.ndarray, thetad: np.ndarray) -> np.ndarray:
    norms = np.sqrt(theta**2 + thetad**2)  # (E, N, T)
    delta = norms[:, :, :-1] - norms[:, :, 1:]  # (E, N, T-1)
    return np.mean(np.mean(delta, axis=2), axis=1)


def _progress_metric_stats(
    theta: np.ndarray,
    thetad: np.ndarray,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average trajectory progress per episode: mean, q_low, q_high."""
    norms = np.sqrt(theta**2 + thetad**2)  # (E, N, T)
    trajectory_progress = np.mean(norms[:, :, :-1] - norms[:, :, 1:], axis=2)  # (E, N)
    return (
        np.mean(trajectory_progress, axis=1),
        np.quantile(trajectory_progress, q_low, axis=1),
        np.quantile(trajectory_progress, q_high, axis=1),
    )


def _baseline_scalar_metric(theta_b: np.ndarray, thetad_b: np.ndarray, v_b: np.ndarray, metric: str) -> float:
    """Compute a single scalar baseline metric from one dataset.

    Accepts baseline arrays as either:
    - (N, T)
    - (E, N, T)
    """
    if theta_b.ndim == 3:
        # Collapse episodes for a single scalar summary.
        theta_b = theta_b.reshape(-1, theta_b.shape[2])
        thetad_b = thetad_b.reshape(-1, thetad_b.shape[2])
        v_b = v_b.reshape(-1, v_b.shape[2])

    if metric == "final_state_norm":
        norms = np.sqrt(theta_b**2 + thetad_b**2)
        return float(np.mean(norms[:, -1]))
    if metric == "final_v":
        return float(np.mean(v_b[:, -1]))
    if metric == "progress":
        norms = np.sqrt(theta_b**2 + thetad_b**2)
        delta = norms[:, :-1] - norms[:, 1:]
        return float(np.mean(np.mean(delta, axis=1)))
    raise ValueError(f"Unknown metric: {metric}")


def _bound_exceedance_fraction_per_episode(
    theta: np.ndarray,
    thetad: np.ndarray,
    baseline_theta: np.ndarray,
    baseline_thetad: np.ndarray,
    dt: float,
    c1: float,
    c2: float,
    c3: float,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Episode-wise fraction of trajectories above norm bound."""
    robust_norm = np.sqrt(theta**2 + thetad**2)  # (E, N, T)
    num_episodes, _, steps = robust_norm.shape
    t = np.arange(steps, dtype=np.float64) * float(dt)

    if baseline_theta.ndim == 2:
        baseline_norm = np.sqrt(baseline_theta**2 + baseline_thetad**2)
    elif baseline_theta.ndim == 3:
        baseline_norm = np.sqrt(baseline_theta**2 + baseline_thetad**2)
    else:
        raise ValueError("baseline theta/thetad must be shape (N,T) or (E,N,T)")

    robust_frac = np.zeros(num_episodes, dtype=np.float64)
    baseline_frac = np.zeros(num_episodes, dtype=np.float64)

    gain = np.sqrt(float(c2) / float(c1))
    decay = np.exp(-0.5 * float(c3) * t)[None, :]  # (1, T)

    for ep in range(num_episodes):
        robust_bound = gain * robust_norm[ep, :, 0][:, None] * decay
        robust_exceeds = np.any(robust_norm[ep, :, 1:] > robust_bound[:, 1:] + tol, axis=1)
        robust_frac[ep] = float(np.mean(robust_exceeds))

        if baseline_norm.ndim == 2:
            baseline_ep = baseline_norm
        else:
            baseline_ep = baseline_norm[ep]
        baseline_bound = gain * baseline_ep[:, 0][:, None] * decay
        baseline_exceeds = np.any(baseline_ep[:, 1:] > baseline_bound[:, 1:] + tol, axis=1)
        baseline_frac[ep] = float(np.mean(baseline_exceeds))

    return robust_frac, baseline_frac


def _v_exceedance_fraction_per_episode(
    v: np.ndarray,
    baseline_v: np.ndarray,
    dt: float,
    c3: float,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Episode-wise fraction of trajectories above V decay bound."""
    num_episodes, _, steps = v.shape
    t = np.arange(steps, dtype=np.float64) * float(dt)

    if baseline_v.ndim == 2:
        baseline_v = baseline_v
    elif baseline_v.ndim == 3:
        baseline_v = baseline_v
    else:
        raise ValueError("baseline_v must be shape (N,T) or (E,N,T)")

    robust_frac = np.zeros(num_episodes, dtype=np.float64)
    baseline_frac = np.zeros(num_episodes, dtype=np.float64)
    decay = np.exp(-float(c3) * t)[None, :]

    for ep in range(num_episodes):
        robust_bound = v[ep, :, 0][:, None] * decay
        robust_exceeds = np.any(v[ep, :, 1:] > robust_bound[:, 1:] + tol, axis=1)
        robust_frac[ep] = float(np.mean(robust_exceeds))

        baseline_ep = baseline_v if baseline_v.ndim == 2 else baseline_v[ep]
        baseline_bound = baseline_ep[:, 0][:, None] * decay
        baseline_exceeds = np.any(baseline_ep[:, 1:] > baseline_bound[:, 1:] + tol, axis=1)
        baseline_frac[ep] = float(np.mean(baseline_exceeds))

    return robust_frac, baseline_frac


def _peak_residual_stats_by_episode(
    residuals: np.ndarray,
    baseline_residuals: np.ndarray,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Peak residual statistics per episode: mean, 10th, and 90th quantiles."""
    if q_low < 0.0 or q_low >= q_high or q_high > 1.0:
        raise ValueError("Require 0 <= q_low < q_high <= 1.")

    r = _as_episode_residual_data(residuals, "residuals")
    b = _as_episode_residual_data(baseline_residuals, "residuals_baseline")
    r_mag = np.linalg.norm(r, axis=-1)  # (E,N,T)
    b_mag = np.linalg.norm(b, axis=-1)  # (E_b,N,T)
    r_peak = np.max(r_mag, axis=2)  # (E,N)
    b_peak = np.max(b_mag, axis=2)

    r_mean = np.mean(r_peak, axis=1)
    r_q10 = np.quantile(r_peak, q_low, axis=1)
    r_q90 = np.quantile(r_peak, q_high, axis=1)

    b_mean = np.mean(b_peak, axis=1)
    b_q10 = np.quantile(b_peak, q_low, axis=1)
    b_q90 = np.quantile(b_peak, q_high, axis=1)

    # If only one baseline collection was run, broadcast its stats to every episode
    # for comparable episode-wise plots against robust/naive episodes.
    num_episodes = int(r_peak.shape[0])
    if b_mean.size == 1 and num_episodes > 1:
        b_mean = np.full(num_episodes, float(b_mean[0]), dtype=np.float64)
        b_q10 = np.full(num_episodes, float(b_q10[0]), dtype=np.float64)
        b_q90 = np.full(num_episodes, float(b_q90[0]), dtype=np.float64)
    elif b_mean.size != num_episodes:
        raise ValueError(
            f"Baseline residual episodes must be either 1 or {num_episodes}; got {b_mean.size}."
        )

    return r_mean, r_q10, r_q90, b_mean, b_q10, b_q90


def _plot_fraction_metric(
    robust_frac: np.ndarray,
    naive_frac: np.ndarray,
    nonrobust_frac: np.ndarray,
    y_label: str,
    title: str,
    out_path: str,
    show: bool,
    calibrate_frac: np.ndarray | None = None,
    alpha_reference: float | None = None,
) -> None:
    n = min(robust_frac.size, naive_frac.size, nonrobust_frac.size)
    if calibrate_frac is not None:
        n = min(n, int(np.asarray(calibrate_frac).size))
    episodes = np.arange(n)
    if calibrate_frac is not None:
        calibrate_frac = np.asarray(calibrate_frac, dtype=np.float64)[:n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episodes, robust_frac[:n], marker="o", linewidth=2.0, color="tab:orange", label="Robust")
    ax.plot(episodes, naive_frac[:n], marker="s", linewidth=2.0, color="tab:green", label="Naive")
    ax.plot(episodes, nonrobust_frac[:n], marker="d", linewidth=2.0, color="tab:blue", label="Nonrobust")
    if calibrate_frac is not None:
        ax.plot(
            episodes,
            calibrate_frac,
            marker="^",
            linewidth=2.0,
            color="tab:purple",
            label="Calibrate once",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    if alpha_reference is not None:
        ax.axhline(float(alpha_reference), linestyle=":", linewidth=2.0, color="k", label=r"$\alpha$")
    _set_integer_episode_ticks(ax)
    ax.legend(loc="best")
    _save_or_show(fig, out_path, show)


def _plot_residual_peak_metric(
    robust_mean: np.ndarray,
    robust_q10: np.ndarray,
    robust_q90: np.ndarray,
    naive_mean: np.ndarray,
    naive_q10: np.ndarray,
    naive_q90: np.ndarray,
    nonrobust_mean: np.ndarray,
    nonrobust_q10: np.ndarray,
    nonrobust_q90: np.ndarray,
    out_path: str,
    show: bool,
    calibrate_mean: np.ndarray | None = None,
) -> None:
    """Plot peak residual with 10/90 quantile error bars."""
    n = min(
        robust_mean.size,
        robust_q10.size,
        robust_q90.size,
        naive_mean.size,
        naive_q10.size,
        naive_q90.size,
        nonrobust_mean.size,
        nonrobust_q10.size,
        nonrobust_q90.size,
    )
    if calibrate_mean is not None:
        n = min(n, int(np.asarray(calibrate_mean).size))
    episodes = np.arange(n)

    fig, ax = plt.subplots(figsize=(8, 5))
    rerr = np.vstack([
        np.maximum(0.0, robust_mean[:n] - robust_q10[:n]),
        np.maximum(0.0, robust_q90[:n] - robust_mean[:n]),
    ])
    nerr = np.vstack([
        np.maximum(0.0, naive_mean[:n] - naive_q10[:n]),
        np.maximum(0.0, naive_q90[:n] - naive_mean[:n]),
    ])
    berr = np.vstack([
        np.maximum(0.0, nonrobust_mean[:n] - nonrobust_q10[:n]),
        np.maximum(0.0, nonrobust_q90[:n] - nonrobust_mean[:n]),
    ])

    ax.errorbar(
        episodes,
        robust_mean[:n],
        yerr=rerr,
        marker="o",
        linewidth=2.0,
        capsize=4,
        color="tab:orange",
        label="Robust",
    )
    ax.errorbar(
        episodes,
        naive_mean[:n],
        yerr=nerr,
        marker="s",
        linewidth=2.0,
        capsize=4,
        color="tab:green",
        label="Naive",
    )
    ax.errorbar(
        episodes,
        nonrobust_mean[:n],
        yerr=berr,
        marker="d",
        linewidth=2.0,
        capsize=4,
        color="tab:blue",
        label="Nonrobust",
    )
    if calibrate_mean is not None:
        calibrate_mean = np.asarray(calibrate_mean, dtype=np.float64)[:n]
        ax.plot(
            episodes,
            calibrate_mean,
            marker="^",
            linewidth=2.0,
            color="tab:purple",
            label="Calibrate once",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel(r"max$_{t} \| \hat f(x_t,u_t)-f(x_t,u_t)\|_2$")
    ax.set_title("Residual Peak vs Episode")
    _set_integer_episode_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _save_or_show(fig, out_path, show)


def _plot_metric(
    robust_metric: np.ndarray,
    naive_metric: np.ndarray,
    nonrobust_scalar: float,
    y_label: str,
    title: str,
    out_path: str,
    show: bool,
    robust_metric_q10: np.ndarray | None = None,
    robust_metric_q90: np.ndarray | None = None,
    naive_metric_q10: np.ndarray | None = None,
    naive_metric_q90: np.ndarray | None = None,
    calibrate_metric: np.ndarray | None = None,
) -> None:
    if robust_metric_q10 is None:
        robust_metric_q10 = np.zeros_like(robust_metric)
    if robust_metric_q90 is None:
        robust_metric_q90 = np.zeros_like(robust_metric)
    if naive_metric_q10 is None:
        naive_metric_q10 = np.zeros_like(naive_metric)
    if naive_metric_q90 is None:
        naive_metric_q90 = np.zeros_like(naive_metric)

    n = min(
        robust_metric.size,
        naive_metric.size,
        robust_metric_q10.size,
        robust_metric_q90.size,
        naive_metric_q10.size,
        naive_metric_q90.size,
    )
    if calibrate_metric is not None:
        n = min(n, int(np.asarray(calibrate_metric).size))
    episodes = np.arange(n)
    robust_metric = np.asarray(robust_metric, dtype=np.float64)[:n]
    naive_metric = np.asarray(naive_metric, dtype=np.float64)[:n]
    robust_metric_q10 = np.asarray(robust_metric_q10, dtype=np.float64)[:n]
    robust_metric_q90 = np.asarray(robust_metric_q90, dtype=np.float64)[:n]
    naive_metric_q10 = np.asarray(naive_metric_q10, dtype=np.float64)[:n]
    naive_metric_q90 = np.asarray(naive_metric_q90, dtype=np.float64)[:n]
    if calibrate_metric is not None:
        calibrate_metric = np.asarray(calibrate_metric, dtype=np.float64)[:n]

    fig, ax = plt.subplots(figsize=(8, 5))
    robust_err = np.vstack([
        np.maximum(0.0, robust_metric - robust_metric_q10),
        np.maximum(0.0, robust_metric_q90 - robust_metric),
    ])
    naive_err = np.vstack([
        np.maximum(0.0, naive_metric - naive_metric_q10),
        np.maximum(0.0, naive_metric_q90 - naive_metric),
    ])
    ax.errorbar(
        episodes,
        robust_metric,
        yerr=robust_err,
        marker="o",
        linewidth=2.0,
        capsize=4,
        color="tab:orange",
        label="Robust",
    )
    ax.errorbar(
        episodes,
        naive_metric,
        yerr=naive_err,
        marker="s",
        linewidth=2.0,
        capsize=4,
        color="tab:green",
        label="Naive",
    )
    nonrobust_series = np.full(n, nonrobust_scalar, dtype=np.float64)
    ax.plot(
        episodes,
        nonrobust_series,
        marker="d",
        linewidth=2.0,
        color="tab:blue",
        label="Nonrobust",
    )
    if calibrate_metric is not None:
        ax.plot(
            episodes,
            calibrate_metric,
            marker="^",
            linewidth=2.0,
            color="tab:purple",
            label="Calibrate once",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    _set_integer_episode_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _save_or_show(fig, out_path, show)


def plot_combined_rq(
    robust_r: np.ndarray,
    robust_q: np.ndarray,
    naive_r: np.ndarray,
    naive_q: np.ndarray,
    out_path: str,
    show: bool,
    calibrate_r: np.ndarray | None = None,
    calibrate_q: np.ndarray | None = None,
) -> None:
    """Plot robust/naive r_j and q_j together on one episode axis."""
    n = min(robust_r.size, naive_r.size, robust_q.size, naive_q.size)
    if calibrate_r is not None:
        n = min(n, int(np.asarray(calibrate_r).size))
    if calibrate_q is not None:
        n = min(n, int(np.asarray(calibrate_q).size))
    episodes = np.arange(n)
    if calibrate_r is not None:
        calibrate_r = np.asarray(calibrate_r, dtype=np.float64)[:n]
    if calibrate_q is not None:
        calibrate_q = np.asarray(calibrate_q, dtype=np.float64)[:n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        episodes,
        robust_r[:n],
        marker="o",
        linewidth=2.0,
        color="tab:orange",
        label="Robust r_j",
    )
    ax.plot(
        episodes,
        robust_q[:n],
        marker="o",
        linewidth=2.0,
        color="tab:red",
        label="Robust q_j",
    )
    ax.plot(
        episodes,
        naive_r[:n],
        marker="s",
        linewidth=2.0,
        color="tab:green",
        label="Naive r_j",
    )
    ax.plot(
        episodes,
        naive_q[:n],
        marker="s",
        linewidth=2.0,
        color="tab:purple",
        label="Naive q_j",
    )
    # Requested: nonrobust r_j baseline as a regular line (not dotted benchmark).
    ax.plot(
        episodes,
        np.zeros(n, dtype=np.float64),
        linewidth=2.0,
        color="tab:blue",
        label="Nonrobust r_j (=0)",
    )
    if calibrate_r is not None:
        ax.plot(
            episodes,
            calibrate_r,
            linewidth=2.0,
            marker="^",
            color="tab:brown",
            label="Calibrate once r_j",
        )
    if calibrate_q is not None:
        ax.plot(
            episodes,
            calibrate_q,
            linewidth=2.0,
            marker="^",
            color="tab:gray",
            label="Calibrate once q_j",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.set_title("r_j and q_j vs Episode (Robust vs Naive)")
    _set_integer_episode_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _save_or_show(fig, out_path, show)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    robust = np.load(args.robust_input)
    naive = np.load(args.naive_input)

    robust_r = np.asarray(robust["r_j"], dtype=np.float64)
    robust_q = np.asarray(robust["q_j"], dtype=np.float64)
    naive_r = np.asarray(naive["r_j"], dtype=np.float64)
    naive_q = np.asarray(naive["q_j"], dtype=np.float64)

    robust_theta = np.asarray(robust["theta"], dtype=np.float64)
    robust_thetad = np.asarray(robust["thetad"], dtype=np.float64)
    robust_v = np.asarray(robust["v"], dtype=np.float64)
    robust_residuals = np.asarray(robust["residuals"], dtype=np.float64)

    naive_theta = np.asarray(naive["theta"], dtype=np.float64)
    naive_thetad = np.asarray(naive["thetad"], dtype=np.float64)
    naive_v = np.asarray(naive["v"], dtype=np.float64)
    naive_residuals = np.asarray(naive["residuals"], dtype=np.float64)

    robust_theta_b = np.asarray(robust["theta_baseline"], dtype=np.float64)
    robust_thetad_b = np.asarray(robust["thetad_baseline"], dtype=np.float64)
    robust_v_b = np.asarray(robust["v_baseline"], dtype=np.float64)
    robust_res_b = np.asarray(robust["residuals_baseline"], dtype=np.float64)

    naive_theta_b = np.asarray(naive["theta_baseline"], dtype=np.float64)
    naive_thetad_b = np.asarray(naive["thetad_baseline"], dtype=np.float64)
    naive_v_b = np.asarray(naive["v_baseline"], dtype=np.float64)
    naive_res_b = np.asarray(naive["residuals_baseline"], dtype=np.float64)

    robust_alpha_bar = float(robust["alpha_bar"]) if "alpha_bar" in robust.files else 0.1
    naive_alpha_bar = float(naive["alpha_bar"]) if "alpha_bar" in naive.files else 0.1
    dt = float(robust["dt"]) if "dt" in robust.files else float(naive["dt"])
    c1 = float(robust["c1"]) if "c1" in robust.files else float(naive["c1"])
    c2 = float(robust["c2"]) if "c2" in robust.files else float(naive["c2"])
    c3 = float(robust["c3"]) if "c3" in robust.files else float(naive["c3"])
    exceedance_alpha_ref = 0.5 * (robust_alpha_bar + naive_alpha_bar)

    max_episodes = int(min(len(robust_theta), len(naive_theta), len(robust_thetad), len(naive_thetad), len(robust_v), len(naive_v)))
    calibrate_r = _calibrate_once_series(naive_r, max_episodes)
    calibrate_q = _calibrate_once_series(naive_q, max_episodes)
    calibrate_theta = _calibrate_once_series(naive_theta, max_episodes)
    calibrate_thetad = _calibrate_once_series(naive_thetad, max_episodes)
    calibrate_v = _calibrate_once_series(naive_v, max_episodes)
    calibrate_residuals = _calibrate_once_series(naive_residuals, max_episodes)

    plot_combined_rq(
        robust_r=robust_r,
        robust_q=robust_q,
        naive_r=naive_r,
        naive_q=naive_q,
        calibrate_r=calibrate_r,
        calibrate_q=calibrate_q,
        out_path=os.path.join(args.out_dir, "rj_qj_combined.png"),
        show=args.show,
    )

    _plot_episode_comparison(
        robust_data=robust_theta,
        naive_data=naive_theta,
        baseline_data_robust=robust_theta_b,
        baseline_data_naive=naive_theta_b,
        calibrate_data=calibrate_theta,
        dt=dt,
        y_label="theta (rad)",
        file_stem="theta",
        out_dir=args.out_dir,
        show=args.show,
    )
    _plot_episode_comparison(
        robust_data=robust_thetad,
        naive_data=naive_thetad,
        baseline_data_robust=robust_thetad_b,
        baseline_data_naive=naive_thetad_b,
        calibrate_data=calibrate_thetad,
        dt=dt,
        y_label="theta_dot (rad/s)",
        file_stem="thetad",
        out_dir=args.out_dir,
        show=args.show,
    )
    _plot_episode_comparison(
        robust_data=robust_v,
        naive_data=naive_v,
        baseline_data_robust=robust_v_b,
        baseline_data_naive=naive_v_b,
        calibrate_data=calibrate_v,
        dt=dt,
        y_label="V(x_t)",
        file_stem="V",
        out_dir=args.out_dir,
        show=args.show,
        exp_decay_c3=c3,
    )
    _plot_norm_episode_comparison(
        robust_theta=robust_theta,
        robust_thetad=robust_thetad,
        naive_theta=naive_theta,
        naive_thetad=naive_thetad,
        baseline_theta_robust=robust_theta_b,
        baseline_thetad_robust=robust_thetad_b,
        baseline_theta_naive=naive_theta_b,
        baseline_thetad_naive=naive_thetad_b,
        dt=dt,
        out_dir=args.out_dir,
        show=args.show,
        calibrate_norm_naive=np.sqrt(calibrate_theta**2 + calibrate_thetad**2),
        c1=c1,
        c2=c2,
        c3=c3,
    )

    calibrate_final_norm = _final_state_norm_metric(calibrate_theta, calibrate_thetad)
    calibrate_final_v = _final_v_metric(calibrate_v)
    calibrate_progress = _progress_metric(calibrate_theta, calibrate_thetad)

    robust_final_norm, robust_final_norm_q10, robust_final_norm_q90 = _final_state_norm_metric_stats(
        robust_theta,
        robust_thetad,
    )
    naive_final_norm, naive_final_norm_q10, naive_final_norm_q90 = _final_state_norm_metric_stats(
        naive_theta,
        naive_thetad,
    )
    nonrobust_final_norm = 0.5 * (
        _baseline_scalar_metric(robust_theta_b, robust_thetad_b, robust_v_b, "final_state_norm")
        + _baseline_scalar_metric(naive_theta_b, naive_thetad_b, naive_v_b, "final_state_norm")
    )
    _plot_metric(
        robust_metric=robust_final_norm,
        naive_metric=naive_final_norm,
        robust_metric_q10=robust_final_norm_q10,
        robust_metric_q90=robust_final_norm_q90,
        naive_metric_q10=naive_final_norm_q10,
        naive_metric_q90=naive_final_norm_q90,
        nonrobust_scalar=nonrobust_final_norm,
        calibrate_metric=calibrate_final_norm,
        y_label=r"Avg final ||[theta, \dot{theta}]||",
        title="Average Final State Norm vs Episode",
        out_path=os.path.join(args.out_dir, "final_state_norm_vs_episode_combined.png"),
        show=args.show,
    )

    robust_final_v, robust_final_v_q10, robust_final_v_q90 = _final_v_metric_stats(robust_v)
    naive_final_v, naive_final_v_q10, naive_final_v_q90 = _final_v_metric_stats(naive_v)
    nonrobust_final_v = 0.5 * (
        _baseline_scalar_metric(robust_theta_b, robust_thetad_b, robust_v_b, "final_v")
        + _baseline_scalar_metric(naive_theta_b, naive_thetad_b, naive_v_b, "final_v")
    )
    _plot_metric(
        robust_metric=robust_final_v,
        naive_metric=naive_final_v,
        robust_metric_q10=robust_final_v_q10,
        robust_metric_q90=robust_final_v_q90,
        naive_metric_q10=naive_final_v_q10,
        naive_metric_q90=naive_final_v_q90,
        nonrobust_scalar=nonrobust_final_v,
        calibrate_metric=calibrate_final_v,
        y_label="Avg final V(x_T)",
        title="Average Final CLF Value vs Episode",
        out_path=os.path.join(args.out_dir, "final_v_vs_episode_combined.png"),
        show=args.show,
    )

    robust_progress, robust_progress_q10, robust_progress_q90 = _progress_metric_stats(robust_theta, robust_thetad)
    naive_progress, naive_progress_q10, naive_progress_q90 = _progress_metric_stats(naive_theta, naive_thetad)
    nonrobust_progress = 0.5 * (
        _baseline_scalar_metric(robust_theta_b, robust_thetad_b, robust_v_b, "progress")
        + _baseline_scalar_metric(naive_theta_b, naive_thetad_b, naive_v_b, "progress")
    )
    _plot_metric(
        robust_metric=robust_progress,
        naive_metric=naive_progress,
        robust_metric_q10=robust_progress_q10,
        robust_metric_q90=robust_progress_q90,
        naive_metric_q10=naive_progress_q10,
        naive_metric_q90=naive_progress_q90,
        nonrobust_scalar=nonrobust_progress,
        calibrate_metric=calibrate_progress,
        y_label=r"Avg progress: ||x_{t-1}|| - ||x_t||",
        title="Average Progress Metric vs Episode",
        out_path=os.path.join(args.out_dir, "progress_vs_episode_combined.png"),
        show=args.show,
    )

    robust_r_mean, robust_r_q10, robust_r_q90, robust_b_mean, robust_b_q10, robust_b_q90 = _peak_residual_stats_by_episode(
        residuals=robust_residuals,
        baseline_residuals=robust_res_b,
    )
    naive_r_mean, naive_r_q10, naive_r_q90, naive_b_mean, naive_b_q10, naive_b_q90 = _peak_residual_stats_by_episode(
        residuals=naive_residuals,
        baseline_residuals=naive_res_b,
    )
    calibrate_r_mean = _peak_residual_stats_by_episode(
        residuals=calibrate_residuals,
        baseline_residuals=calibrate_residuals,
    )[0]
    n_res = min(robust_r_mean.size, naive_r_mean.size, robust_b_mean.size, naive_b_mean.size)
    nonrobust_r_mean = 0.5 * (_align_baseline_series(robust_b_mean, n_res) + _align_baseline_series(naive_b_mean, n_res))
    nonrobust_r_q10 = 0.5 * (_align_baseline_series(robust_b_q10, n_res) + _align_baseline_series(naive_b_q10, n_res))
    nonrobust_r_q90 = 0.5 * (_align_baseline_series(robust_b_q90, n_res) + _align_baseline_series(naive_b_q90, n_res))
    _plot_residual_peak_metric(
        robust_mean=robust_r_mean[:n_res],
        robust_q10=robust_r_q10[:n_res],
        robust_q90=robust_r_q90[:n_res],
        naive_mean=naive_r_mean[:n_res],
        naive_q10=naive_r_q10[:n_res],
        naive_q90=naive_r_q90[:n_res],
        nonrobust_mean=nonrobust_r_mean,
        nonrobust_q10=nonrobust_r_q10,
        nonrobust_q90=nonrobust_r_q90,
        calibrate_mean=calibrate_r_mean[:n_res],
        out_path=os.path.join(args.out_dir, "residual_peak_vs_episode_combined.png"),
        show=args.show,
    )

    robust_bound_frac, robust_baseline_bound_frac = _bound_exceedance_fraction_per_episode(
        theta=robust_theta,
        thetad=robust_thetad,
        baseline_theta=robust_theta_b,
        baseline_thetad=robust_thetad_b,
        dt=dt,
        c1=c1,
        c2=c2,
        c3=c3,
    )
    naive_bound_frac, naive_baseline_bound_frac = _bound_exceedance_fraction_per_episode(
        theta=naive_theta,
        thetad=naive_thetad,
        baseline_theta=naive_theta_b,
        baseline_thetad=naive_thetad_b,
        dt=dt,
        c1=c1,
        c2=c2,
        c3=c3,
    )
    n_bound = min(robust_bound_frac.size, naive_bound_frac.size, robust_baseline_bound_frac.size, naive_baseline_bound_frac.size)
    nonrobust_bound = 0.5 * (
        _align_baseline_series(robust_baseline_bound_frac, n_bound)
        + _align_baseline_series(naive_baseline_bound_frac, n_bound)
    )
    calibrate_norm_frac, _ = _bound_exceedance_fraction_per_episode(
        theta=calibrate_theta,
        thetad=calibrate_thetad,
        baseline_theta=robust_theta_b,
        baseline_thetad=robust_thetad_b,
        dt=dt,
        c1=c1,
        c2=c2,
        c3=c3,
    )
    _plot_fraction_metric(
        robust_frac=robust_bound_frac[:n_bound],
        naive_frac=naive_bound_frac[:n_bound],
        nonrobust_frac=nonrobust_bound,
        calibrate_frac=calibrate_norm_frac[:n_bound],
        y_label="Fraction trajectories above norm bound",
        title="Fraction Above Norm Bound vs Episode",
        out_path=os.path.join(args.out_dir, "bound_exceedance_fraction_vs_episode_combined.png"),
        show=args.show,
        alpha_reference=exceedance_alpha_ref,
    )

    robust_v_frac, robust_baseline_v_frac = _v_exceedance_fraction_per_episode(
        v=robust_v,
        baseline_v=robust_v_b,
        dt=dt,
        c3=c3,
    )
    naive_v_frac, naive_baseline_v_frac = _v_exceedance_fraction_per_episode(
        v=naive_v,
        baseline_v=naive_v_b,
        dt=dt,
        c3=c3,
    )
    n_v = min(robust_v_frac.size, naive_v_frac.size, robust_baseline_v_frac.size, naive_baseline_v_frac.size)
    nonrobust_v = 0.5 * (
        _align_baseline_series(robust_baseline_v_frac, n_v)
        + _align_baseline_series(naive_baseline_v_frac, n_v)
    )
    calibrate_v_frac, _ = _v_exceedance_fraction_per_episode(
        v=calibrate_v,
        baseline_v=robust_v_b,
        dt=dt,
        c3=c3,
    )
    _plot_fraction_metric(
        robust_frac=robust_v_frac[:n_v],
        naive_frac=naive_v_frac[:n_v],
        nonrobust_frac=nonrobust_v,
        calibrate_frac=calibrate_v_frac[:n_v],
        y_label="Fraction trajectories above V bound",
        title="Fraction Above V Bound vs Episode",
        out_path=os.path.join(args.out_dir, "v_exceedance_fraction_vs_episode_combined.png"),
        show=args.show,
        alpha_reference=exceedance_alpha_ref,
    )

    if not args.show:
        print(f"Saved combined plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
