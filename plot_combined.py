"""Combined episode-level plots from robust and naive SR-CR runs.

This script loads two rollout .npz files:
- robust run from main.py
- naive run from main_naive.py

It plots episode-level comparisons for:
1) r_j and q_j
2) average final ||[theta, theta_dot]||
3) average final V(x_T)
4) average progress metric: ||x_{t-1}|| - ||x_t||

Nonrobust line is built by averaging baseline (r=0) statistics from both files.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

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
        fig.savefig(path, dpi=150)
        plt.close(fig)


def _trajectory_scores_from_residuals(residuals: np.ndarray) -> np.ndarray:
    """Return per-trajectory max residual norm scores."""
    # residuals shape can be (N, T, 2) or (E, N, T, 2)
    if residuals.ndim == 4:
        residuals = residuals.reshape(-1, residuals.shape[2], residuals.shape[3])
    norms = np.linalg.norm(residuals, axis=2)  # (N, T)
    return np.max(norms, axis=1)


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


def _final_v_metric(v: np.ndarray) -> np.ndarray:
    return np.mean(v[:, :, -1], axis=1)


def _progress_metric(theta: np.ndarray, thetad: np.ndarray) -> np.ndarray:
    norms = np.sqrt(theta**2 + thetad**2)  # (E, N, T)
    delta = norms[:, :, :-1] - norms[:, :, 1:]  # (E, N, T-1)
    return np.mean(np.mean(delta, axis=2), axis=1)


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


def _plot_metric(
    robust_metric: np.ndarray,
    naive_metric: np.ndarray,
    nonrobust_scalar: float,
    y_label: str,
    title: str,
    out_path: str,
    show: bool,
) -> None:
    n = min(robust_metric.size, naive_metric.size)
    episodes = np.arange(n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episodes, robust_metric[:n], marker="o", linewidth=2.0, color="tab:orange", label="Robust")
    ax.plot(episodes, naive_metric[:n], marker="s", linewidth=2.0, color="tab:green", label="Naive")
    ax.axhline(nonrobust_scalar, linestyle="--", linewidth=2.0, color="tab:blue", label="Nonrobust (avg)")
    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    ax.set_title(title)
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
) -> None:
    """Plot robust/naive r_j and q_j together on one episode axis."""
    n = min(robust_r.size, naive_r.size, robust_q.size, naive_q.size)
    episodes = np.arange(n)

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

    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.set_title("r_j and q_j vs Episode (Robust vs Naive)")
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

    naive_theta = np.asarray(naive["theta"], dtype=np.float64)
    naive_thetad = np.asarray(naive["thetad"], dtype=np.float64)
    naive_v = np.asarray(naive["v"], dtype=np.float64)

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

    plot_combined_rq(
        robust_r=robust_r,
        robust_q=robust_q,
        naive_r=naive_r,
        naive_q=naive_q,
        out_path=os.path.join(args.out_dir, "rj_qj_combined.png"),
        show=args.show,
    )

    robust_final_norm = _final_state_norm_metric(robust_theta, robust_thetad)
    naive_final_norm = _final_state_norm_metric(naive_theta, naive_thetad)
    nonrobust_final_norm = 0.5 * (
        _baseline_scalar_metric(robust_theta_b, robust_thetad_b, robust_v_b, "final_state_norm")
        + _baseline_scalar_metric(naive_theta_b, naive_thetad_b, naive_v_b, "final_state_norm")
    )
    _plot_metric(
        robust_metric=robust_final_norm,
        naive_metric=naive_final_norm,
        nonrobust_scalar=nonrobust_final_norm,
        y_label=r"Avg final ||[theta, \dot{theta}]||",
        title="Average Final State Norm vs Episode",
        out_path=os.path.join(args.out_dir, "final_state_norm_vs_episode_combined.png"),
        show=args.show,
    )

    robust_final_v = _final_v_metric(robust_v)
    naive_final_v = _final_v_metric(naive_v)
    nonrobust_final_v = 0.5 * (
        _baseline_scalar_metric(robust_theta_b, robust_thetad_b, robust_v_b, "final_v")
        + _baseline_scalar_metric(naive_theta_b, naive_thetad_b, naive_v_b, "final_v")
    )
    _plot_metric(
        robust_metric=robust_final_v,
        naive_metric=naive_final_v,
        nonrobust_scalar=nonrobust_final_v,
        y_label="Avg final V(x_T)",
        title="Average Final CLF Value vs Episode",
        out_path=os.path.join(args.out_dir, "final_v_vs_episode_combined.png"),
        show=args.show,
    )

    robust_progress = _progress_metric(robust_theta, robust_thetad)
    naive_progress = _progress_metric(naive_theta, naive_thetad)
    nonrobust_progress = 0.5 * (
        _baseline_scalar_metric(robust_theta_b, robust_thetad_b, robust_v_b, "progress")
        + _baseline_scalar_metric(naive_theta_b, naive_thetad_b, naive_v_b, "progress")
    )
    _plot_metric(
        robust_metric=robust_progress,
        naive_metric=naive_progress,
        nonrobust_scalar=nonrobust_progress,
        y_label=r"Avg progress: ||x_{t-1}|| - ||x_t||",
        title="Average Progress Metric vs Episode",
        out_path=os.path.join(args.out_dir, "progress_vs_episode_combined.png"),
        show=args.show,
    )

    if not args.show:
        print(f"Saved combined plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
