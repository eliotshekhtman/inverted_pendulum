"""Collect transition data and fit polynomial nominal model weights Mf and Mg.

Model form:
    phi(x) = InvertedPendulum.features(x)
    x_dot_nom(x, u) = Mf @ phi(x) + (Mg @ phi(x)) * u

We fit Mf and Mg by least squares on random one-step transitions generated from
true dynamics.
"""

from __future__ import annotations

import argparse

import numpy as np

from clf_controller import RobustCLFController
from pendulum_env import InvertedPendulum
from state_sampling import sample_state_for_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit learned nominal dynamics for inverted pendulum.")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--theta_bound", type=float, default=float(np.pi / 2.0))
    parser.add_argument("--thetad_bound", type=float, default=10.0)
    parser.add_argument("--u_min", type=float, default=-7.0)
    parser.add_argument("--u_max", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="nominal_model_weights.npz")
    return parser.parse_args()


def collect_random_transitions(
    env: InvertedPendulum,
    P_for_sampling: np.ndarray,
    num_samples: int,
    dt: float,
    theta_bound: float,
    thetad_bound: float,
    u_min: float,
    u_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random (x, u, x_next) with training sampler V(x) <= 1.3."""
    xs = np.zeros((num_samples, 2), dtype=np.float64)
    us = np.zeros((num_samples,), dtype=np.float64)
    x_next = np.zeros((num_samples, 2), dtype=np.float64)

    for i in range(num_samples):
        x = sample_state_for_training(
            P=P_for_sampling,
            rng=rng,
            theta_bound=theta_bound,
            thetad_bound=thetad_bound,
        )
        u = float(rng.uniform(u_min, u_max))
        t = float(rng.uniform(0.0, 5.0))

        xs[i] = x
        us[i] = u
        x_next[i] = env.step(x, u, t, dt)

    return xs, us, x_next


def fit_mf_mg(
    xs: np.ndarray, us: np.ndarray, x_next: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Fit Mf and Mg with least squares on x_dot ≈ Mf*phi + (Mg*phi)*u."""
    phi = np.array([InvertedPendulum.features(x) for x in xs], dtype=np.float64)  # (n, m)
    phi_dim = int(phi.shape[1])
    z = np.concatenate([phi, phi * us[:, None]], axis=1)  # (n, 2m)

    x_dot = (x_next - xs) / float(dt)  # (n, 2)

    # Solve z @ W^T ≈ x_dot, where W = [Mf | Mg] has shape (2, 2m).
    coef, _, _, _ = np.linalg.lstsq(z, x_dot, rcond=None)  # (2m, 2)
    W = coef.T  # (2, 2m)
    Mf = W[:, :phi_dim]
    Mg = W[:, phi_dim:]

    x_dot_hat = z @ coef
    residuals = x_dot - x_dot_hat
    mse = float(np.mean((x_dot_hat - x_dot) ** 2))
    return Mf, Mg, mse, residuals


def mismatch_stats(residuals: np.ndarray) -> tuple[float, float, float]:
    """Return r-style residual norm summaries: max, 90th percentile, mean."""
    residuals = np.asarray(residuals, dtype=np.float64)
    norms = np.linalg.norm(residuals, axis=1)
    return float(np.max(norms)), float(np.percentile(norms, 90.0)), float(np.mean(norms))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    env_true = InvertedPendulum(
        g=9.81,
        m=1.0,
        l=1.0,
        b=0.01,
        u_min=args.u_min,
        u_max=args.u_max,
        disturbance_amp=0.0,
    )
    controller_for_sampling = RobustCLFController(
        env=env_true,
        k_fb=(8.0, 5.0),
        c3=0.5,
        use_robust_term=True,
        auto_select_k=True,
        stability_margin=1e-3,
        max_p_condition=1e7,
    )

    xs, us, x_next = collect_random_transitions(
        env=env_true,
        P_for_sampling=controller_for_sampling.P,
        num_samples=args.num_samples,
        dt=args.dt,
        theta_bound=args.theta_bound,
        thetad_bound=args.thetad_bound,
        u_min=args.u_min,
        u_max=args.u_max,
        rng=rng,
    )

    Mf, Mg, mse, residuals = fit_mf_mg(xs, us, x_next, args.dt)
    mismatch_max, mismatch_p90, mismatch_mean = mismatch_stats(residuals)

    np.savez(
        args.output,
        Mf=Mf,
        Mg=Mg,
        dt=float(args.dt),
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        mse=float(mse),
        mismatch_max=float(mismatch_max),
        mismatch_p90=float(mismatch_p90),
        mismatch_mean=float(mismatch_mean),
    )

    print(f"Saved learned nominal weights to: {args.output}")
    print(f"Fit MSE on x_dot: {mse:.8f}")
    print(
        "Training residual norm stats "
        f"(max / p90 / mean): {mismatch_max:.8f} / {mismatch_p90:.8f} / {mismatch_mean:.8f}"
    )
    print("Mf=")
    print(Mf)
    print("Mg=")
    print(Mg)


if __name__ == "__main__":
    main()
