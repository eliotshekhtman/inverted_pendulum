"""Collect transition data and fit polynomial nominal model weights Mf and Mg.

Model form:
    phi(x) = [1, theta, theta_dot, theta^2, theta*theta_dot, theta_dot^2]
    x_dot_nom(x, u) = Mf @ phi(x) + (Mg @ phi(x)) * u

We fit Mf and Mg by least squares on random one-step transitions generated from
true dynamics.
"""

from __future__ import annotations

import argparse

import numpy as np

from pendulum_env import InvertedPendulum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit learned nominal dynamics for inverted pendulum.")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--theta_min", type=float, default=-1.0)
    parser.add_argument("--theta_max", type=float, default=1.0)
    parser.add_argument("--thetad_min", type=float, default=-3.0)
    parser.add_argument("--thetad_max", type=float, default=3.0)
    parser.add_argument("--u_min", type=float, default=-7.0)
    parser.add_argument("--u_max", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="nominal_model_weights.npz")
    return parser.parse_args()


def collect_random_transitions(
    env: InvertedPendulum,
    num_samples: int,
    dt: float,
    theta_min: float,
    theta_max: float,
    thetad_min: float,
    thetad_max: float,
    u_min: float,
    u_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random (x, u, x_next) samples using true dynamics integration."""
    xs = np.zeros((num_samples, 2), dtype=np.float64)
    us = np.zeros((num_samples,), dtype=np.float64)
    x_next = np.zeros((num_samples, 2), dtype=np.float64)

    for i in range(num_samples):
        x = np.array(
            [
                rng.uniform(theta_min, theta_max),
                rng.uniform(thetad_min, thetad_max),
            ],
            dtype=np.float64,
        )
        u = float(rng.uniform(u_min, u_max))
        t = float(rng.uniform(0.0, 5.0))

        xs[i] = x
        us[i] = u
        x_next[i] = env.step(x, u, t, dt)

    return xs, us, x_next


def fit_mf_mg(xs: np.ndarray, us: np.ndarray, x_next: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit Mf and Mg with least squares on x_dot ≈ Mf*phi + (Mg*phi)*u."""
    n = xs.shape[0]

    phi = np.array([InvertedPendulum.features(x) for x in xs], dtype=np.float64)  # (n, 6)
    z = np.concatenate([phi, phi * us[:, None]], axis=1)  # (n, 12)

    x_dot = (x_next - xs) / float(dt)  # (n, 2)

    # Solve z @ W^T ≈ x_dot, where W = [Mf | Mg] has shape (2, 12).
    coef, _, _, _ = np.linalg.lstsq(z, x_dot, rcond=None)  # (12, 2)
    W = coef.T  # (2, 12)
    Mf = W[:, :6]
    Mg = W[:, 6:]

    x_dot_hat = z @ coef
    mse = float(np.mean((x_dot_hat - x_dot) ** 2))
    return Mf, Mg, mse


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

    xs, us, x_next = collect_random_transitions(
        env=env_true,
        num_samples=args.num_samples,
        dt=args.dt,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        thetad_min=args.thetad_min,
        thetad_max=args.thetad_max,
        u_min=args.u_min,
        u_max=args.u_max,
        rng=rng,
    )

    Mf, Mg, mse = fit_mf_mg(xs, us, x_next, args.dt)

    np.savez(
        args.output,
        Mf=Mf,
        Mg=Mg,
        dt=float(args.dt),
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        mse=float(mse),
    )

    print(f"Saved learned nominal weights to: {args.output}")
    print(f"Fit MSE on x_dot: {mse:.8f}")
    print("Mf=")
    print(Mf)
    print("Mg=")
    print(Mg)


if __name__ == "__main__":
    main()
