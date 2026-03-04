"""Robust CLF-QP controller for inverted pendulum stabilization."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from pendulum_env import InvertedPendulum


class RobustCLFController:
    """Softened robust CLF-QP controller with CARE-based Lyapunov matrix."""

    def __init__(
        self,
        env: InvertedPendulum,
        q_diag: tuple[float, float] = (10.0, 1.0),
        r_val: float = 1.0,
        penalty_sigma: float = 1000.0,
        decay_rate: float = 1.0,
        u_min: float = -15.0,
        u_max: float = 15.0,
    ) -> None:
        self.env = env
        self.Q = np.diag(np.asarray(q_diag, dtype=np.float64))
        self.R = np.array([[float(r_val)]], dtype=np.float64)
        self.penalty_sigma = float(penalty_sigma)
        self.decay_rate = float(decay_rate)
        self.u_min = float(u_min)
        self.u_max = float(u_max)

        self.A, self.B = self._linearized_nominal_dynamics()
        self.P = self._solve_care_hamiltonian(self.A, self.B, self.Q, self.R)

    def _linearized_nominal_dynamics(self) -> tuple[np.ndarray, np.ndarray]:
        """Linearize nominal dynamics around upright equilibrium x = [0, 0]."""
        a21 = self.env.g / self.env.l_hat
        b2 = 1.0 / (self.env.m_hat * self.env.l_hat**2)

        A = np.array(
            [
                [0.0, 1.0],
                [a21, 0.0],
            ],
            dtype=np.float64,
        )
        B = np.array([[0.0], [b2]], dtype=np.float64)
        return A, B

    @staticmethod
    def _solve_care_hamiltonian(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Solve continuous-time CARE via Hamiltonian invariant subspace."""
        n = A.shape[0]
        R_inv = np.linalg.inv(R)

        H = np.block(
            [
                [A, -B @ R_inv @ B.T],
                [-Q, -A.T],
            ]
        )

        eigvals, eigvecs = np.linalg.eig(H)
        stable_idx = np.where(np.real(eigvals) < 0.0)[0]
        if stable_idx.size != n:
            raise RuntimeError(
                f"CARE solver failed: expected {n} stable eigenvalues, got {stable_idx.size}."
            )

        V_stable = eigvecs[:, stable_idx]
        X = V_stable[:n, :]
        Y = V_stable[n:, :]

        if np.linalg.matrix_rank(X) < n:
            raise RuntimeError("CARE solver failed: stable subspace basis is singular.")

        P_complex = Y @ np.linalg.inv(X)
        P = np.real_if_close(P_complex, tol=1e5).astype(np.float64)
        return 0.5 * (P + P.T)

    def V(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(x.T @ self.P @ x)

    def grad_V(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return 2.0 * (self.P @ x)

    def compute_control(self, x: np.ndarray, r: float) -> float:
        """Solve softened robust CLF-QP and return torque command."""
        x = np.asarray(x, dtype=np.float64)
        r = float(max(0.0, r))

        grad = self.grad_V(x)
        Vx = self.V(x)
        drift = self.env.f_drift(x)
        g = self.env.g_ctrl(x)

        a_drift = float(grad @ drift)
        a_ctrl = float(grad @ g)
        grad_norm = float(np.linalg.norm(grad, ord=2))

        u = cp.Variable(name="u")
        sigma = cp.Variable(nonneg=True, name="sigma")

        lhs = a_drift + a_ctrl * u + self.decay_rate * Vx
        rhs = -grad_norm * r + sigma

        constraints = [
            lhs <= rhs,
            u >= self.u_min,
            u <= self.u_max,
        ]

        objective = cp.Minimize(0.5 * cp.square(u) + self.penalty_sigma * cp.square(sigma))
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if u.value is None or problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                problem.solve(solver=cp.SCS, warm_start=True, verbose=False)
        except Exception:
            return 0.0

        if u.value is None:
            return 0.0

        return float(np.clip(u.value, self.u_min, self.u_max))
