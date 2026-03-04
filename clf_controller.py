"""CLF-QP controller for inverted pendulum stabilization."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from pendulum_env import InvertedPendulum


class RobustCLFController:
    """CLF-QP controller aligned with the CBF-CLF-Helper pendulum demo.

    It builds a quadratic Lyapunov function V(x)=x^T P x, then solves a QP
    each step to choose u that enforces CLF decrease (up to slack).
    """

    def __init__(
        self,
        env: InvertedPendulum,
        kp: float = 6.0,
        kd: float = 5.0,
        clf_rate: float = 3.0,
        weight_input: float = 1.0,
        weight_slack: float = 100000.0,
        u_ref: float = 0.0,
        use_robust_term: bool = False,
        u_min: float | None = None,
        u_max: float | None = None,
    ) -> None:
        """Initialize CLF and QP weights.

        Args:
            env: Pendulum model providing nominal f(x), g(x), and bounds.
            kp, kd: helper-style feedback gains used to build CLF matrix P.
            clf_rate: CLF decay coefficient in inequality LfV + LgV*u <= -cV + sigma.
            weight_input: QP penalty on control effort deviation from u_ref.
            weight_slack: QP penalty on CLF violation slack sigma.
            u_ref: nominal desired torque (0 by default, as in helper demo).
            use_robust_term: if True, include -||grad V||*r in CLF constraint.
            u_min, u_max: optional torque bounds overriding environment limits.
        """
        self.env = env
        self.kp = float(kp)
        self.kd = float(kd)
        self.clf_rate = float(clf_rate)
        self.weight_input = float(weight_input)
        self.weight_slack = float(weight_slack)
        self.u_ref = float(u_ref)
        self.use_robust_term = bool(use_robust_term)
        self.u_min = float(self.env.u_min if u_min is None else u_min)
        self.u_max = float(self.env.u_max if u_max is None else u_max)

        self.A_clf = self._clf_linearized_dynamics()
        self.Q_clf = self.clf_rate * np.eye(2, dtype=np.float64)
        self.P = self._solve_continuous_lyapunov(self.A_clf.T, self.Q_clf)

    def _clf_linearized_dynamics(self) -> np.ndarray:
        """Build helper-style closed-loop linearized A matrix.

        This is the same A used in the MATLAB demo before solving the
        Lyapunov equation for P.
        """
        I_hat = self.env.nominal_inertia()
        c_bar = (self.env.m_hat * self.env.g * self.env.l_hat) / (2.0 * I_hat)
        b_bar = self.env.b_hat / I_hat
        return np.array(
            [
                [0.0, 1.0],
                [c_bar - self.kp / I_hat, -b_bar - self.kd / I_hat],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _solve_continuous_lyapunov(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Solve the continuous Lyapunov equation A P + P A^T + Q = 0.

        We vectorize P and solve a linear system using Kronecker products:
        vec(AP + PA^T) = (I⊗A + A⊗I) vec(P).
        """
        n = A.shape[0]
        lhs = np.kron(np.eye(n), A) + np.kron(A, np.eye(n))
        rhs = -Q.reshape(-1, order="F")
        p_vec = np.linalg.solve(lhs, rhs)
        P = p_vec.reshape((n, n), order="F")
        return 0.5 * (P + P.T)

    def V(self, x: np.ndarray) -> float:
        """Evaluate quadratic CLF value V(x) = x^T P x."""
        x = np.asarray(x, dtype=np.float64)
        return float(x.T @ self.P @ x)

    def grad_V(self, x: np.ndarray) -> np.ndarray:
        """Evaluate CLF gradient ∇V(x) = 2 P x."""
        x = np.asarray(x, dtype=np.float64)
        return 2.0 * (self.P @ x)

    def compute_control(self, x: np.ndarray, r: float) -> float:
        """Solve one CLF-QP and return saturated torque.

        Constraint form:
            LfV(x) + LgV(x) u <= -clf_rate * V(x) - robust_term + sigma
        where sigma is a slack variable penalized in the objective.
        """
        x = np.asarray(x, dtype=np.float64)
        r = float(max(0.0, r))

        grad = self.grad_V(x)
        Vx = self.V(x)
        drift = self.env.f_drift(x)
        g = self.env.g_ctrl(x)

        a_drift = float(grad @ drift)
        a_ctrl = float(grad @ g)
        robust_term = float(np.linalg.norm(grad, ord=2)) * r if self.use_robust_term else 0.0

        u = cp.Variable(name="u")
        sigma = cp.Variable(name="sigma")

        lhs = a_drift + a_ctrl * u
        rhs = -self.clf_rate * Vx - robust_term + sigma

        constraints = [
            lhs <= rhs,
            u >= self.u_min,
            u <= self.u_max,
        ]

        objective = cp.Minimize(
            0.5 * self.weight_input * cp.square(u - self.u_ref) + self.weight_slack * cp.square(sigma)
        )
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
