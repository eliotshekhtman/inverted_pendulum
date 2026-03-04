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
        k_fb: tuple[float, float] | None = (8.0, 5.0),
        c3: float = 0.5,
        weight_input: float = 1.0,
        weight_slack: float = 100000.0,
        u_ref: float = 0.0,
        use_robust_term: bool = False,
        u_min: float | None = None,
        u_max: float | None = None,
        auto_select_k: bool = False, # Overrides k_fb
        stability_margin: float = 1e-3,
        max_p_condition: float = 1e8,
    ) -> None:
        """Initialize CLF and QP weights.

        Args:
            env: Pendulum model providing nominal f(x), g(x), and bounds.
            k_fb: fixed feedback gains [k_theta, k_theta_dot] used only to build
                the CLF matrix through closed-loop Jacobian linearization.
                Ignored if auto_select_k=True.
            c3: exponential decay rate used for Q = c3 * I and in CLF-QP decay.
            weight_input: QP penalty on control effort deviation from u_ref.
            weight_slack: QP penalty on CLF violation slack sigma.
            u_ref: nominal desired torque (0 by default, as in helper demo).
            use_robust_term: if True, include -||grad V||*r in CLF constraint.
            u_min, u_max: optional torque bounds overriding environment limits.
            auto_select_k: if True, search for a stabilizing K with better P
                conditioning on the learned nominal model.
            stability_margin: require max real eigenvalue(A_cl) <= -margin.
            max_p_condition: reject candidates with cond(P) above this threshold.
        """
        self.env = env
        if k_fb is None:
            self.k_fb = np.array([8.0, 5.0], dtype=np.float64)
        else:
            self.k_fb = np.asarray(k_fb, dtype=np.float64).reshape(2)
        self.c3 = float(c3)
        self.weight_input = float(weight_input)
        self.weight_slack = float(weight_slack)
        self.u_ref = float(u_ref)
        self.use_robust_term = bool(use_robust_term)
        self.u_min = float(self.env.u_min if u_min is None else u_min)
        self.u_max = float(self.env.u_max if u_max is None else u_max)
        self.auto_select_k = bool(auto_select_k)
        self.stability_margin = float(stability_margin)
        self.max_p_condition = float(max_p_condition)

        if self.auto_select_k:
            self.k_fb = self._select_feedback_gain()

        self.A_clf = self._closed_loop_jacobian_at_origin(self.k_fb)
        self.Q_clf = self.c3 * np.eye(2, dtype=np.float64)
        self.P = self._solve_continuous_lyapunov(self.A_clf.T, self.Q_clf)

    def _feedback_u(self, x: np.ndarray, k_fb: np.ndarray | None = None) -> float:
        """Fixed feedback law u = Kx used only for CLF linearization."""
        x = np.asarray(x, dtype=np.float64).reshape(2)
        k = self.k_fb if k_fb is None else np.asarray(k_fb, dtype=np.float64).reshape(2)
        return float(k @ x)

    def _closed_loop_nominal_dynamics(self, x: np.ndarray, k_fb: np.ndarray | None = None) -> np.ndarray:
        """Nominal closed-loop vector field f_cl(x) = f(x) + g(x) * (Kx)."""
        return self.env.nominal_dynamics(x, self._feedback_u(x, k_fb))

    def _closed_loop_jacobian_at_origin(self, k_fb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Numerically approximate Jacobian A = df_cl/dx at x = [0, 0].

        Central finite differences are used to support arbitrary learned
        nominal models without symbolic differentiation.
        """
        x0 = np.zeros(2, dtype=np.float64)
        A = np.zeros((2, 2), dtype=np.float64)
        for i in range(2):
            e = np.zeros(2, dtype=np.float64)
            e[i] = eps
            f_plus = self._closed_loop_nominal_dynamics(x0 + e, k_fb)
            f_minus = self._closed_loop_nominal_dynamics(x0 - e, k_fb)
            A[:, i] = (f_plus - f_minus) / (2.0 * eps)
        return A

    def _select_feedback_gain(self) -> np.ndarray:
        """Search over candidate gains and pick one with stable A and well-conditioned P."""
        # Practical low-dimensional grid; adjust as needed for tighter tuning.
        k_theta_grid = np.linspace(2.0, 20.0, 19)
        k_dtheta_grid = np.linspace(1.0, 15.0, 15)

        best_k = None
        best_score = np.inf

        for k_theta in k_theta_grid:
            for k_dtheta in k_dtheta_grid:
                k = np.array([k_theta, k_dtheta], dtype=np.float64)
                try:
                    A = self._closed_loop_jacobian_at_origin(k)
                    eigvals = np.linalg.eigvals(A)
                    max_real = float(np.max(np.real(eigvals)))
                    if max_real > -self.stability_margin:
                        continue

                    Q = self.c3 * np.eye(2, dtype=np.float64)
                    P = self._solve_continuous_lyapunov(A.T, Q)
                    p_eigs = np.linalg.eigvalsh(P)
                    min_eig = float(np.min(p_eigs))
                    if min_eig <= 1e-10:
                        continue

                    cond_p = float(np.linalg.cond(P))
                    if not np.isfinite(cond_p) or cond_p > self.max_p_condition:
                        continue

                    # Prefer better-conditioned P and reasonable gain magnitude.
                    score = cond_p + 1e-3 * float(np.dot(k, k))
                    if score < best_score:
                        best_score = score
                        best_k = k
                except np.linalg.LinAlgError:
                    continue

        if best_k is None:
            raise RuntimeError(
                "Auto-select K failed: no candidate produced stable A and well-conditioned P. "
                "Try relaxing max_p_condition or expanding gain search ranges."
            )
        return best_k

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
            LfV(x) + LgV(x) u <= -c3 * V(x) - robust_term + sigma
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
        rhs = -self.c3 * Vx - robust_term + sigma

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
