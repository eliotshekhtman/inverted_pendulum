"""Inverted pendulum environment with true and nominal dynamics."""

from __future__ import annotations

import numpy as np


class InvertedPendulum:
    """Inverted pendulum benchmark with model mismatch and disturbance."""

    def __init__(
        self,
        g: float = 9.81,
        m: float = 1.0,
        l: float = 1.0,
        k: float = 0.1,
        m_hat: float = 0.75,
        l_hat: float = 0.9,
    ) -> None:
        self.g = float(g)
        self.m = float(m)
        self.l = float(l)
        self.k = float(k)
        self.m_hat = float(m_hat)
        self.l_hat = float(l_hat)

    def disturbance(self, t: float) -> float:
        """External disturbance in true acceleration dynamics."""
        return 0.2 * np.sin(2.0 * float(t))

    def true_dynamics(self, x: np.ndarray, u: float, t: float) -> np.ndarray:
        """True dynamics f_true(x, u, t)."""
        x = np.asarray(x, dtype=np.float64)
        u = float(u)

        x1_dot = x[1]
        x2_dot = (
            (self.g / self.l) * np.sin(x[0])
            - (self.k / (self.m * self.l**2)) * x[1]
            + (1.0 / (self.m * self.l**2)) * u
            + self.disturbance(t)
        )
        return np.array([x1_dot, x2_dot], dtype=np.float64)

    def f_drift(self, x: np.ndarray) -> np.ndarray:
        """Nominal drift dynamics f_drift(x)."""
        x = np.asarray(x, dtype=np.float64)
        return np.array([x[1], (self.g / self.l_hat) * np.sin(x[0])], dtype=np.float64)

    def g_ctrl(self, _x: np.ndarray) -> np.ndarray:
        """Nominal control vector field g_ctrl(x)."""
        return np.array([0.0, 1.0 / (self.m_hat * self.l_hat**2)], dtype=np.float64)

    def compute_residual(self, x: np.ndarray, u: float, t: float) -> np.ndarray:
        """Residual epsilon = f_true - (f_drift + g_ctrl * u)."""
        x = np.asarray(x, dtype=np.float64)
        u = float(u)
        nominal = self.f_drift(x) + self.g_ctrl(x) * u
        return self.true_dynamics(x, u, t) - nominal

    def step(self, x: np.ndarray, u: float, t: float, dt: float) -> np.ndarray:
        """Advance true dynamics by one RK4 step."""
        x = np.asarray(x, dtype=np.float64)
        u = float(np.clip(u, -15.0, 15.0))
        dt = float(dt)

        k1 = self.true_dynamics(x, u, t)
        k2 = self.true_dynamics(x + 0.5 * dt * k1, u, t + 0.5 * dt)
        k3 = self.true_dynamics(x + 0.5 * dt * k2, u, t + 0.5 * dt)
        k4 = self.true_dynamics(x + dt * k3, u, t + dt)

        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
