"""Inverted pendulum environment with true and nominal dynamics."""

from __future__ import annotations

import numpy as np


class InvertedPendulum:
    """Inverted pendulum model with separate true and nominal dynamics.

    State convention:
    - x[0] = theta (angle from upright, radians)
    - x[1] = theta_dot (angular velocity, rad/s)

    This class exposes:
    - true dynamics used for simulation
    - nominal dynamics used by the controller/QP
    - residual = (true - nominal), used by conformal updates
    """

    def __init__(
        self,
        g: float = 9.81,
        m: float = 1.0,
        l: float = 1.0,
        b: float = 0.01,
        m_hat: float | None = None,
        l_hat: float | None = None,
        b_hat: float | None = None,
        u_min: float = -7.0,
        u_max: float = 7.0,
        disturbance_amp: float = 0.0,
        disturbance_freq: float = 2.0,
    ) -> None:
        """Store physical and nominal parameters.

        Args:
            g, m, l, b: true gravity/mass/length/friction parameters.
            m_hat, l_hat, b_hat: nominal parameters seen by controller. If None,
                defaults to the corresponding true values.
            u_min, u_max: actuator limits applied during simulation.
            disturbance_amp, disturbance_freq: sinusoidal disturbance injected
                only in the true acceleration equation.
        """
        self.g = float(g)
        self.m = float(m)
        self.l = float(l)
        self.b = float(b)
        self.m_hat = float(self.m if m_hat is None else m_hat)
        self.l_hat = float(self.l if l_hat is None else l_hat)
        self.b_hat = float(self.b if b_hat is None else b_hat)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.disturbance_amp = float(disturbance_amp)
        self.disturbance_freq = float(disturbance_freq)

    def true_inertia(self) -> float:
        """Pendulum inertia used in true dynamics: I = m*l^2/3."""
        return self.m * self.l**2 / 3.0

    def nominal_inertia(self) -> float:
        """Nominal inertia used by model-based controller terms."""
        return self.m_hat * self.l_hat**2 / 3.0

    def disturbance(self, t: float) -> float:
        """Time-varying unknown disturbance added to true acceleration."""
        return self.disturbance_amp * np.sin(self.disturbance_freq * float(t))

    def true_dynamics(self, x: np.ndarray, u: float, t: float) -> np.ndarray:
        """Compute physical state derivative x_dot under true dynamics.

        Returns:
            np.ndarray shape (2,), [theta_dot, theta_ddot_true].
        """
        x = np.asarray(x, dtype=np.float64)
        u = float(u)
        I = self.true_inertia()

        x1_dot = x[1]
        x2_dot = ((-self.b * x[1]) + (self.m * self.g * self.l * np.sin(x[0]) / 2.0)) / I
        x2_dot += -(1.0 / I) * u + self.disturbance(t)
        return np.array([x1_dot, x2_dot], dtype=np.float64)

    def f_drift(self, x: np.ndarray) -> np.ndarray:
        """Nominal *drift* term f(x), i.e., dynamics without control input.

        In control-affine form x_dot = f(x) + g(x)u:
        - f(x) is the part that evolves naturally from physics/state
        - g(x)u is the part directly caused by control u
        """
        x = np.asarray(x, dtype=np.float64)
        I_hat = self.nominal_inertia()
        x2_dot = ((-self.b_hat * x[1]) + (self.m_hat * self.g * self.l_hat * np.sin(x[0]) / 2.0)) / I_hat
        return np.array([x[1], x2_dot], dtype=np.float64)

    def g_ctrl(self, _x: np.ndarray) -> np.ndarray:
        """Nominal control direction g(x) in x_dot = f(x) + g(x)u."""
        return np.array([0.0, -1.0 / self.nominal_inertia()], dtype=np.float64)

    def compute_residual(self, x: np.ndarray, u: float, t: float) -> np.ndarray:
        """Model mismatch residual epsilon(x,u,t) = true_dynamics - nominal_dynamics.

        This is the quantity scored by conformal/SR-CR updates.
        """
        x = np.asarray(x, dtype=np.float64)
        u = float(u)
        nominal = self.f_drift(x) + self.g_ctrl(x) * u
        return self.true_dynamics(x, u, t) - nominal

    def step(self, x: np.ndarray, u: float, t: float, dt: float) -> np.ndarray:
        """Advance state by one time step using RK4 integration.

        RK4 (4th-order Runge-Kutta) evaluates derivatives at 4 points (k1..k4)
        within the interval and combines them. It is more accurate/stable than
        single-step Euler at the same dt.
        """
        x = np.asarray(x, dtype=np.float64)
        u = float(np.clip(u, self.u_min, self.u_max))
        dt = float(dt)

        k1 = self.true_dynamics(x, u, t)
        k2 = self.true_dynamics(x + 0.5 * dt * k1, u, t + 0.5 * dt)
        k3 = self.true_dynamics(x + 0.5 * dt * k2, u, t + 0.5 * dt)
        k4 = self.true_dynamics(x + dt * k3, u, t + dt)

        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
