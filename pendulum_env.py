"""Inverted pendulum environment with true and learned nominal dynamics."""

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

    Nominal model parameterization:
      phi(x) = feature vector returned by features(x)
      f_drift(x) = Mf @ phi(x),   Mf shape (2, m)
      g_ctrl(x)  = Mg @ phi(x),   Mg shape (2, m)
      where m = len(phi(x)).
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
        Mf: np.ndarray | None = None,
        Mg: np.ndarray | None = None,
        nominal_weights_path: str | None = None,
    ) -> None:
        """Store physical and nominal parameters.

        Args:
            g, m, l, b: true gravity/mass/length/friction parameters.
            m_hat, l_hat, b_hat: nominal parameters seen by controller. If None,
                defaults to the corresponding true values.
            u_min, u_max: actuator limits applied during simulation.
            disturbance_amp, disturbance_freq: sinusoidal disturbance injected
                only in the true acceleration equation.
            Mf, Mg: optional learned nominal model weights.
            nominal_weights_path: optional .npz file containing Mf and Mg.
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
        self.Mf = np.zeros((2, self.feature_dim()), dtype=np.float64)
        self.Mg = np.zeros((2, self.feature_dim()), dtype=np.float64)

        if nominal_weights_path is not None:
            self.load_nominal_weights(nominal_weights_path)
        elif Mf is not None and Mg is not None:
            self.set_nominal_weights(Mf, Mg)
        else:
            self._set_bootstrap_nominal_weights()

    def true_inertia(self) -> float:
        """Pendulum inertia used in true dynamics: I = m*l^2/3."""
        return self.m * self.l**2 / 3.0

    def nominal_inertia(self) -> float:
        """Nominal inertia used by model-based controller terms."""
        return self.m_hat * self.l_hat**2 / 3.0

    def disturbance(self, t: float) -> float:
        """Time-varying unknown disturbance added to true acceleration."""
        return self.disturbance_amp * np.sin(self.disturbance_freq * float(t))

    @staticmethod
    def features(x: np.ndarray) -> np.ndarray:
        """Polynomial state features phi(x) used by learned nominal model."""
        x = np.asarray(x, dtype=np.float64)
        theta = x[0]
        theta_dot = x[1]
        return np.array(
            [
                1.0,
                theta,
                theta_dot,
                theta**2,
                theta * theta_dot,
                theta_dot**2,
                theta**3,
                theta**2 * theta_dot,
                theta * theta_dot**2,
                theta_dot**3,
            ],
            dtype=np.float64,
        )

    def feature_dim(self) -> int:
        """Current feature vector length m used by Mf/Mg."""
        return int(self.features(np.zeros(2, dtype=np.float64)).size)

    def set_nominal_weights(self, Mf: np.ndarray, Mg: np.ndarray) -> None:
        """Set learned nominal model weights after shape/type validation."""
        Mf = np.asarray(Mf, dtype=np.float64)
        Mg = np.asarray(Mg, dtype=np.float64)
        expected = (2, self.feature_dim())
        if Mf.shape != expected or Mg.shape != expected:
            raise ValueError(f"Mf and Mg must both have shape {expected}, got {Mf.shape} and {Mg.shape}.")
        self.Mf = Mf
        self.Mg = Mg

    def save_nominal_weights(self, path: str) -> None:
        """Save learned nominal weights to a .npz file."""
        np.savez(path, Mf=self.Mf, Mg=self.Mg)

    def load_nominal_weights(self, path: str) -> None:
        """Load learned nominal weights from a .npz file with keys Mf and Mg."""
        data = np.load(path)
        self.set_nominal_weights(data["Mf"], data["Mg"])

    def _set_bootstrap_nominal_weights(self) -> None:
        """Initialize nominal model with a small-angle physics-based bootstrap.

        This fallback keeps the pipeline runnable before data fitting.
        """
        I_hat = self.nominal_inertia()
        c_theta = (self.m_hat * self.g * self.l_hat) / (2.0 * I_hat)
        c_dtheta = -self.b_hat / I_hat
        c_u = -1.0 / I_hat

        m = self.feature_dim()
        if m < 3:
            raise RuntimeError("Feature vector must have at least 3 entries for bootstrap nominal model.")
        Mf = np.zeros((2, m), dtype=np.float64)
        Mg = np.zeros((2, m), dtype=np.float64)
        Mf[0, 2] = 1.0
        Mf[1, 1] = c_theta
        Mf[1, 2] = c_dtheta
        Mg[1, 0] = c_u
        self.set_nominal_weights(Mf, Mg)

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
        phi = self.features(x)
        return self.Mf @ phi

    def g_ctrl(self, _x: np.ndarray) -> np.ndarray:
        """Nominal control direction g(x) in x_dot = f(x) + g(x)u."""
        phi = self.features(_x)
        return self.Mg @ phi

    def nominal_dynamics(self, x: np.ndarray, u: float) -> np.ndarray:
        """Nominal model state derivative x_dot = f_drift(x) + g_ctrl(x)*u."""
        return self.f_drift(x) + self.g_ctrl(x) * float(u)

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
