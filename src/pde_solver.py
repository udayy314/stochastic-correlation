"""
Minimal PDE solver utilities for 1D problems.

Currently includes a simple explicit solver for the 1D heat equation:

    u_t = alpha * u_xx

using a second-order central difference in space and forward Euler in time.
This is primarily a lightweight scaffold you can extend for your project.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .finite_differences import central_difference_second

__all__ = ["heat_equation"]


@dataclass
class HeatEqConfig:
    """Configuration for the explicit 1D heat equation solver.

    Attributes:
        alpha: Diffusivity coefficient (must be > 0)
        dx: Spatial grid spacing (uniform, must be > 0)
        dt: Time step (must be > 0)
        steps: Number of explicit time steps to take (>= 1)
        bc: Boundary condition type: "dirichlet" (u=0 at boundaries) or
            "neumann" (zero-gradient at boundaries)
    """

    alpha: float
    dx: float
    dt: float
    steps: int
    bc: str = "dirichlet"


def _apply_bc(u: np.ndarray, bc: str) -> None:
    if bc == "dirichlet":
        u[0] = 0.0
        u[-1] = 0.0
    elif bc == "neumann":
        # zero gradient: du/dx = 0 -> copy interior neighbors
        u[0] = u[1]
        u[-1] = u[-2]
    else:
        raise ValueError("Unsupported bc. Use 'dirichlet' or 'neumann'.")


def heat_equation(u0: np.ndarray, config: HeatEqConfig) -> np.ndarray:
    """
    Solve u_t = alpha * u_xx on a uniform 1D grid with an explicit scheme.

    Args:
        u0: Initial condition as a 1D array (will not be modified).
        config: HeatEqConfig with alpha, dx, dt, steps, and bc.

    Returns:
        Final solution array after `steps` explicit updates.

    Notes:
        Stability (CFL) for explicit method requires alpha * dt / dx^2 <= 1/2.
        This function does not enforce it, but will warn if exceeded.
    """
    u = np.asarray(u0, dtype=float).copy()
    if u.ndim != 1:
        raise ValueError("heat_equation currently expects a 1D array")

    alpha, dx, dt, steps = config.alpha, config.dx, config.dt, config.steps
    if not (alpha > 0 and dx > 0 and dt > 0 and steps >= 1):
        raise ValueError("alpha, dx, dt must be > 0 and steps >= 1")

    cfl = alpha * dt / (dx * dx)
    if cfl > 0.5:
        # Lightweight runtime warning without importing warnings
        print(f"[heat_equation] Warning: explicit scheme likely unstable (CFL={cfl:.3f} > 0.5)")

    for _ in range(steps):
        lap = central_difference_second(u, dx)
        u = u + alpha * dt * lap
        _apply_bc(u, config.bc)

    return u


if __name__ == "__main__":
    # Tiny demo: Gaussian blob diffuses with Dirichlet BCs
    nx = 101
    L = 1.0
    x = np.linspace(0.0, L, nx)
    dx = x[1] - x[0]
    u0 = np.exp(-((x - 0.5) ** 2) / (2 * 0.05**2))

    cfg = HeatEqConfig(alpha=0.01, dx=dx, dt=1e-4, steps=1000, bc="dirichlet")
    uT = heat_equation(u0, cfg)
    print(f"Solution stats: min={uT.min():.4f}, max={uT.max():.4f}")
