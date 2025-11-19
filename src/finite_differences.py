"""
Finite difference utilities for simple 1D grids.

These helpers operate on 1D NumPy arrays. Boundary values are handled
conservatively: we set derivatives at the endpoints to 0.0 by default.

If you need multidimensional or different boundary treatments, extend these
functions or wrap them appropriately in your solver code.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

__all__ = [
    "forward_diff",
    "backward_diff",
    "central_diff",
    "second_central_diff",
]


def _ensure_1d(u: np.ndarray) -> np.ndarray:
    if u.ndim != 1:
        raise ValueError("finite difference helpers here expect a 1D array")
    return u


def forward_diff(u: np.ndarray, dx: float) -> np.ndarray:
    """
    First derivative using the forward difference scheme.

    D+ u_i = (u_{i+1} - u_i) / dx

    - Assumes uniform spacing dx > 0
    - Endpoints: derivative at the last point is set to 0.0
    """
    u = _ensure_1d(np.asarray(u))
    if dx <= 0:
        raise ValueError("dx must be positive")

    du = np.zeros_like(u)
    du[:-1] = (u[1:] - u[:-1]) / dx
    du[-1] = 0.0
    return du


def backward_diff(u: np.ndarray, dx: float) -> np.ndarray:
    """
    First derivative using the backward difference scheme.

    D- u_i = (u_i - u_{i-1}) / dx

    - Assumes uniform spacing dx > 0
    - Endpoints: derivative at the first point is set to 0.0
    """
    u = _ensure_1d(np.asarray(u))
    if dx <= 0:
        raise ValueError("dx must be positive")

    du = np.zeros_like(u)
    du[1:] = (u[1:] - u[:-1]) / dx
    du[0] = 0.0
    return du


def central_diff(u: np.ndarray, dx: float) -> np.ndarray:
    """
    First derivative using the central difference scheme.

    (D0 u)_i = (u_{i+1} - u_{i-1}) / (2*dx)

    - Assumes uniform spacing dx > 0
    - Endpoints: derivatives at the first and last points are set to 0.0
    """
    u = _ensure_1d(np.asarray(u))
    if dx <= 0:
        raise ValueError("dx must be positive")

    du = np.zeros_like(u)
    du[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
    du[0] = 0.0
    du[-1] = 0.0
    return du


def second_central_diff(u: np.ndarray, dx: float) -> np.ndarray:
    """
    Second derivative (Laplacian in 1D) using the central difference scheme.

    (D2 u)_i = (u_{i+1} - 2*u_i + u_{i-1}) / (dx^2)

    - Assumes uniform spacing dx > 0
    - Endpoints: second derivative at the first and last points are set to 0.0
    """
    u = _ensure_1d(np.asarray(u))
    if dx <= 0:
        raise ValueError("dx must be positive")

    d2u = np.zeros_like(u)
    d2u[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
    d2u[0] = 0.0
    d2u[-1] = 0.0
    return d2u
