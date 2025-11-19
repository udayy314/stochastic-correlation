import numpy as np

def central_difference_first(f, dx):
    """Second-order first derivative"""
    df = np.zeros_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
    # Forward/backward differences at boundaries for Neumann
    df[0] = (f[1] - f[0]) / dx
    df[-1] = (f[-1] - f[-2]) / dx
    return df

def central_difference_second(f, dx):
    """Second-order second derivative"""
    d2f = np.zeros_like(f)
    d2f[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / (dx**2)
    # Neumann boundary treatment
    d2f[0] = 2 * (f[1] - f[0]) / (dx**2)  # Using ∂A/∂ρ=0 at boundary
    d2f[-1] = 2 * (f[-2] - f[-1]) / (dx**2)
    return d2f