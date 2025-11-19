import numpy as np
from scipy.linalg import solve_banded

class JacobianCalculator:
    def __init__(self, rho_grid, d_tau, corr_process):
        self.rho_grid = rho_grid
        self.d_rho = rho_grid[1] - rho_grid[0]
        self.d_tau = d_tau
        self.corr_process = corr_process
        self.n_points = len(rho_grid)
        
    def compute_jacobian(self, A, tau):
        """
        Compute the Jacobian matrix J_ij = ∂R_i/∂A_j
        
        Parameters:
        - A: current solution vector at time m+1
        - tau: current time
        
        Returns:
        - J: tridiagonal Jacobian matrix in banded format (3 x n_points)
        """
        J = np.zeros((3, self.n_points))
        
        # Precompute mu_tilde and sigma_sq for all grid points
        mu_tilde = np.array([self.corr_process.risk_neutral_drift(rho) for rho in self.rho_grid])
        sigma_sq = np.array([self.corr_process.diffusion(rho)**2 for rho in self.rho_grid])
        
        # Interior points (i = 1 to n_points-2)
        for i in range(1, self.n_points - 1):
            # Finite differences at point i
            A_rho = (A[i+1] - A[i-1]) / (2 * self.d_rho)
            A_rho_rho = (A[i+1] - 2*A[i] + A[i-1]) / (self.d_rho**2)
            
            # Jacobian components
            # Main diagonal (j = i)
            J[1, i] = 1.0  # From A_i term
            
            # Contribution from ∂²A/∂ρ² term
            J[1, i] -= (self.d_tau/2) * 0.5 * sigma_sq[i] * (-2/self.d_rho**2)
            
            # Upper diagonal (j = i+1)
            J[2, i] = 0.0  # Will be set when i+1 is the main point
            J[2, i-1] = - (self.d_tau/2) * (
                mu_tilde[i]/(2 * self.d_rho) +                    # From ∂A/∂ρ term
                0.5 * sigma_sq[i]/(self.d_rho**2) +              # From ∂²A/∂ρ² term  
                sigma_sq[i] * A_rho/(4 * self.d_rho**2)          # From (∂A/∂ρ)² term
            )
            
            # Lower diagonal (j = i-1)  
            J[0, i] = - (self.d_tau/2) * (
                -mu_tilde[i]/(2 * self.d_rho) +                  # From ∂A/∂ρ term
                0.5 * sigma_sq[i]/(self.d_rho**2) -              # From ∂²A/∂ρ² term
                sigma_sq[i] * A_rho/(4 * self.d_rho**2)          # From (∂A/∂ρ)² term
            )
        
        # Boundary conditions (Neumann)
        self._apply_boundary_conditions(J)
        
        return J
    
    def _apply_boundary_conditions(self, J):
        """Apply Neumann boundary conditions to Jacobian"""
        # Left boundary (i=0): A_1 = A_0
        J[1, 0] = 1.0
        J[2, 0] = -1.0  # A_0 - A_1 = 0
        
        # Right boundary (i=n_points-1): A_{N-1} = A_{N-2}  
        J[1, -1] = 1.0
        J[0, -1] = -1.0  # A_{N-1} - A_{N-2} = 0