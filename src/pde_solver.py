import numpy as np
from finite_differences import central_difference_first, central_difference_second
from nonlinear_solver import newton_raphson_pde
from jacobian import JacobianCalculator

class CrankNicolsonSolver:
    def __init__(self, rho_min=0.01, rho_max=0.99, n_rho=100, tau_max=50, n_tau=1000):
        self.rho_min, self.rho_max = rho_min, rho_max
        self.n_rho, self.n_tau = n_rho, n_tau
        self.tau_max = tau_max
        
        # Spatial grid
        self.rho_grid = np.linspace(rho_min, rho_max, n_rho)
        self.d_rho = self.rho_grid[1] - self.rho_grid[0]
        
        # Time grid  
        self.tau_grid = np.linspace(0, tau_max, n_tau)
        self.d_tau = self.tau_grid[1] - self.tau_grid[0]
        
        # Solution matrix: A[tau_index, rho_index]
        self.A = np.zeros((n_tau, n_rho))
        
    def setup_system(self, corr_process, f1_func, f2_func):
        """Set up the PDE system with given parameters"""
        self.corr_process = corr_process
        self.f1_func = f1_func
        self.f2_func = f2_func
        self.jacobian_calc = JacobianCalculator(self.rho_grid, self.d_tau, corr_process)
        
    def solve(self):
        """Main solver using Crank-Nicolson with Newton-Raphson"""
        print("Starting Crank-Nicolson solver...")
        
        for n in range(self.n_tau - 1):
            tau_current = self.tau_grid[n]
            tau_next = self.tau_grid[n + 1]
            A_current = self.A[n, :].copy()
            
            # Newton-Raphson for implicit step
            def residual(A_next_guess):
                return self._compute_residual(A_current, A_next_guess, tau_current, tau_next)
                
            def jacobian(A_next_guess):
                return self.jacobian_calc.compute_jacobian(A_next_guess, tau_next)
            
            # Initial guess (explicit Euler step for better convergence)
            A_guess = A_current + self.d_tau * self._rhs(A_current, tau_current)
            self.A[n+1, :] = newton_raphson_pde(residual, jacobian, A_guess)
            
            if n % 100 == 0:
                print(f"Progress: {n/self.n_tau*100:.1f}%")
                
        return self.A
    
    def _rhs(self, A, tau):
        """Right-hand side of the PDE at fixed tau"""
        A_rho = central_difference_first(A, self.d_rho)
        A_rho_rho = central_difference_second(A, self.d_rho)
        
        mu_tilde = np.array([self.corr_process.risk_neutral_drift(rho) for rho in self.rho_grid])
        sigma_sq = np.array([self.corr_process.diffusion(rho)**2 for rho in self.rho_grid])
        
        f1 = self.f1_func(tau)
        f2 = self.f2_func(tau)
        
        rhs = (mu_tilde * A_rho + 
               0.5 * sigma_sq * (A_rho_rho + A_rho**2) + 
               f1 + f2 * self.rho_grid)
        return rhs
    
    def _compute_residual(self, A_curr, A_next, tau_curr, tau_next):
        """Residual for Crank-Nicolson scheme"""
        rhs_curr = self._rhs(A_curr, tau_curr)
        rhs_next = self._rhs(A_next, tau_next)
        
        return A_next - A_curr - 0.5 * self.d_tau * (rhs_curr + rhs_next)