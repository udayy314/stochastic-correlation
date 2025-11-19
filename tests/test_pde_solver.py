import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from model_parameters import CorrelationProcess, ConvergenceModel
from pde_solver import CrankNicolsonSolver

def test_trivial_case():
    """Test 1: Simple case where we know the solution"""
    print("=== Test 1: Trivial Case ===")
    
    # Set up parameters for a simple test
    corr_process = CorrelationProcess(mu=0.0, sigma=0.0, lambda_corr=0.0)
    
    def f1_func(tau):
        return 1.0  # Constant
    
    def f2_func(tau):
        return 0.0  # Zero
    
    # Create solver with coarse grid for quick testing
    solver = CrankNicolsonSolver(n_rho=20, n_tau=100, tau_max=5)
    solver.setup_system(corr_process, f1_func, f2_func)
    
    # Run solver
    A_solution = solver.solve()
    
    print(f"Solution shape: {A_solution.shape}")
    print(f"Final A values: {A_solution[-1, :5]}...")  # First 5 values
    
    return solver, A_solution

def test_against_known_solution():
    """Test 2: Case where PDE reduces to something with known solution"""
    print("\n=== Test 2: Known Solution Test ===")
    
    # When μ̃=0 and σ=0, PDE becomes: ∂A/∂τ = f₁(τ) + f₂(τ)ρ
    # Solution: A(ρ,τ) = ∫₀ᵗ [f₁(s) + f₂(s)ρ] ds
    
    corr_process = CorrelationProcess(mu=0.0, sigma=0.0, lambda_corr=0.0)
    
    def f1_func(tau):
        return 2.0  # Constant
    
    def f2_func(tau):
        return 1.0  # Constant
    
    solver = CrankNicolsonSolver(n_rho=20, n_tau=100, tau_max=5)
    solver.setup_system(corr_process, f1_func, f2_func)
    A_numerical = solver.solve()
    
    # Analytical solution for this case
    tau_grid = solver.tau_grid
    rho_grid = solver.rho_grid
    A_analytical = np.zeros_like(A_numerical)
    
    for i, tau in enumerate(tau_grid):
        A_analytical[i, :] = 2.0 * tau + 1.0 * tau * rho_grid
    
    error = np.max(np.abs(A_numerical - A_analytical))
    print(f"Maximum error against analytical solution: {error:.2e}")
    
    return solver, A_numerical, A_analytical

def test_convergence():
    """Test 3: Check if solution converges with grid refinement"""
    print("\n=== Test 3: Grid Convergence Test ===")
    
    errors = []
    grid_sizes = [10, 20, 40]
    
    for n_rho in grid_sizes:
        corr_process = CorrelationProcess(mu=0.1, sigma=0.1, lambda_corr=0.0)
        
        def f1_func(tau):
            return 0.1 * np.exp(-0.1 * tau)
        
        def f2_func(tau):
            return 0.05 * np.exp(-0.05 * tau)
        
        solver = CrankNicolsonSolver(n_rho=n_rho, n_tau=200, tau_max=10)
        solver.setup_system(corr_process, f1_func, f2_func)
        A_solution = solver.solve()
        
        # Store some metric of the solution
        errors.append(np.max(A_solution[-1, :]))
    
    print(f"Solution behavior with grid refinement: {errors}")
    return errors

if __name__ == "__main__":
    # Run all tests
    #test_trivial_case()
    test_against_known_solution() 
    #test_convergence()