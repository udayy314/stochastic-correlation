import numpy as np
from scipy.linalg import solve_banded

def newton_raphson_pde(residual_function, jacobian_function, A_guess, max_iter=10, tol=1e-8):
    """
    Newton-Raphson solver for nonlinear PDE system
    
    Parameters:
    - residual_function: function that returns F(A) 
    - jacobian_function: function that returns Jacobian matrix J(A) in banded format
    - A_guess: initial guess for solution vector
    - max_iter: maximum iterations
    - tol: convergence tolerance
    """
    A_current = A_guess.copy()
    
    for iteration in range(max_iter):
        F = residual_function(A_current)
        J = jacobian_function(A_current)
        
        # Solve J·ΔA = -F using banded solver
        delta_A = solve_banded((1, 1), J, -F)
        
        A_new = A_current + delta_A
        
        # Check convergence
        residual_norm = np.linalg.norm(F)
        update_norm = np.linalg.norm(delta_A)
        
        print(f"Iteration {iteration+1}: Residual norm = {residual_norm:.2e}, Update norm = {update_norm:.2e}")
        
        if update_norm < tol:
            print(f"Newton-Raphson converged in {iteration+1} iterations")
            return A_new
            
        A_current = A_new
    
    print("Warning: Newton-Raphson did not converge within maximum iterations")
    return A_current