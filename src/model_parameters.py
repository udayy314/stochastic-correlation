import numpy as np
from dataclasses import dataclass

@dataclass
class CorrelationProcess:
    """Parameters for the stochastic correlation process (Example 1 from paper)"""
    mu: float = 0.1      # Drift parameter
    sigma: float = 0.25  # Volatility parameter  
    lambda_corr: float = -5.0  # Market price of correlation risk
    
    def risk_neutral_drift(self, rho):
        """Compute μ̃(ρ) = μ(ρ) - λσ(ρ)"""
        # For Example 1 process: μ̃(ρ) = ρ(1-ρ)(μ - λσ - ρσ²)
        return rho * (1 - rho) * (self.mu - self.lambda_corr * self.sigma - rho * self.sigma**2)
    
    def diffusion(self, rho):
        """Compute σ(ρ)"""
        return self.sigma * rho * (1 - rho)

@dataclass 
class ConvergenceModel:
    """Parameters for the convergence model from paper Section 4"""
    a: float = 0.0938
    b: float = 3.67
    sigma_d: float = 0.032
    c: float = 0.2087  
    d: float = 0.035
    sigma_u: float = 0.016
    lambda_d: float = 3.315
    lambda_u: float = -0.655

print("hellowrld")