import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class LAGSModel:
    def __init__(self, beta=40, gamma=0.4, mu=2000, delta=1, nx=100):
        """
        Initialize LAGS model with parameters from the paper
        beta: preferential degradation rate of active receptors
        gamma: dimensionless diffusion constant
        mu: dimensionless ligand unbinding rate
        delta: ratio of diffusion constants (active/inactive)
        nx: number of spatial points
        """
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.delta = delta
        self.nx = nx
        self.dx = 1.0 / (nx - 1)

    def ligand_profile(self, x, u0=1.0, g=0.1):
        """Define ligand concentration profile with gradient g"""
        return u0 * (1 + g * x)

    def spatial_derivative(self, c):
        """Compute second spatial derivative using central differences with periodic boundary conditions"""
        d2c = np.zeros_like(c)
        # Interior points
        d2c[1:-1] = (c[2:] - 2 * c[1:-1] + c[:-2]) / (self.dx**2)
        # Periodic boundary conditions
        d2c[0] = (c[1] - 2 * c[0] + c[-1]) / (self.dx**2)
        d2c[-1] = (c[0] - 2 * c[-1] + c[-2]) / (self.dx**2)
        return d2c

    def rhs(self, t, y):
        """Right hand side of the reaction-diffusion equations"""
        n = self.nx
        r, b = y[:n], y[n:]
        x = np.linspace(0, 1, n)
        u = self.ligand_profile(x)

        # Compute diffusion terms
        d2r = self.spatial_derivative(r)
        d2b = self.spatial_derivative(b)

        # Reaction terms for inactive receptors (r)
        dr_dt = self.gamma * d2r + 1 - r - self.mu * u * r + self.mu * b

        # Reaction terms for active receptors (b)
        db_dt = (
            self.gamma / self.delta * d2b
            - self.beta * b
            + self.mu * u * r
            - self.mu * b
        )

        return np.concatenate([dr_dt, db_dt])

    def simulate(self, t_span=(0, 100), t_eval=None):
        """Simulate the system until steady state"""
        x = np.linspace(0, 1, self.nx)

        # Initial conditions: uniform distribution
        r0 = np.ones(self.nx)
        b0 = np.zeros(self.nx)
        y0 = np.concatenate([r0, b0])

        # Solve the system
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)

        sol = solve_ivp(self.rhs, t_span, y0, t_eval=t_eval, method="BDF")

        return sol, x

    def plot_results(self, with_diffusion=True):
        """Compare results with and without diffusion"""
        # Simulate with current gamma
        gamma_orig = self.gamma
        if not with_diffusion:
            self.gamma = 1e-4  # Very small diffusion

        sol, x = self.simulate()

        # Extract final state
        r = sol.y[: self.nx, -1]
        b = sol.y[self.nx :, -1]
        rt = r + b  # Total receptors

        # Reset gamma
        self.gamma = gamma_orig

        return x, rt, b


def compare_diffusion_effects():
    """Compare the effects of diffusion on receptor polarization"""
    model = LAGSModel()

    # Get results with and without diffusion
    x, rt_with, b_with = model.plot_results(with_diffusion=True)
    x, rt_without, b_without = model.plot_results(with_diffusion=False)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot total receptors
    ax1.plot(x, rt_with, "b-", label="With diffusion")
    ax1.plot(x, rt_without, "b--", label="Without diffusion")
    ax1.set_title("Total Receptors")
    ax1.set_xlabel("Position along cell")
    ax1.set_ylabel("% change in rt")
    ax1.legend()

    # Plot active receptors
    ax2.plot(x, b_with, "r-", label="With diffusion")
    ax2.plot(x, b_without, "r--", label="Without diffusion")
    ax2.set_title("Active Receptors")
    ax2.set_xlabel("Position along cell")
    ax2.set_ylabel("% change in b")
    ax2.legend()

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = compare_diffusion_effects()
    plt.show()
