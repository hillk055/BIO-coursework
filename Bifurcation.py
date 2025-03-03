import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Morris-Lecar Model Parameters
params = {
    "gfast": 20, "gslow": 20, "gleak": 2,
    "ENa": 50, "EK": -100, "Eleak": -70, "C": 2,
    "bm": -8, "cm": 18, "cw": 10, "ww": 0.15
}

# Function to solve for equilibrium points
def equilibrium(V, I_stim, params, bw):
    m_inf = 0.5 * (1 + np.tanh((V - params["bm"]) / params["cm"]))
    w_inf = 0.5 * (1 + np.tanh((V - bw) / params["cw"]))
    I_Na = params["gfast"] * m_inf * (V - params["ENa"])
    I_K = params["gslow"] * w_inf * (V - params["EK"])
    I_leak = params["gleak"] * (V - params["Eleak"])
    return I_stim - I_Na - I_K - I_leak

# Compute Jacobian and check stability
def is_stable(V, bw, params):
    """ Checks stability by computing the eigenvalues of the Jacobian matrix """
    m_inf = 0.5 * (1 + np.tanh((V - params["bm"]) / params["cm"]))
    w_inf = 0.5 * (1 + np.tanh((V - bw) / params["cw"]))
    tau_w = 1 / np.cosh((V - bw) / (2 * params["cw"]))

    # Compute derivatives for Jacobian
    dV_dt = -params["gfast"] * (m_inf * (V - params["ENa"])) - params["gslow"] * w_inf * (V - params["EK"]) - params["gleak"]
    dw_dt = params["ww"] * (1 - w_inf) / tau_w

    # Construct Jacobian matrix
    J = np.array([[dV_dt / params["C"], -params["gslow"] * (V - params["EK"]) / params["C"]],
                  [dw_dt, -1 / tau_w]])

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(J)
    return np.all(np.real(eigenvalues) < 0)  # True if all eigenvalues are negative (stable)

# Varying I_stim and finding equilibrium points
I_stim_values = np.linspace(0, 100, 200)  # Range of applied current
V_equilibria = []

# Initial guess for roots
V_initial_guesses = [-70, 0, 40]

for I_stim in I_stim_values:
    try:
        V_roots = fsolve(equilibrium, V_initial_guesses, args=(I_stim, params, 0))
        V_roots = np.unique(np.round(V_roots.real, decimals=3))  # Keep only unique real solutions

        if len(V_roots) > 0:
            V_equilibria.append((I_stim, V_roots))
            V_initial_guesses = V_roots  # Update guesses for next iteration

    except RuntimeError:
        pass  # Ignore failed solutions

# Plot bifurcation

plt.figure(figsize=(8, 6))

for I_stim, V_roots in V_equilibria:
    for V in V_roots:
        if is_stable(V, 0, params):
            plt.plot(I_stim, V, 'bo', markersize=2)  # Stable: solid blue
        else:
            plt.plot(I_stim, V, 'ro', markersize=2, linestyle='dotted')  # Unstable: red dotted

plt.ylim(-70, -30)
plt.xlabel("I_stim (Applied Current)")
plt.ylabel("V (Membrane Potential)")
plt.title("Bifurcation Diagram of Morris-Lecar Model (Class 1)")
plt.grid(True)
plt.show()
