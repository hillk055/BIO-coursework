import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Modified Morris–Lecar parameters
C = 20.0           # Membrane capacitance (µF/cm²)
g_Ca = 4.0         # Calcium conductance (mS/cm²)
g_K = 8.0          # Potassium conductance (mS/cm²)
g_L = 2.0          # Leak conductance (mS/cm²)
E_Ca = 120.0       # Calcium reversal potential (mV)
E_K = -84.0        # Potassium reversal potential (mV)
E_L = -60.0        # Leak reversal potential (mV)

# Fast current activation parameters
bm = -1.2          # Half-activation voltage for m_inf (adjustable parameter)
V2 = 18.0          # Slope factor for m_inf

# Slow variable (recovery) parameters
bw = 0           # Half-activation voltage for w_inf (this parameter shifts the w-nullcline)
V4 = 30.0          # Slope factor for w_inf

phi = 0.04         # Rate constant for w (1/ms)
I = 90.0           # External stimulus current (µA/cm²)

def m_inf(V):
    """Fast activation variable (e.g., representing Ca²⁺ or Na⁺ channels)."""
    return 0.5 * (1 + np.tanh((V - bm) / V2))

def w_inf(V):
    """Steady-state value for the recovery variable (e.g., slow K⁺ current),
       shifted by parameter bw to mimic the effects described in the paper."""
    return 0.5 * (1 + np.tanh((V - bw) / V4))

def lambda_w(V):
    """Voltage-dependent rate for the recovery variable."""
    return phi * np.cosh((V - bw) / (2 * V4))

def dX_dt(X, t):
    V, w = X
    dVdt = (I - g_Ca * m_inf(V) * (V - E_Ca) - g_K * w * (V - E_K) - g_L * (V - E_L)) / C
    dwdt = lambda_w(V) * (w_inf(V) - w)
    return [dVdt, dwdt]

# Simulate the system trajectory over time
t = np.linspace(0, 500, 5000)
X0 = [-20, 0.3]  # Initial conditions for V and w
sol = odeint(dX_dt, X0, t)

# Compute the nullclines over a range of membrane potentials
V_range = np.linspace(-80, 80, 400)
# V-nullcline: set dV/dt = 0 and solve for w
w_V_nullcline = (I - g_Ca * m_inf(V_range) * (V_range - E_Ca) - g_L * (V_range - E_L)) / (g_K * (V_range - E_K))
# w-nullcline: given directly by w_inf(V)
w_nullcline_V = w_inf(V_range)

# Plot the phase plane with nullclines and trajectory
plt.figure(figsize=(8, 6))
plt.plot(V_range, w_V_nullcline, 'b-', label='V-nullcline')
plt.plot(V_range, w_nullcline_V, 'r-', label=f'w-nullcline (bw = {bw:.1f})')
plt.plot(sol[:, 0], sol[:, 1], 'k-', lw=2, label='Trajectory')
plt.xlabel('Membrane Potential V (mV)')
plt.ylabel('Recovery Variable w')
plt.title('Modified Morris–Lecar Phase Plane')
plt.legend()
plt.xlim([-80, 80])
plt.ylim([0, 1])
plt.grid()
plt.show()
