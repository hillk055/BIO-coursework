import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Morris-Lecar Model Parameters
def morris_lecar(t, Y, I_stim, gfast, gslow, gleak, ENa, EK, Eleak, C, bw, ww, bm, cm, cw):
    V, w = Y

    # Activation functions
    m_inf = 0.5 * (1 + np.tanh((V - bm) / cm))
    w_inf = 0.5 * (1 + np.tanh((V - bw) / cw))
    tau_w = 1 / np.cosh((V - bw) / (2 * cw))

    # Ionic currents
    I_Na = gfast * m_inf * (V - ENa)
    I_K = gslow * w * (V - EK)
    I_leak = gleak * (V - Eleak)

    # Differential equations
    dVdt = (I_stim - I_Na - I_K - I_leak) / C
    dwdt = ww * (w_inf - w) / tau_w

    return [dVdt, dwdt]


# Time parameters
t_span = [0, 200]
t_eval = np.linspace(*t_span, 1000)

# Fixed Parameters
params = {
    "gfast": 20, "gslow": 20, "gleak": 2,
    "ENa": 50, "EK": -100, "Eleak": -70, "C": 2,
    "bm": -1.2, "cm": 18, "cw": 10, "ww": 0.15
}

# Class-specific parameters
class_params = {
    "Class 1": {"bw": 0, "I_stim": 40},  # Low-frequency spiking
    "Class 2": {"bw": -13, "I_stim": 40},  # Abrupt frequency jump
    "Class 3": {"bw": -21, "I_stim": 20},  # Single transient spike
}

plt.figure(figsize=(10, 8))

for i, (exc_class, c_params) in enumerate(class_params.items()):
    sol = solve_ivp(morris_lecar, t_span, [-70, 0], t_eval=t_eval, args=(
        c_params["I_stim"], params["gfast"], params["gslow"], params["gleak"],
        params["ENa"], params["EK"], params["Eleak"], params["C"],
        c_params["bw"], params["ww"], params["bm"], params["cm"], params["cw"]
    ))

    plt.subplot(3, 1, i + 1)
    plt.plot(sol.t, sol.y[0], label=f"{exc_class} - V (Membrane Potential)", color='b')
    plt.plot(sol.t, sol.y[1], label=f"{exc_class} - w (Recovery Variable)", color='r', linestyle='dashed')
    plt.ylabel("State Variables")
    plt.title(f"{exc_class} Excitability")
    plt.legend()

plt.xlabel("Time (ms)")
plt.tight_layout()
plt.show()

# Explanation of Parameters and Behavior:
# - "Class 1" exhibits continuous spiking with a low threshold.
# - "Class 2" has an abrupt transition to spiking at a minimum frequency.
# - "Class 3" generates only a single spike before stabilizing.
# - The membrane potential (V) and the recovery variable (w) show how the neuron responds dynamically to stimulation.
