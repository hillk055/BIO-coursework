import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def prescott_model(t, Y, I_stim, gfast, gslow, gleak, ENa, EK, Eleak, C, bw, ww, bm, cm, cw):
    V, w = Y

    m_inf = 0.5 * (1 + np.tanh((V - bm) / cm))
    w_inf = 0.5 * (1 + np.tanh((V - bw) / cw))
    tau_w = 1 / np.cosh((V - bw) / (2 * cw))  

    I_Na = gfast * m_inf * (V - ENa)
    I_K = gslow * w * (V - EK)
    I_leak = gleak * (V - Eleak)

    dVdt = (I_stim - I_Na - I_K - I_leak) / C
    dwdt = ww * (w_inf - w) / tau_w

    return [dVdt, dwdt]

# Time params
t_span = [0, 200]
t_eval = np.linspace(*t_span, 1000)

params = {
    "gfast": 20, "gslow": 20, "gleak": 2,
    "ENa": 50, "EK": -100, "Eleak": -70, "C": 2,
    "bm": -1.2, "cm": 18, "cw": 10, "ww": 0.15
}

class_params = {
    "Class 1": {"bw": -21, "I_stim": 37},
    "Class 2": {"bw": -21, "I_stim": 60},
}

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
plt.subplots_adjust(hspace=0.4)

for i, (label, c_params) in enumerate(class_params.items()):
    sol = solve_ivp(prescott_model, t_span, [-70, 0], t_eval=t_eval, args=(
        c_params["I_stim"], params["gfast"], params["gslow"], params["gleak"],
        params["ENa"], params["EK"], params["Eleak"], params["C"],
        c_params["bw"], params["ww"], params["bm"], params["cm"], params["cw"]
    ))

    ax = axes[i]
    ax.plot(sol.t, sol.y[0], color='k')
    ax.set_title(label, fontsize=14)
    ax.text(-0.12, 1.05, f"{chr(65+i)}", transform=ax.transAxes,
            fontsize=18, fontweight='bold', ha="center", va="center")
    ax.tick_params(axis='both', labelsize=14)

fig.text(0.068, 0.5, "Membrane Potential (mV)", va="center", rotation="vertical", fontsize=20)
fig.text(0.5, 0.03, "Time (ms)", ha="center", fontsize=20)
plt.show()
