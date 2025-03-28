import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class MorrisLecarModel:
    def __init__(self, parameters):
        self.params = parameters

    def v_nullcline(self, V):
        p = self.params
        m_inf = 0.5 * (1 + np.tanh((V - p["bm"]) / p["cm"]))
        I_Na = p["gfast"] * m_inf * (V - p["ENa"])
        I_leak = p["gleak"] * (V - p["Eleak"])
        return (p["I_stim"] - I_Na - I_leak) / (p["gslow"] * (V - p["EK"]))

    def w_nullcline(self, V, bw):
        p = self.params
        return 0.5 * (1 + np.tanh((V - bw) / p["cw"]))

    def morris_lecar_ode(self, state, t, bw):
        p = self.params
        V, w = state
        m = 0.5 * (1 + np.tanh((V - p["bm"]) / p["cm"]))
        w_inf = 0.5 * (1 + np.tanh((V - bw) / p["cw"]))
        lambda_w = p["ww"] * np.cosh((V - bw) / (2 * p["cw"]))
        dVdt = (p["I_stim"] - p["gfast"] * m * (V - p["ENa"]) -
                p["gleak"] * (V - p["Eleak"]) - p["gslow"] * w * (V - p["EK"])) / p["C"]
        dwdt = lambda_w * (w_inf - w)
        return [dVdt, dwdt]

    def simulate_trajectory(self, t, init_state, bw):
        sol = odeint(self.morris_lecar_ode, init_state, t, args=(bw,))
        return sol

    def plot_nullclines_and_trajectory(self, bw, init_state, t_range, V_range):
        V_values = np.linspace(V_range[0], V_range[1], 300)
        w_v_null = self.v_nullcline(V_values)
        w_w_null = self.w_nullcline(V_values, bw)

        plt.figure(figsize=(8, 6))
        plt.plot(V_values, w_v_null, 'r', linewidth=2.5, label='V-nullcline')
        plt.plot(V_values, w_w_null, 'b', linewidth=2.5, label=f'w-nullcline (bw={bw} mV)')

        t = np.linspace(t_range[0], t_range[1], 10000)
        sol = self.simulate_trajectory(t, init_state, bw)
        plt.plot(sol[:, 0], sol[:, 1], 'k-', linewidth=2, label='Trajectory')

        plt.xlabel("Membrane Potential V (mV)")
        plt.ylabel("Recovery Variable w")
        plt.legend()
        plt.grid(True)
        plt.ylim(-0.1, 1)
        plt.title("Phase Plane: Nullclines & Trajectory")
        plt.tight_layout()
        plt.show()


def main():
    parameters = {
        "gfast": 20, "gslow": 20, "gleak": 2,
        "ENa": 50, "EK": -100, "Eleak": -70,
        "C": 2, "bm": -1.2, "cm": 18, "cw": 10, "ww": 0.15,
        "I_stim": 37
    }
    bw = 0
    init_state = [-60, 0]
    t_range = [0, 1000]
    V_range = [-80, 50]

    model = MorrisLecarModel(parameters)
    model.plot_nullclines_and_trajectory(bw, init_state, t_range, V_range)


if __name__ == "__main__":
    main()
