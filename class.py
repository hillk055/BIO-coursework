import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve


class MorrisLecarModel:

    def __init__(self, parameters):

        self.params = parameters

    def v_nullcline(self, V):

        p = self.params
        m_inf = 0.5 * (1 + np.tanh((V - p["bm"]) / p["cm"]))
        I_Na = p["gfast"] * m_inf * (V - p["ENa"])
        I_leak = p["gleak"] * (V - p["Eleak"])
        # Solve: I_stim - I_Na - I_leak = gslow * w * (V - EK)
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
        w_v_null = self.V_nullcline(V_values)
        w_w_null = self.w_nullcline(V_values, bw)

        plt.figure(figsize=(8, 6))
        plt.plot(V_values, w_v_null, 'r', linewidth=3, label='V-nullcline')
        plt.plot(V_values, w_w_null, 'b', linewidth=3, label=f'w-nullcline (bw={bw} mV)')

        t = np.linspace(t_range[0], t_range[1], 10000)
        sol = self.simulate_trajectory(t, init_state, bw)
        plt.plot(sol[:, 0], sol[:, 1], 'k-', linewidth=3, label='Trajectory')

        plt.xlabel("Membrane Potential V (mV)")
        plt.ylabel("Recovery Variable w")
        plt.legend()
        plt.grid(True)
        plt.ylim(-0.1, 1)
        plt.title("Nullclines & Trajectory")
        plt.show()

    def morris_lecar_rhs(self, y, t, I_stim, b_w):
        p = self.params
        V, w = y
        m_inf = 0.5 * (1 + np.tanh((V - p["bm"]) / p["cm"]))
        w_inf = 0.5 * (1 + np.tanh((V - b_w) / p["cw"]))
        tau_w = 1.0 / np.cosh((V - b_w) / (2.0 * p["cw"]))
        I_fast = p["gfast"] * m_inf * (V - p["ENa"])
        I_slow = p["gslow"] * w * (V - p["EK"])
        I_leak = p["gleak"] * (V - p["Eleak"])
        dVdt = (1.0 / p["C"]) * (I_stim - I_fast - I_slow - I_leak)
        dwdt = p["ww"] * (w_inf - w) / tau_w
        return [dVdt, dwdt]

    def morris_lecar_f(self, Vw, I_stim, b_w):
        p = self.params
        V, w = Vw
        m_inf = 0.5 * (1 + np.tanh((V - p["bm"]) / p["cm"]))
        w_inf = 0.5 * (1 + np.tanh((V - b_w) / p["cw"]))
        tau_w = 1.0 / np.cosh((V - b_w) / (2.0 * p["cw"]))
        I_fast = p["gfast"] * m_inf * (V - p["ENa"])
        I_slow = p["gslow"] * w * (V - p["EK"])
        I_leak = p["gleak"] * (V - p["Eleak"])
        f1 = (1.0 / p["C"]) * (I_stim - I_fast - I_slow - I_leak)
        f2 = p["ww"] * (w_inf - w) / tau_w
        return [f1, f2]

    def jacobian_morris_lecar(self, V, w, I_stim, b_w):
        p = self.params
        dm_inf_dV = 0.5 * (1.0 / p["cm"]) * (1.0 / np.cosh((V - p["bm"]) / p["cm"]) ** 2)
        dw_inf_dV = 0.5 * (1.0 / p["cw"]) * (1.0 / np.cosh((V - b_w) / p["cw"]) ** 2)
        denom = np.cosh((V - b_w) / (2 * p["cw"]))
        dtau_w_dV = - (1.0 / (2 * p["cw"])) * np.sinh((V - b_w) / (2 * p["cw"])) * (1.0 / denom ** 2)
        m_inf = 0.5 * (1 + np.tanh((V - p["bm"]) / p["cm"]))
        w_inf = 0.5 * (1 + np.tanh((V - b_w) / p["cw"]))
        tau_w = 1.0 / denom
        dI_fast_dV = p["gfast"] * (dm_inf_dV * (V - p["ENa"]) + m_inf)
        dI_slow_dV = p["gslow"] * w
        dI_leak_dV = p["gleak"]
        df1dV = (1.0 / p["C"]) * (- dI_fast_dV - dI_slow_dV - dI_leak_dV)
        df1dw = (1.0 / p["C"]) * (- p["gslow"] * (V - p["EK"]))
        partA = dw_inf_dV / tau_w
        partB = (w_inf - w) * (-1.0 / tau_w ** 2 * dtau_w_dV)
        df2dV = p["ww"] * (partA + partB)
        df2dw = p["ww"] * (-1.0 / tau_w)
        return np.array([[df1dV, df1dw], [df2dV, df2dw]])

    def find_fixed_points(self, I_stim, b_w, guesses=None, tol=1e-6, max_iter=100):
        p = self.params
        if guesses is None:
            V_vals = np.linspace(-80, 40, 8)
            w_vals = np.linspace(0, 1, 5)
            guesses = [[vv, ww] for vv in V_vals for ww in w_vals]
        sol_list = []
        for guess in guesses:
            sol = fsolve(self.morris_lecar_f, guess, args=(I_stim, b_w), xtol=tol, maxfev=max_iter)
            res = self.morris_lecar_f(sol, I_stim, b_w)
            if np.linalg.norm(res) < 1e-5:
                if not any(np.linalg.norm(np.array(s) - sol) < 1e-3 for s in sol_list):
                    sol_list.append(sol)
        results = []
        for s in sol_list:
            Vfp, wfp = s
            J = self.jacobian_morris_lecar(Vfp, wfp, I_stim, b_w)
            eigvals = np.linalg.eigvals(J)
            stable = np.all(np.real(eigvals) < 0.0)
            results.append((Vfp, wfp, stable))
        return results

    def plot_bifurcation(self, b_w=0, I_min=0.0, I_max=60.0, n_steps=61, tmax=300.0, t_transient=200.0):
        p = self.params
        Ivals = np.linspace(I_min, I_max, n_steps)
        eq_stable_V = []
        eq_unstab_V = []
        lc_min = np.full(n_steps, np.nan)
        lc_max = np.full(n_steps, np.nan)
        V0 = -70.0
        w0 = 0.0
        for i, I_stim in enumerate(Ivals):
            fps = self.find_fixed_points(I_stim, b_w)
            stable_V_list = []
            unstab_V_list = []
            for (Vfp, wfp, is_stable) in fps:
                if is_stable:
                    stable_V_list.append(Vfp)
                else:
                    unstab_V_list.append(Vfp)
            eq_stable_V.append(stable_V_list)
            eq_unstab_V.append(unstab_V_list)

            # Integrate to detect limit cycles
            t_span = np.linspace(0, tmax, 2000)
            sol = odeint(self.morris_lecar_rhs, [V0, w0], t_span, args=(I_stim, b_w))
            V_trace = sol[:, 0]
            idx_trans = np.where(t_span > t_transient)[0]
            V_post = V_trace[idx_trans] if len(idx_trans) > 0 else V_trace
            if (V_post.max() - V_post.min()) > 2.0:
                lc_min[i] = V_post.min()
                lc_max[i] = V_post.max()
            else:
                lc_min[i] = np.nan
                lc_max[i] = np.nan

            V0 = V_trace[-1]
            w0 = sol[-1, 1]

        plt.figure(figsize=(7, 5))
        for i, I_stim in enumerate(Ivals):
            for vs in eq_stable_V[i]:
                plt.plot(I_stim, vs, 'ko')  # stable fixed points
            for vu in eq_unstab_V[i]:
                plt.plot(I_stim, vu, 'kx', alpha=0.7)  # unstable fixed points
        mask_spk = ~np.isnan(lc_min)
        plt.plot(Ivals[mask_spk], lc_min[mask_spk], 'r.', label='LC min')
        plt.plot(Ivals[mask_spk], lc_max[mask_spk], 'r.', label='LC max')
        plt.xlabel('I_stim (µA/cm²)')
        plt.ylabel('Voltage (mV)')
        plt.title(f'Morris-Lecar Bifurcation (b_w = {b_w} mV)')
        plt.grid(True)
        plt.legend()
        plt.show()

    # -----------------------------
    # Time Series using solve_ivp (for different excitability classes)
    # -----------------------------
    def morris_lecar_ivp(self, t, Y, I_stim, b_w):
        p = self.params
        V, w = Y
        m_inf = 0.5 * (1 + np.tanh((V - p["bm"]) / p["cm"]))
        w_inf = 0.5 * (1 + np.tanh((V - b_w) / p["cw"]))
        tau_w = 1 / np.cosh((V - b_w) / (2 * p["cw"]))
        I_Na = p["gfast"] * m_inf * (V - p["ENa"])
        I_K = p["gslow"] * w * (V - p["EK"])
        I_leak = p["gleak"] * (V - p["Eleak"])
        dVdt = (I_stim - I_Na - I_K - I_leak) / p["C"]
        dwdt = p["ww"] * (w_inf - w) / tau_w
        return [dVdt, dwdt]

    def plot_time_series(self, class_params, t_span=[0, 200], num_points=1000):
        """
        class_params: dictionary with keys like "Class 1", "Class 2", each containing a dictionary
                      with at least keys "bw" and "I_stim".
        """
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        num_classes = len(class_params)
        fig, axes = plt.subplots(num_classes, 1, figsize=(10, 8), sharex=True)
        if num_classes == 1:
            axes = [axes]
        for i, (exc_class, cp) in enumerate(class_params.items()):
            sol = solve_ivp(self.morris_lecar_ivp, t_span, [-70, 0], t_eval=t_eval,
                            args=(cp["I_stim"], cp["bw"]))
            ax = axes[i]
            ax.plot(sol.t, sol.y[0], 'k')
            ax.set_title(exc_class, fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            ax.text(-0.1, 1.05, f"{chr(65 + i)}", transform=ax.transAxes,
                    fontsize=18, fontweight='bold', ha="center", va="center")
        fig.text(0.07, 0.5, "Membrane Potential (mV)", va="center",
                 rotation="vertical", fontsize=20)
        fig.text(0.5, 0.03, "Time (ms)", ha="center", fontsize=20)
        plt.tight_layout()
        plt.show()


def main():

    parameters = {
        "gfast": 20, "gslow": 20, "gleak": 2,
        "ENa": 50, "EK": -100, "Eleak": -70,
        "C": 2, "bm": -1.2, "cm": 18, "cw": 10, "ww": 0.15,
        "I_stim": 37  # default stimulus value
    }

    bw = 0  # chosen bw value (can be varied)
    init_state = [-60, 0]
    t_range = [0, 1000]  # simulation time range (ms)
    V_range = [-80, 50]  # voltage range for nullcline plot

    model = MorrisLecarModel(parameters=parameters)

    model.plot_nullclines_and_trajectory(bw, init_state, t_range, V_range)

    # 2. Bifurcation analysis (find fixed points and limit cycles)
    model.plot_bifurcation(b_w=bw, I_min=0, I_max=60, n_steps=61, tmax=300, t_transient=200)

    # 3. Plot time series for different excitability classes (using solve_ivp)
    class_params = {
        "Class 1": {"bw": -21, "I_stim": 37},  # e.g., low-frequency spiking
        "Class 2": {"bw": -21, "I_stim": 60}  # e.g., abrupt frequency jump
    }
    model.plot_time_series(class_params, t_span=[0, 200], num_points=1000)


if __name__ == "__main__":

    main()
