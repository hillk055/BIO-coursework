import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from typing import Tuple, List


class PrescottModel:

    def __init__(self, bw: float, I_stim: float) -> None:

        self.I_stim = I_stim
        self.b_w = bw

        # Fixed model parameters
        self.g_fast: int = 20
        self.g_slow: int = 20
        self.g_leak: int = 2
        self.E_Na: int = 50
        self.E_K: int = -100
        self.E_leak: int = -70
        self.C: int = 2
        self.b_m: float = -1.2
        self.c_m: int = 18
        self.c_w: int = 10
        self.w_w: float = 0.15

        self.dwdt = None
        self.dvdt = None

    def prescott_equations(self, y: Tuple[float, float], t: float, I: float) -> List[float]:
        V, w = y

        m_inf = 0.5 * (1 + np.tanh((V - self.b_m) / self.c_m))
        w_inf = 0.5 * (1 + np.tanh((V - self.b_w) / self.c_w))
        tau_w = 1.0 / np.cosh((V - self.b_w) / (2.0 * self.c_w))

        dVdt = (1.0 / self.C) * (I - self.g_fast * m_inf * (V - self.E_Na)
                                 - self.g_slow * w * (V - self.E_K)
                                 - self.g_leak * (V - self.E_leak))
        dwdt = self.w_w * (w_inf - w) / tau_w

        return [dVdt, dwdt]

    def compute_jacobian(self, V: float, w: float, I: float) -> np.ndarray:
        dm_dV = 0.5 / self.c_m / np.cosh((V - self.b_m) / self.c_m) ** 2
        dw_dV = 0.5 / self.c_w / np.cosh((V - self.b_w) / self.c_w) ** 2
        tau_w = 1.0 / np.cosh((V - self.b_w) / (2 * self.c_w))
        dtau_dV = -np.sinh((V - self.b_w) / (2 * self.c_w)) / (2 * self.c_w * np.cosh((V - self.b_w) / (2 * self.c_w)) ** 2)

        m_inf = 0.5 * (1 + np.tanh((V - self.b_m) / self.c_m))
        w_inf = 0.5 * (1 + np.tanh((V - self.b_w) / self.c_w))

        dI_fast_dV = self.g_fast * (dm_dV * (V - self.E_Na) + m_inf)
        dI_slow_dV = self.g_slow * w
        dI_leak_dV = self.g_leak

        df1dV = -(dI_fast_dV + dI_slow_dV + dI_leak_dV) / self.C
        df1dw = -self.g_slow * (V - self.E_K) / self.C

        part1 = dw_dV / tau_w
        part2 = (w_inf - w) * (-dtau_dV / tau_w ** 2)
        df2dV = self.w_w * (part1 + part2)
        df2dw = -self.w_w / tau_w

        return np.array([[df1dV, df1dw], [df2dV, df2dw]])

    def find_fixed_points(self, I: float) -> List[Tuple[float, float, bool]]:
        V_guesses = np.linspace(-80, 40, 6)
        w_guesses = np.linspace(0, 1, 4)
        initial_guesses = [[v, w] for v in V_guesses for w in w_guesses]

        fixed_points = []

        for guess in initial_guesses:
            sol = fsolve(lambda y: self.prescott_equations(y, 0, I), guess, xtol=1e-6)
            residual = np.linalg.norm(self.prescott_equations(sol, 0, I))

            if residual < 1e-5 and not any(np.linalg.norm(sol - fp) < 1e-2 for fp in fixed_points):
                fixed_points.append(sol)

        results = []
        for V_fp, w_fp in fixed_points:
            J = self.compute_jacobian(V_fp, w_fp, I)
            eigvals = np.linalg.eigvals(J)
            stable = np.all(np.real(eigvals) < 0)
            results.append((V_fp, w_fp, stable))

        return results

    def plot_bifurcation(self, i_range: Tuple[float, float] = (0, 60), steps: int = 61) -> None:
        i_values = np.linspace(*i_range, steps)
        lc_min = np.full(steps, np.nan)
        lc_max = np.full(steps, np.nan)
        stable_pts, unstable_pts = [], []

        initial_voltage = -70.0
        recovery_initial_value = 0.0

        for i, I in enumerate(i_values):
            fps = self.find_fixed_points(I)
            stable_pts.append([V for V, _, s in fps if s])
            unstable_pts.append([V for V, _, s in fps if not s])

            t = np.linspace(0, 300, 2000)
            sol = odeint(self.prescott_equations, [initial_voltage, recovery_initial_value], t, args=(I,))
            V_trace = sol[:, 0]
            V_post = V_trace[t > 200]

            if V_post.max() - V_post.min() > 2.0:
                lc_min[i], lc_max[i] = V_post.min(), V_post.max()

            initial_voltage, recovery_initial_value = sol[-1]

        plt.figure(figsize=(7, 5))
        for i, I in enumerate(i_values):
            plt.plot([I] * len(stable_pts[i]), stable_pts[i], 'ko')
            plt.plot([I] * len(unstable_pts[i]), unstable_pts[i], 'kx')

        has_lc = ~np.isnan(lc_min)
        plt.plot(i_values[has_lc], lc_min[has_lc], 'r.', label='LC min')
        plt.plot(i_values[has_lc], lc_max[has_lc], 'r.', label='LC max')

        plt.xlabel('I_stim (µA/cm²)')
        plt.ylabel('Membrane Potential (mV)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.show()

def main():
    model = PrescottModel(10, 60)
    model.plot_bifurcation(i_range=(0, 60), steps=61)

if __name__ == "__main__":
    main()


