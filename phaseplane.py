import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from typing import Tuple, List


class PrescottModel:
    def __init__(self, bw: int, I_stim: int) -> None:

        self.bw = bw
        self.I_stim = I_stim

        self.gfast: float = 20
        self.gslow: float = 20
        self.gleak: float = 2
        self.ENa: float = 50
        self.EK: float = -100
        self.Eleak: float = -70
        self.C: float = 2
        self.bm: float = -1.2
        self.cm: float = 18
        self.cw: float = 10
        self.ww: float = 0.15

    def v_nullcline(self, V) -> np.ndarray:

        m_inf: float = 0.5 * (1 + np.tanh((V - self.bm) / self.cm))

        return ((self.I_stim - (self.gfast * m_inf * (V - self.ENa)) - (self.gleak * (V - self.Eleak))) /
                (self.gslow * (V - self.EK)))

    def w_nullcline(self, V) -> np.ndarray:
        return 0.5 * (1 + np.tanh((V - self.bw) / self.cw))

    def ode_system(self, state: List[float], t: float) -> List[float]:
        V, w = state
        m = 0.5 * (1 + np.tanh((V - self.bm) / self.cm))
        w_inf = 0.5 * (1 + np.tanh((V - self.bw) / self.cw))
        lambda_w = self.ww * np.cosh((V - self.bw) / (2 * self.cw))

        dvdt = (self.I_stim
                - self.gfast * m * (V - self.ENa)
                - self.gleak * (V - self.Eleak)
                - self.gslow * w * (V - self.EK)) / self.C
        dwdt = lambda_w * (w_inf - w)
        return [dvdt, dwdt]

    def trajectory(self, t: np.ndarray, init_state: List[float]):

        return odeint(self.ode_system, init_state, t)

    def plot_nullclines_and_trajectory(self, init_state: List[float], t_range: Tuple[float, float],
                                       v_range: Tuple[float, float]) -> None:

        v_values = np.linspace(v_range[0], v_range[1], 300)
        v_null = self.v_nullcline(v_values)
        w_null = self.w_nullcline(v_values)

        t = np.linspace(t_range[0], t_range[1], 10000)
        sol = self.trajectory(t, init_state)

        plt.figure(figsize=(8, 6))
        plt.plot(v_values, v_null, 'r', linewidth=2.5, label='V nullcline')
        plt.plot(v_values, w_null, 'b', linewidth=2.5, label=f'w nullcline (bw={self.bw} mV)')
        plt.plot(sol[:, 0], sol[:, 1], 'k-', linewidth=2, label='Trajectory')

        plt.xlabel("Membrane Potential V (mV)")
        plt.ylabel("Recovery Variable w")
        plt.title("Prescott Phase Plane: Nullclines & Trajectory")
        plt.legend()
        plt.grid(True)
        plt.ylim(-0.1, 1)
        plt.tight_layout()
        plt.show()


def main():
    bw = 0
    I_stim = 37
    init_state = [-60, 0]
    t_range = (0, 1000)
    V_range = (-80, 50)

    model = PrescottModel(bw=bw, I_stim=I_stim)
    model.plot_nullclines_and_trajectory(init_state, t_range, V_range)


if __name__ == "__main__":
    main()

