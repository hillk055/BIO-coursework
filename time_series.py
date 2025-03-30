import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class PrescottModel:

    def __init__(self, bw: float, I_stim: float) -> None:

        self.I_stim = I_stim
        self.bw = bw

        self.gfast: int = 20
        self.gslow: int = 20
        self.gleak: int = 2
        self.ENa: int = 50
        self.EK: int = -100
        self.Eleak: int = -70
        self.C: int = 2
        self.bm: float = -1.2
        self.cm: int = 18
        self.cw: int = 10
        self.ww: float = 0.15

        self.dwdt = None
        self.dvdt = None

    def model(self, t: list, Y: list) -> list:

        V, w = Y

        m_inf = 0.5 * (1 + np.tanh((V - self.bm) / self.cm))
        w_inf = 0.5 * (1 + np.tanh((V - self.bw) / self.cw))
        tau_w = 1 / np.cosh((V - self.bw) / (2 * self.cw))

        self.dvdt = (self.I_stim
                - self.gfast * m_inf * (V - self.ENa)
                - self.gslow * w * (V - self.EK)
                - self.gleak * (V - self.Eleak)) / self.C
        self.dwdt = self.ww * (w_inf - w) / tau_w

        return [self.dvdt, self.dwdt]

    def solve(self, t_span, y0, t_eval):

        return solve_ivp(self.model, t_span, y0, t_eval=t_eval)



def main():

    t = np.linspace(0, 200, 1000)

    # Define model instances for different classes
    model_class1 = PrescottModel(bw=-21, I_stim=37)
    model_class3 = PrescottModel(bw=-21, I_stim=60)

    models = [("class 1", model_class1), ("class 2", model_class3)]

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.4)

    for i, (label, model) in enumerate(models):
        sol = model.solve([0, 200], [-70, 0], t_eval=t)

        ax = axes[i]
        ax.plot(sol.t, sol.y[0], color='k')
        ax.set_title(label, fontsize=14)
        ax.text(-0.12, 1.05, f"{chr(65+i)}", transform=ax.transAxes,
                fontsize=18, fontweight='bold', ha="center", va="center")
        ax.tick_params(axis='both', labelsize=14)

    fig.text(0.068, 0.5, "Membrane Potential (mV)", va="center", rotation="vertical", fontsize=20)
    fig.text(0.5, 0.03, "Time (ms)", ha="center", fontsize=20)
    plt.show()


if __name__ == "__main__":

    main()
