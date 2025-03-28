import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

# params

C = 2.0
g_fast = 20.0
g_slow = 7.0
g_leak = 2.0
E_Na = 50.0
E_K = -100.0
E_leak = -70.0
b_m = -1.2
c_m = 18.0
b_w = -13.0
c_w = 10.0
w_w = 0.15


def prescott_rhs(y, t, I):

    V, w = y

    m_inf = 0.5 * (1 + np.tanh((V - b_m) / c_m))
    w_inf = 0.5 * (1 + np.tanh((V - b_w) / c_w))
    tau_w = 1.0 / np.cosh((V - b_w) / (2.0 * c_w))

    dVdt = (1.0 / C) * (I - g_fast * m_inf * (V - E_Na)
                        - g_slow * w * (V - E_K)
                        - g_leak * (V - E_leak))
    dwdt = w_w * (w_inf - w) / tau_w

    return [dVdt, dwdt]


def compute_jacobian(V, w, I):

    """Jacobian matrix for stability analysis at a fixed point"""
    dm_dV = 0.5 / c_m / np.cosh((V - b_m) / c_m) ** 2
    dw_dV = 0.5 / c_w / np.cosh((V - b_w) / c_w) ** 2
    tau_w = 1.0 / np.cosh((V - b_w) / (2 * c_w))
    dtau_dV = -np.sinh((V - b_w) / (2 * c_w)) / (2 * c_w * np.cosh((V - b_w) / (2 * c_w)) ** 2)

    m_inf = 0.5 * (1 + np.tanh((V - b_m) / c_m))
    w_inf = 0.5 * (1 + np.tanh((V - b_w) / c_w))

    dI_fast_dV = g_fast * (dm_dV * (V - E_Na) + m_inf)
    dI_slow_dV = g_slow * w
    dI_leak_dV = g_leak

    df1dV = -(dI_fast_dV + dI_slow_dV + dI_leak_dV) / C
    df1dw = -g_slow * (V - E_K) / C

    part1 = dw_dV / tau_w
    part2 = (w_inf - w) * (-dtau_dV / tau_w ** 2)
    df2dV = w_w * (part1 + part2)
    df2dw = -w_w / tau_w

    return np.array([[df1dV, df1dw],
                     [df2dV, df2dw]])


def find_fixed_points(I):

    """Find all unique fixed points for a given input current"""
    # Generate some reasonable initial guesses
    V_guesses = np.linspace(-80, 40, 6)
    w_guesses = np.linspace(0, 1, 4)
    initial_guesses = [[v, w] for v in V_guesses for w in w_guesses]

    fixed_points = []

    for guess in initial_guesses:
        sol = fsolve(lambda y: prescott_rhs(y, 0, I), guess, xtol=1e-6)
        residual = np.linalg.norm(prescott_rhs(sol, 0, I))

        # Check if it's a valid and unique fixed point
        if residual < 1e-5 and not any(np.linalg.norm(sol - fp) < 1e-2 for fp in fixed_points):
            fixed_points.append(sol)

    results = []
    for V_fp, w_fp in fixed_points:
        J = compute_jacobian(V_fp, w_fp, I)
        eigvals = np.linalg.eigvals(J)
        stable = np.all(np.real(eigvals) < 0)
        results.append((V_fp, w_fp, stable))

    return results


def plot_bifurcation(i_values: list = (0, 60), steps: int= 61):
    """Plot bifurcation diagram showing fixed points and limit cycles"""
    i_values = np.linspace(*i_values, steps)
    lc_min = np.full(steps, np.nan)
    lc_max = np.full(steps, np.nan)
    stable_pts, unstable_pts = [], []

    initial_voltage, recovery_initial_value = -70.0, 0.0  # initial condition

    for i, I in enumerate(i_values):
        fps = find_fixed_points(I)
        stable_pts.append([V for V, _, s in fps if s])
        unstable_pts.append([V for V, _, s in fps if not s])

        t = np.linspace(0, 300, 2000)
        sol = odeint(prescott_rhs, [initial_voltage, recovery_initial_value], t, args=(I,))
        V_trace = sol[:, 0]
        V_post = V_trace[t > 200]

        if V_post.max() - V_post.min() > 2.0:
            lc_min[i], lc_max[i] = V_post.min(), V_post.max()

        # use the obtained value as th e next initial conditions for the next loop
        initial_voltage, recovery_initial_value = sol[-1]

    #   # plot stuff
    plt.figure(figsize=(7, 5))
    for i, I in enumerate(i_values):
        plt.plot([I] * len(stable_pts[i]), stable_pts[i], 'ko')
        plt.plot([I] * len(unstable_pts[i]), unstable_pts[i], 'kx')

    has_lc = ~np.isnan(lc_min)

    plt.plot(i_values[has_lc], lc_min[has_lc], 'r.', label='LC min')
    plt.plot(i_values[has_lc], lc_max[has_lc], 'r.', label='LC max')

    plt.xlabel('I_stim (µA/cm²)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Bifurcation Diagram - Prescott Neuron Model')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    b_w = -10  # tweak for effect
    plot_bifurcation(I_range=(0, 60), steps=61)
