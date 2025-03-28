import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

def prescott_rhs(y, t, I_stim,
                 C, g_fast, g_slow, g_leak,
                 E_Na, E_K, E_leak,
                 b_m, c_m, b_w, c_w, w_w):
    V, w = y
    m_inf = 0.5 * (1 + np.tanh((V - b_m) / c_m))
    w_inf = 0.5 * (1 + np.tanh((V - b_w) / c_w))
    tau_w = 1.0 / np.cosh((V - b_w) / (2.0 * c_w))
    I_fast = g_fast * m_inf * (V - E_Na)
    I_slow = g_slow * w * (V - E_K)
    I_leak = g_leak * (V - E_leak)
    dVdt = (1.0 / C) * (I_stim - I_fast - I_slow - I_leak)
    dwdt = w_w * (w_inf - w) / tau_w
    return [dVdt, dwdt]

def prescott_f(Vw, I_stim,
               C, g_fast, g_slow, g_leak,
               E_Na, E_K, E_leak,
               b_m, c_m, b_w, c_w, w_w):
    V, w = Vw
    m_inf = 0.5 * (1 + np.tanh((V - b_m) / c_m))
    w_inf = 0.5 * (1 + np.tanh((V - b_w) / c_w))
    tau_w = 1.0 / np.cosh((V - b_w) / (2.0 * c_w))
    I_fast = g_fast * m_inf * (V - E_Na)
    I_slow = g_slow * w * (V - E_K)
    I_leak = g_leak * (V - E_leak)
    f1 = (1.0 / C) * (I_stim - I_fast - I_slow - I_leak)
    f2 = w_w * (w_inf - w) / tau_w
    return [f1, f2]

def jacobian_prescott(V, w, I_stim,
                      C, g_fast, g_slow, g_leak,
                      E_Na, E_K, E_leak,
                      b_m, c_m, b_w, c_w, w_w):
    dm_inf_dV = 0.5 * (1.0 / c_m) * (1.0 / np.cosh((V - b_m) / c_m) ** 2)
    dw_inf_dV = 0.5 * (1.0 / c_w) * (1.0 / np.cosh((V - b_w) / c_w) ** 2)
    denom = np.cosh((V - b_w) / (2 * c_w))
    dtau_w_dV = - (1.0 / (2 * c_w)) * np.sinh((V - b_w) / (2 * c_w)) * (1.0 / denom ** 2)
    m_inf = 0.5 * (1 + np.tanh((V - b_m) / c_m))
    w_inf = 0.5 * (1 + np.tanh((V - b_w) / c_w))
    tau_w = 1.0 / denom
    I_fast = g_fast * m_inf * (V - E_Na)
    I_slow = g_slow * w * (V - E_K)
    I_leak = g_leak * (V - E_leak)
    dI_fast_dV = g_fast * (dm_inf_dV * (V - E_Na) + m_inf)
    dI_slow_dV = g_slow * w
    dI_leak_dV = g_leak
    df1dV = (1.0 / C) * (-dI_fast_dV - dI_slow_dV - dI_leak_dV)
    dI_fast_dw = 0.0
    dI_slow_dw = g_slow * (V - E_K)
    dI_leak_dw = 0.0
    df1dw = (1.0 / C) * (-dI_fast_dw - dI_slow_dw - dI_leak_dw)
    partA = dw_inf_dV / tau_w
    partB = (w_inf - w) * (-1.0 / tau_w ** 2 * dtau_w_dV)
    df2dV = w_w * (partA + partB)
    df2dw = w_w * (-1.0 / tau_w)
    return np.array([[df1dV, df1dw],
                     [df2dV, df2dw]])

def find_fixed_points(I_stim,
                      C, g_fast, g_slow, g_leak,
                      E_Na, E_K, E_leak,
                      b_m, c_m, b_w, c_w, w_w,
                      guesses=None,
                      tol=1e-6, max_iter=100):
    if guesses is None:
        V_vals = np.linspace(-80, 40, 8)
        w_vals = np.linspace(0, 1, 5)
        guesses = []
        for vv in V_vals:
            for ww in w_vals:
                guesses.append([vv, ww])
    sol_list = []
    for guess in guesses:
        sol = fsolve(prescott_f, guess,
                     args=(I_stim,
                           C, g_fast, g_slow, g_leak,
                           E_Na, E_K, E_leak,
                           b_m, c_m, b_w, c_w, w_w),
                     xtol=tol, maxfev=max_iter)
        res = prescott_f(sol, I_stim,
                         C, g_fast, g_slow, g_leak,
                         E_Na, E_K, E_leak,
                         b_m, c_m, b_w, c_w, w_w)
        if np.linalg.norm(res) < 1e-5:
            already_in_list = False
            for s in sol_list:
                dist = np.linalg.norm(s[0:2] - sol)
                if dist < 1e-3:
                    already_in_list = True
                    break
            if not already_in_list:
                sol_list.append(sol)
    results = []
    for s in sol_list:
        Vfp, wfp = s
        J = jacobian_prescott(Vfp, wfp, I_stim,
                              C, g_fast, g_slow, g_leak,
                              E_Na, E_K, E_leak,
                              b_m, c_m, b_w, c_w, w_w)
        eigvals = np.linalg.eigvals(J)
        stable = np.all(np.real(eigvals) < 0.0)
        results.append((Vfp, wfp, stable))
    return results

def plot_bifurcation(C=2.0, g_fast=20.0, g_slow=7.0, g_leak=2.0, E_Na=50.0, E_K=-100.0, E_leak=-70.0,
b_m=-1.2, c_m=18.0, b_w=-13.0, c_w=10.0, w_w=0.15, I_min=0.0, I_max=60.0, n_steps=61, tmax=300.0, t_transient=200.0):
    Ivals = np.linspace(I_min, I_max, n_steps)
    eq_stable_V = []
    eq_unstab_V = []
    lc_min = np.full(n_steps, np.nan)
    lc_max = np.full(n_steps, np.nan)
    V0 = -70.0
    w0 = 0.0
    for i, I_stim in enumerate(Ivals):
        fps = find_fixed_points(I_stim,
                                C, g_fast, g_slow, g_leak,
                                E_Na, E_K, E_leak,
                                b_m, c_m, b_w, c_w, w_w,
                                guesses=None)
        stable_V_list = []
        unstab_V_list = []
        for (Vfp, wfp, is_stable) in fps:
            if is_stable:
                stable_V_list.append(Vfp)
            else:
                unstab_V_list.append(Vfp)
        eq_stable_V.append(stable_V_list)
        eq_unstab_V.append(unstab_V_list)
        t_span = np.linspace(0, tmax, 2000)
        sol = odeint(prescott_rhs, [V0, w0], t_span,
                     args=(I_stim,
                           C, g_fast, g_slow, g_leak,
                           E_Na, E_K, E_leak,
                           b_m, c_m, b_w, c_w, w_w))
        V_trace = sol[:, 0]
        w_trace = sol[:, 1]
        idx_trans = np.where(t_span > t_transient)[0]
        if len(idx_trans) == 0:
            idx_trans = [0]
        V_post = V_trace[idx_trans]
        v_range = V_post.max() - V_post.min()
        if v_range > 2.0:
            lc_min[i] = V_post.min()
            lc_max[i] = V_post.max()
        else:
            lc_min[i] = np.nan
            lc_max[i] = np.nan
        V0 = V_trace[-1]
        w0 = w_trace[-1]
    plt.figure(figsize=(7, 5))
    for i, I_stim in enumerate(Ivals):
        stables = eq_stable_V[i]
        unstables = eq_unstab_V[i]
        for vs in stables:
            plt.plot(I_stim, vs, 'ko')
        for vu in unstables:
            plt.plot(I_stim, vu, 'kx', alpha=0.7)
    mask_spk = ~np.isnan(lc_min)
    plt.plot(Ivals[mask_spk], lc_min[mask_spk], 'r.', label='LC min')
    plt.plot(Ivals[mask_spk], lc_max[mask_spk], 'r.', label='LC max')
    plt.xlabel('I_{stim} (µA/cm^2)')
    plt.ylabel('Voltage (mV)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    bw = -10
    plot_bifurcation(b_w=bw, I_min=0.0, I_max=60.0, n_steps=61, tmax=300.0, t_transient=200.0)
