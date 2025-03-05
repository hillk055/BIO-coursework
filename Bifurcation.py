import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

##############################################################################
# Morris–Lecar RHS and helpers
##############################################################################

def morris_lecar_rhs(y, t, I_stim,
                     C, g_fast, g_slow, g_leak,
                     E_Na, E_K, E_leak,
                     b_m, c_m, b_w, c_w, w_w):
    """
    Right-hand side for the 2D Morris-Lecar system:
      dV/dt = (1/C)*(I_stim - I_fast - I_slow - I_leak)
      dw/dt = w_w * (w_inf - w)/tau_w
    """
    V, w = y

    m_inf = 0.5*(1 + np.tanh((V - b_m)/c_m))
    w_inf = 0.5*(1 + np.tanh((V - b_w)/c_w))
    tau_w = 1.0/np.cosh((V - b_w)/(2.0*c_w))

    I_fast = g_fast*m_inf*(V - E_Na)
    I_slow = g_slow*w*(V - E_K)
    I_leak = g_leak*(V - E_leak)

    dVdt = (1.0/C)*(I_stim - I_fast - I_slow - I_leak)
    dwdt = w_w*(w_inf - w)/tau_w

    return [dVdt, dwdt]

def morris_lecar_f(Vw, I_stim,
                   C, g_fast, g_slow, g_leak,
                   E_Na, E_K, E_leak,
                   b_m, c_m, b_w, c_w, w_w):
    """
    Algebraic function F(V,w) = 0 for fixed points:
      F_1(V,w) = dV/dt
      F_2(V,w) = dw/dt
    We'll pass this to fsolve to find solutions.
    """
    V, w = Vw

    m_inf = 0.5*(1 + np.tanh((V - b_m)/c_m))
    w_inf = 0.5*(1 + np.tanh((V - b_w)/c_w))
    tau_w = 1.0/np.cosh((V - b_w)/(2.0*c_w))

    I_fast = g_fast*m_inf*(V - E_Na)
    I_slow = g_slow*w*(V - E_K)
    I_leak = g_leak*(V - E_leak)

    f1 = (1.0/C)*(I_stim - I_fast - I_slow - I_leak)
    f2 = w_w*(w_inf - w)/tau_w

    return [f1, f2]

def jacobian_morris_lecar(V, w, I_stim,
                          C, g_fast, g_slow, g_leak,
                          E_Na, E_K, E_leak,
                          b_m, c_m, b_w, c_w, w_w):
    """
    Compute the Jacobian matrix J = d(F_1, F_2)/d(V, w) at a fixed point (V,w).
    We'll use partial derivatives of f1(V,w), f2(V,w).
    """
    # Derivatives of the steady-state gating expressions
    # m_inf(V)
    dm_inf_dV = 0.5*(1.0/c_m)* (1.0/np.cosh((V - b_m)/c_m)**2 )
    # w_inf(V)
    dw_inf_dV = 0.5*(1.0/c_w)* (1.0/np.cosh((V - b_w)/c_w)**2 )
    # tau_w(V)
    # tau_w = 1/cosh((V-b_w)/(2*c_w)) => derivative is a bit messy:
    denom = np.cosh((V - b_w)/(2*c_w))
    dtau_w_dV = - (1.0/(2*c_w)) * np.sinh((V - b_w)/(2*c_w)) * (1.0/denom**2)
    # rewrite for clarity:
    # dtau_w_dV = derivative of 1/denom wrt V => -1/denom^2 * d(denom)/dV
    # d(denom)/dV = (1/(2*c_w)) * sinh((V-b_w)/(2*c_w))

    m_inf = 0.5*(1 + np.tanh((V - b_m)/c_m))
    w_inf = 0.5*(1 + np.tanh((V - b_w)/c_w))
    tau_w = 1.0/denom

    # Currents
    I_fast = g_fast*m_inf*(V - E_Na)
    I_slow = g_slow*w*(V - E_K)
    I_leak = g_leak*(V - E_leak)

    # For f1 = (1/C) [ I_stim - I_fast - I_slow - I_leak ]
    # partial wrt V
    dI_fast_dV = g_fast*(
        dm_inf_dV*(V - E_Na) + m_inf
    )
    dI_slow_dV = g_slow*w
    dI_leak_dV = g_leak

    df1dV = (1.0/C)*(
        - dI_fast_dV
        - dI_slow_dV
        - dI_leak_dV
    )
    # partial wrt w
    dI_fast_dw  = 0.0
    dI_slow_dw  = g_slow*(V - E_K)
    dI_leak_dw  = 0.0

    df1dw = (1.0/C)*(
        - dI_fast_dw
        - dI_slow_dw
        - dI_leak_dw
    )

    # For f2 = w_w * (w_inf - w)/tau_w
    # partial wrt V
    # f2 = (w_w/tau_w)*(w_inf - w)
    #     => df2/dV = w_w * [ (dw_inf_dV)/tau_w + (w_inf - w)* d(1/tau_w)/dV ]
    #                - the second term is a product rule
    #                note that w is constant wrt V in partial derivative
    partA = dw_inf_dV/tau_w
    partB = (w_inf - w)* ( -1.0/tau_w**2 * dtau_w_dV )
    df2dV = w_w*(partA + partB)

    # partial wrt w
    # f2 = w_w * (w_inf - w)/tau_w
    #     => df2/dw = w_w*( -1 / tau_w )
    df2dw = w_w*( -1.0/tau_w )

    return np.array([[df1dV, df1dw],
                     [df2dV, df2dw]])


##############################################################################
# Finding fixed points (stable or unstable)
##############################################################################

def find_fixed_points(I_stim,
                      C, g_fast, g_slow, g_leak,
                      E_Na, E_K, E_leak,
                      b_m, c_m, b_w, c_w, w_w,
                      guesses=None,
                      tol=1e-6, max_iter=100):
    """
    For a single I_stim, attempt to find *all* real fixed points by
    passing multiple (V, w) initial guesses to fsolve. Return a list
    of (V*, w*, stability), where stability is True (stable) or False (unstable).

    We classify each solution by the sign of the Jacobian's eigenvalues
    at (V*, w*). Solutions with all negative real parts => stable.
    """
    if guesses is None:
        # Some generic guesses in a (voltage, w) grid.
        # You can adjust these ranges or add more guesses as needed.
        V_vals = np.linspace(-80, 40, 8)
        w_vals = np.linspace(0, 1, 5)
        guesses = []
        for vv in V_vals:
            for ww in w_vals:
                guesses.append([vv, ww])

    sol_list = []
    for guess in guesses:
        sol = fsolve(morris_lecar_f, guess,
                     args=(I_stim,
                           C, g_fast, g_slow, g_leak,
                           E_Na, E_K, E_leak,
                           b_m, c_m, b_w, c_w, w_w),
                     xtol=tol, maxfev=max_iter)
        # Evaluate residual to make sure it's truly a solution
        res = morris_lecar_f(sol, I_stim,
                             C, g_fast, g_slow, g_leak,
                             E_Na, E_K, E_leak,
                             b_m, c_m, b_w, c_w, w_w)
        if np.linalg.norm(res) < 1e-5:
            # Check if we already have this solution (within tolerance)
            # We’ll do a simple check distance < 1e-3 or so
            already_in_list = False
            for s in sol_list:
                dist = np.linalg.norm(s[0:2] - sol)
                if dist < 1e-3:
                    already_in_list = True
                    break
            if not already_in_list:
                sol_list.append(sol)

    # Now classify each solution as stable or unstable
    results = []
    for s in sol_list:
        Vfp, wfp = s
        # compute Jacobian
        J = jacobian_morris_lecar(Vfp, wfp, I_stim,
                                  C, g_fast, g_slow, g_leak,
                                  E_Na, E_K, E_leak,
                                  b_m, c_m, b_w, c_w, w_w)
        eigvals = np.linalg.eigvals(J)
        # stable if real parts are all negative
        stable = np.all(np.real(eigvals) < 0.0)
        results.append( (Vfp, wfp, stable) )

    return results


##############################################################################
# Main function: Sweep I_stim, find stable/unstable fixed pts and stable LC
##############################################################################

def sweep_and_plot_bifurcation(C=2.0,
                               g_fast=20.0,
                               g_slow=7.0,
                               g_leak=2.0,
                               E_Na=50.0,
                               E_K=-100.0,
                               E_leak=-70.0,
                               b_m=-1.2,
                               c_m=18.0,
                               b_w=-13.0,   # pick for Class 1, 2, or 3
                               c_w=10.0,
                               w_w=0.15,
                               I_min=0.0,
                               I_max=60.0,
                               n_steps=61,
                               tmax=300.0,
                               t_transient=200.0):
    """
    1) For each I_stim in [I_min..I_max],
       - Find *all* fixed points, stable or unstable
       - Integrate forward to see if there's a stable limit cycle (and record min/max)
    2) Plot the results:
       - stable FP in solid black
       - unstable FP in dashed black
       - stable LC min/max in red
    """
    Ivals = np.linspace(I_min, I_max, n_steps)

    # Arrays to store stable/unstable eq data
    # We'll store them in list-of-lists format; each I has some solutions
    eq_stable_V = []
    eq_unstab_V = []

    # For the stable limit cycle
    lc_min = np.full(n_steps, np.nan)
    lc_max = np.full(n_steps, np.nan)

    # We'll do a simple approach to time-marching from a fixed initial condition.
    # You could chain from previous solutions if you prefer continuity in param.
    V0 = -70.0
    w0 = 0.0

    for i, I_stim in enumerate(Ivals):
        # -------------------------------------------------------
        # 1) Find all fixed points (stable & unstable)
        # -------------------------------------------------------
        fps = find_fixed_points(I_stim,
                               C, g_fast, g_slow, g_leak,
                               E_Na, E_K, E_leak,
                               b_m, c_m, b_w, c_w, w_w,
                               guesses=None)  # uses a default grid
        stable_V_list = []
        unstab_V_list = []
        for (Vfp, wfp, is_stable) in fps:
            if is_stable:
                stable_V_list.append(Vfp)
            else:
                unstab_V_list.append(Vfp)
        eq_stable_V.append(stable_V_list)
        eq_unstab_V.append(unstab_V_list)

        # -------------------------------------------------------
        # 2) Integrate to see if there's a stable limit cycle
        #    (and measure its min/max voltage)
        # -------------------------------------------------------
        t_span = np.linspace(0, tmax, 2000)
        sol = odeint(morris_lecar_rhs, [V0, w0], t_span,
                     args=(I_stim,
                           C, g_fast, g_slow, g_leak,
                           E_Na, E_K, E_leak,
                           b_m, c_m, b_w, c_w, w_w))
        V_trace = sol[:,0]
        w_trace = sol[:,1]

        # discard transient
        idx_trans = np.where(t_span > t_transient)[0]
        if len(idx_trans) == 0:
            idx_trans = [0]
        V_post = V_trace[idx_trans]

        v_range = V_post.max() - V_post.min()
        if v_range > 2.0:
            # spiking
            lc_min[i] = V_post.min()
            lc_max[i] = V_post.max()
        else:
            lc_min[i] = np.nan
            lc_max[i] = np.nan

        # update (V0, w0) for next iteration
        V0 = V_trace[-1]
        w0 = w_trace[-1]

    # -------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------
    plt.figure(figsize=(7,5))

    # plot stable eq in black solid, unstable eq in black dashed
    for i, I_stim in enumerate(Ivals):
        stables  = eq_stable_V[i]
        unstables= eq_unstab_V[i]
        for vs in stables:
            plt.plot(I_stim, vs, 'ko')  # stable
        for vu in unstables:
            plt.plot(I_stim, vu, 'kx', alpha=0.7)  # mark as unstable

    # plot stable limit cycle in red (min & max)
    mask_spk = ~np.isnan(lc_min)
    plt.plot(Ivals[mask_spk], lc_min[mask_spk], 'r.', label='LC min')
    plt.plot(Ivals[mask_spk], lc_max[mask_spk], 'r.', label='LC max')

    plt.xlabel('I_{stim} (µA/cm^2)')
    plt.ylabel('Voltage (mV)')
    plt.title(f'Morris-Lecar Bifurcation (b_w={b_w} mV)')
    plt.grid(True)
    plt.legend()
    plt.show()

##############################################################################
# Example usage
##############################################################################

if __name__ == "__main__":

    # Try b_w = -5, -13, -21, etc.
    bw = -10  # near Class 2 regime
    sweep_and_plot_bifurcation(b_w=bw,
                               I_min=0.0,
                               I_max=60.0,
                               n_steps=61,
                               tmax=300.0,
                               t_transient=200.0)
