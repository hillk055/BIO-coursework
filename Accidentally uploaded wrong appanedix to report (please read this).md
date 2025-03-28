------------------------------------------------------
Please read below :)


I’ve spoken to Mark about the possibly of disregarding the appendix and looking at this one instead, and he said that would be okay. The reason for this is I accidentally uploaded the wrong PDF where I had just a collection of ideas but was not supposed to be anywhere near the final thing. As a result it doesn’t make any sense and there are lots of incomplete sentences etc. Apologies for this but I was wondering if it would be possible to mark the report based on this instead :).

-------------------------------------------------------

Appendix that was supposed to be submitted

To plot the phase plane we use python libraries such as numpy for vectorised math operations, matplotlib.pyplot for visualisation, and scipy.integrate.odeint for numerically solving the system of ordinary differential equations (ODEs). The model is wrapped in a class called PrescottModel, which is initialised with a dictionary of the parameters given in the Prescott paper. The method v_nullcline calculates the V-nullcline where membrane voltage doesn't change. The method w_nullcline calculates the w-nullcline where the gating variable doesn't change The core dynamics are defined in prescott_ode, which implements the Prescott formulation. simulate_trajectory integrates the model forward in time from an initial condition to generate a trajectory in phase space. Finally, plot_nullclines_and_trajectory plots the V- and w-nullclines along with the trajectory, providing a phase-plane view of the system. 

For the state variable time series plots we begin by outlining the parameters again in a dictionary, these are passed as arguments to the prescott_model function. Alongside this we set up a separate dictionary that contains the  different b_w configurations showing class 1 and 3 dynamics. After this we parse ODEs and parameters into solve_ivp function within the scipy.integrate module. This function returns the values of T (time) and Y (membrane voltage) which are then plotted using the matplotlib library similar to the phase plane.

To generate the bifurcation diagram, we begin by outlining the parameter of interest—here, the external input current I stem—which is passed as a range into the plot_bifurcation function. The function iterates over this range, and for each value of I, it attempts to identify all fixed points using the find_fixed_points function. This is done by supplying a some initial guesses to fsolve, which uses the Prescott model equations to determine equilibrium points. 
.
Once the fixed points are identified, a Jacobian matrix is computed at each point using the compute_jacobian function. This is used to calculate eigenvalues and determine the stability of the fixed points. Stable and unstable fixed points are stored separately.
At the same time, the full system of ODEs is numerically integrated using odeint from the scipy.integrate module, with the same value of I and the most recent solution as the initial condition. The solution's membrane potential V is then examined during the post-transient period (after t = 200) to detect the presence of a limit cycle. If the membrane potential exhibits sustained oscillations (i.e., the range exceeds a threshold), the minimum and maximum values are stored to plot the bounds of the limit cycle.
All this information—stable points, unstable points, and limit cycle bounds—is plotted using matplotlib.
