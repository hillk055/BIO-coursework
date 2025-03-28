Mistaken appendix upload (please read)


I’ve spoken to Mark about the possibility of disregarding the appendix and looking at this one instead, and he said that would be okay. 
The reason for this is that I accidentally uploaded the wrong PDF, where I just had a collection of ideas. 


--------------------------------------------------------------------------------------------------------------------------
Appendix that was supposed to be submitted.

To plot the phase plane we use python libraries such as numpy for vectorised math operations, matplotlib.pyplot for visualisation, 
and scipy.integrate.odeint for numerically solving the system of ordinary differential equations (ODEs). The model is wrapped in a class 
called PrescottModel, which is initialised with a dictionary of the parameters given in the Prescott paper. The method v_nullcline 
calculates the V-nullcline where membrane voltage doesn't change. The method w_nullcline calculates the w-nullcline where the gating 
variable doesn't change The core dynamics are defined in prescott_ode, which implements the Prescott formulation. simulate_trajectory
integrates the model forward in time from an initial condition to generate a trajectory in phase space. Finally, 
plot_nullclines_and_trajectory plots the V- and w-nullclines along with the trajectory, providing a phase-plane view of the system. 
