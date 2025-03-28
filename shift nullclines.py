import numpy as np
import matplotlib.pyplot as plt


params = {
    "gfast": 20,    
    "gslow": 20,  
    "gleak": 2,     
    "ENa": 50,      
    "EK": -100,     
    "Eleak": -70,   
    "C": 2,     
    "bm": -8,       
    "cm": 18,       
    "cw": 10,      
    "ww": 0.15,     
    "I_stim": 37    
}

def V_nullcline(V, params):
  
    m_inf = 0.5 * (1 + np.tanh((V - params["bm"]) / params["cm"]))
    I_Na = params["gfast"] * m_inf * (V - params["ENa"])
    I_leak = params["gleak"] * (V - params["Eleak"])
    return (params["I_stim"] - I_Na - I_leak) / (params["gslow"] * (V - params["EK"]))

def w_nullcline(V, bw, params):
   
    return 0.5 * (1 + np.tanh((V - bw) / params["cw"]))


V_values = np.linspace(-80, 50, 500)

bw1 = 0
bw2 = -20

plt.figure(figsize=(10, 8))


plt.plot(V_values, V_nullcline(V_values, params), 'k-', linewidth=3, label='V-nullcline')

plt.plot(V_values, w_nullcline(V_values, bw1, params), 'r-', linewidth=3, label=f'w-nullcline (bw = {bw1} mV)')

plt.plot(V_values, w_nullcline(V_values, bw2, params), 'r--', linewidth=3, label=f'w-nullcline (bw = {bw2} mV)')

# Final plot settings
plt.xlabel("Membrane Potential V (mV)", fontsize=30)
plt.ylabel("Recovery Variable w", fontsize=30)

plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.legend(fontsize=22.5)
plt.grid(True)
plt.show()
