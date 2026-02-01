import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv(
    "yamuna_forcing_data_with_metadata.csv",
    skiprows=3,
    encoding="latin1"
)


H = 8.0          
dz = 0.2           
N = int(H/dz)
dt = 30             
days_to_sim = 1000


T_min, T_opt, T_max = 0.0, 26.0, 40.0 
mu_max = 0.8 / 86400  
m = 0.05 / 86400      
Ks = 0.015            
kw = 0.5              
kc = 0.05             
I_opt = 150           
Dz = 1e-4            

def phi_temp(T):
    if T <= T_min or T >= T_max: return 0.0
    num = (T - T_max) * (T - T_min)**2
    den = (T_opt - T_min) * ((T_opt - T_min)*(T - T_opt) - (T_opt - T_max)*(T_opt + T_min - 2*T))
    return max(0.0, num / den)


X = np.ones(N) * 0.1  
depths = np.linspace(0, H, N)
biomass_heatmap = np.zeros((days_to_sim, N))


print(f"Starting {days_to_sim}-Day Run (8m Depth)...")

for day in range(days_to_sim):
    row = data.iloc[day % len(data)]
    T_curr = row['Temperature_C']
    I_surf = (row['Solar_MJ_m2_day'] * 1e6) / 86400 
    S_curr = row['PO4_mg_L']
    phi_T = phi_temp(T_curr)
    
    for _ in range(int(86400/dt)):
        
        cum_shading = np.cumsum(X) * dz
        I_z = I_surf * np.exp(-(kw * depths + kc * cum_shading))
        
        
        phi_nut = S_curr / (S_curr + Ks)
        phi_light = (I_z / I_opt) * np.exp(1 - I_z / I_opt)
        net_rate = (mu_max * phi_T * phi_nut * phi_light) - m
        
        X_new = X.copy()
        for j in range(1, N-1):
            
            mixing = Dz * (X[j+1] - 2*X[j] + X[j-1]) / (dz**2)
            
            
            vp = 0.2/3600 if I_z[j] < 50 else -0.2/3600
            if vp > 0: 
                advection = vp * (X[j] - X[j+1]) / dz
            else:      
                advection = vp * (X[j-1] - X[j]) / dz
            
            X_new[j] = X[j] + dt * (net_rate[j] * X[j] + mixing - advection)
        
        X = np.maximum(X_new, 1e-6) 
        X[0], X[-1] = X[1], X[-2]
    
    biomass_heatmap[day, :] = X
    
    if (day + 1) % 100 == 0:
        print(f"Day {day+1:04d} | Mean Biomass: {np.mean(X):.4f}")


plt.figure(figsize=(15, 8))


plt.imshow(biomass_heatmap.T, aspect='auto', origin='upper', 
           extent=[0, days_to_sim, H, 0], cmap='viridis')

plt.colorbar(label='Biomass (mg/L)')
plt.title(f"Cyanobacteria Vertical Migration: 8m Deep ", fontsize=14)
plt.xlabel("Time (Days)")
plt.ylabel("Depth (m)")

plt.savefig("migration_8m.jpg", dpi=200)
plt.show()