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


kw, kc = 0.5, 0.05    
Dz = 1e-4           
I_opt = 150           
m = 0.05 / 86400      


T_min, T_max = 12.0, 42.0  


mu_tox = 0.7 / 86400  
T_opt_tox = 24.0      


mu_non = 0.9 / 86400  
T_opt_non = 31.0      

def phi_temp_custom(T, T_opt):
    """Cardinal Temperature Model for Genotype Competition"""
    if T <= T_min or T >= T_max: return 0.0
    num = (T - T_max) * (T - T_min)**2
    den = (T_opt - T_min) * ((T_opt - T_min)*(T - T_opt) - (T_opt - T_max)*(T_opt + T_min - 2*T))
    return max(0.0, num / den)


X_tox = np.ones(N) * 0.05   
X_non = np.ones(N) * 0.05   
depths = np.linspace(0, H, N)

history_tox = []
history_non = []


print(f"{'Day':<8} | {'Temp':<8} | {'Toxic (mg/L)':<15} | {'Non-Toxic':<15} | {'Dominant'}")
print("-" * 75)

for day in range(days_to_sim):
    
    row = data.iloc[day % len(data)]
    T_curr = row['Temperature_C']
    I_surf = (row['Solar_MJ_m2_day'] * 1e6) / 86400 
    S_curr = row['PO4_mg_L']
    
    phi_tox = phi_temp_custom(T_curr, T_opt_tox)
    phi_non = phi_temp_custom(T_curr, T_opt_non)
    
   
    for _ in range(int(86400/dt)):
        
        X_total = X_tox + X_non
        cum_shading = np.cumsum(X_total) * dz
        I_z = I_surf * np.exp(-(kw * depths + kc * cum_shading))
        
        
        phi_nut = S_curr / (S_curr + 0.015)
        phi_light = (I_z / I_opt) * np.exp(1 - I_z / I_opt)
        
        net_tox = (mu_tox * phi_tox * phi_nut * phi_light) - m
        net_non = (mu_non * phi_non * phi_nut * phi_light) - m
        
        X_new_tox = X_tox.copy()
        X_new_non = X_non.copy()
        
        
        for j in range(1, N-1):
            
            vp = 0.2/3600 if I_z[j] < 50 else -0.2/3600
            
            
            mix_t = Dz * (X_tox[j+1] - 2*X_tox[j] + X_tox[j-1]) / (dz**2)
            if vp > 0: adv_t = vp * (X_tox[j] - X_tox[j+1])/dz 
            else:      adv_t = vp * (X_tox[j-1] - X_tox[j])/dz
            X_new_tox[j] = X_tox[j] + dt * (net_tox[j] * X_tox[j] + mix_t - adv_t)
            
            mix_n = Dz * (X_non[j+1] - 2*X_non[j] + X_non[j-1]) / (dz**2)
            if vp > 0: adv_n = vp * (X_non[j] - X_non[j+1])/dz 
            else:      adv_n = vp * (X_non[j-1] - X_non[j])/dz
            X_new_non[j] = X_non[j] + dt * (net_non[j] * X_non[j] + mix_n - adv_n)

        
        X_tox = np.maximum(X_new_tox, 1e-6)
        X_non = np.maximum(X_new_non, 1e-6)
        X_tox[0], X_tox[-1] = X_tox[1], X_tox[-2]
        X_non[0], X_non[-1] = X_non[1], X_non[-2]

    
    m_tox, m_non = np.mean(X_tox), np.mean(X_non)
    history_tox.append(m_tox)
    history_non.append(m_non)

    
    if (day + 1) % 100 == 0:
        winner = "TOXIC" if m_tox > m_non else "NON-TOXIC"
        print(f"{day+1:<8} | {T_curr:<8.2f} | {m_tox:<15.4f} | {m_non:<15.4f} | {winner}")


plt.figure(figsize=(12, 6))
plt.stackplot(range(days_to_sim), history_tox, history_non, 
              labels=['Toxic Genotype', 'Non-Toxic Genotype'], 
              colors=['#d9534f', '#5cb85c'], alpha=0.8)

plt.title(f"Yamuna Competition Model: H={H}m | T_opt={T_opt_tox}/{T_opt_non}C", fontsize=14)
plt.xlabel("Days")
plt.ylabel("Depth-Averaged Biomass (mg/L)")
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()