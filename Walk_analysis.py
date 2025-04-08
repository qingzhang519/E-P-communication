# -*- coding: utf-8 -*-
# Analyze the SAW trajectory of the polymer chain and save data such as contour length and average end-to-end distance (EED). 
# The figures included are for testing purposes.
#
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd


CASE=3

if CASE==0:
    from SAW_pivot_3D import Walk,repeat
elif CASE==1:
    from SAW_pivot_2D import Walk,repeat
elif CASE==2:
    from SAW_pivot_spherical_surface import Walk, R, repeat
else:
    from SAW_pivot_spherical_interior import Walk, R, repeat


# Get dimensions from Walk array
N_step = Walk.shape[0]
N_run = Walk.shape[2]

print('The change in the end-to-end distance with the time')
start_time = time.time()
t = np.arange(1, N_run//20 + 1)
dis_t = np.sqrt(np.sum((Walk[-1, :, t-1] - Walk[0, :, t-1])**2, axis=1))
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

Dis_t = dis_t.flatten()
plt.figure()
plt.plot(t, Dis_t, '-')
plt.xlabel('Time (*20 steps)', fontsize=15)
plt.ylabel('End-to-end distance', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()

print('Radius of gyration')
start_time = time.time()
Rg = np.zeros(len(t))
for i in range(len(t)):
    walk = Walk[:, :, t[i]-1]
    X0 = np.mean(walk, axis=0)
    Rg[i] = np.sqrt(np.mean(np.sum((walk - np.ones((N_step, 1)) * X0)**2, axis=1)))
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

plt.figure()
plt.plot(t, Rg, '-')
plt.xlabel('Time', fontsize=15)
plt.ylabel('Gyration radius', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()

# 3D plot of walks
for j in [0, N_run-1]:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Walk[:, 0, j], Walk[:, 1, j], Walk[:, 2, j], s=15, marker='.')
            
    # Create sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones(30), np.cos(v))
    
    ax.plot_surface(x, y, z, alpha=0.5, shade=True)
    ax.set_aspect('equal')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

print('End-to-end distance (internal)')
start_time = time.time()
b = 50 if N_step > 50 else 0
CL = np.zeros(N_step - 2*b - 1)
L_3d = np.zeros(N_step - 2*b - 1)
n2 = Walk.shape[1]
n3 = Walk.shape[2]
m = min(50000,n3)
for h in range(N_step - 2*b - 1):
    End2 = np.zeros((n2, m * (N_step - h - 2*b)))
    for i in range(b, N_step - h - b):
        start_idx = (i - b) * m
        end_idx = (i - b + 1) * m
        End2[:, start_idx:end_idx] = Walk[h+i, :, -m:] - Walk[i, :, -m:]
    
    R_end = np.sqrt(np.sum(End2**2, axis=0))
    L_3d[h] = np.mean(R_end)
    CL[h] = h + 1 
# Create the log-log plot of average EED vs. contour length
plt.figure(figsize=(8, 6))
plt.loglog(CL, L_3d, 'o', markersize=10)
I = np.where((CL > 5) & (L_3d>0))[0]
coef = np.polyfit(np.log(CL[I]), np.log(L_3d[I]), 1)
X = np.arange(1, np.max(CL)+1)
Y = np.exp(coef[0] * np.log(X) + coef[1])
plt.loglog(X, Y, '-r', linewidth=2)
# Add labels and legend
plt.xlabel('contour length', fontsize=15)
plt.ylabel('3d space distance', fontsize=15)
plt.legend(['simulation', f'd∝L$^{{{coef[0]:.3f}}}$'], fontsize=15)
# Set tick font size
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
plt.show()
plt.xlim([1,300])
#plt.ylim([1,300])
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

# MSD vs time
print('MSD')
start_time = time.time()
End = Walk[-1, :, :] - Walk[0, :, :]
End2 = End.reshape(End.shape[0], End.shape[1])

if N_run >= 2e5:
    End2 = End2[:, 1000:]  # Remove early 1000 points

T = np.arange(0, 501, 5)
MSD = np.zeros(len(T))

for s in range(len(T)):
    t_val = T[s]
    End3 = End2[:, t_val:] - End2[:, :-t_val if t_val != 0 else None]
    MSD[s] = np.sum(np.sum(End3**2, axis=0)) / (End2.shape[1] - t_val)

print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

# Save to Excel
output_filename = 'SAW_pivot_spherical_interior_R='+ str(R)+'_'+str(repeat)+'.xlsx'
with pd.ExcelWriter(output_filename) as writer:
    pd.DataFrame({'CL': CL, 'L_3d': L_3d}).to_excel(writer, sheet_name='CL_EEDistance', index=False)
    pd.DataFrame({'T': T, 'MSD': MSD}).to_excel(writer, sheet_name='T_MSD', index=False)
    pd.DataFrame({'Dis_t': Dis_t, 'Rg': Rg}).to_excel(writer, sheet_name='T_EEDis_Rg', index=False)

# Clear Walk array if needed
# del Walk    