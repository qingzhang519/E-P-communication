# -*- coding: utf-8 -*-
"""
# This is the Python code for simulating a SAW chain partially confined inside a sphere using the pivot algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from numpy import cos, sin, pi

# Simulation parameters
r = 1  # bead size
R = 20  # spherical radius
N_step = 401  #  Total number of beads in the chain 
r0 = 0  # translational motion 0  1 10
N_run = int(2e4)  # Total walk number in a run
n_part=40;   # [1 4 10 20 40 100 400] 
n_step = (N_step - 1) // n_part + 1
I_ss = [i * (n_step - 1) for i in range(n_part + 1)]

# chain conformation ensemble obtained from random walks
Walk = np.zeros((N_step, 3, N_run))

## generate the initial (walk) conformation
# set the random number seed for repeatability of the initial conformation
repeat = 0
np.random.seed(repeat)
# %% to randomly form initial configuration
print('Initializing........')

for h in range(1, 101):
    print(h)
    Walk[0, :, 0] = 0
    r2 = np.random.rand(N_step * 1000)
    Phi = r2 * 2 * pi
    r3 = np.random.rand(N_step * 1000)
    Theta = np.arccos(1 - 2 * r3)
    
    p = 0
    s = 2
    while s <= N_step:
        flag = 1
        while flag > 0:
            p += 1
            phi = Phi[p-1]  # Python is 0-indexed
            theta = Theta[p-1]
            
            Xd = r * np.array([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])
            Xd2 = Xd + Walk[s-2, :, 0] 
            
            # Check for overlap
            if np.min(np.sum((Walk[:s-1, :, 0] - np.ones((s-1, 1)) * Xd2)**2, axis=1)) >= r**2:
                flag = 0
                Walk[s-1, :, 0] = Xd2  # Python is 0-indexed
                s += 1
            else:
                flag += 1
                if flag > 5:
                    flag = 0
                    s -= 1
    
    # Move the center to the origin
    Walk[:, :, 0] = Walk[:, :, 0] - np.ones((N_step, 1)) * np.mean(Walk[:, :, 0], axis=0)

    if np.max(np.sum(Walk[I_ss, :, 0]**2, axis=1)) < R**2:
        break


Dis=100    
# compute the minimal distance
for i in range(N_step):
    for j in range(i + 1, N_step):
        d = np.sqrt(np.sum((Walk[i, :, 0] - Walk[j, :, 0]) ** 2))
        if d < Dis:
            Dis = d

# check if self-avoiding
if Dis >= 1 - 1e-9:
    print("The initial conformation is self-avoiding!!!")

# plot the intial conformation of the chain
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Walk[:, 0, 0], Walk[:, 1, 0], Walk[:, 2, 0], '.-', markersize=20)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(True)
ax.set_box_aspect([1, 1, 1])  
plt.show()

# Generate random angles and step indices
Phi = np.random.rand(N_run) * 2 * pi
Theta = np.arccos(1 - 2 * np.random.rand(N_run))
S = np.floor(N_step * np.random.rand(N_run)).astype(int) + 1

start_time = time.time()
for i in range(1, N_run):
    s = S[i]
    Walk[:, :, i] = Walk[:, :, i-1]  # copy previous walk
    
    phi = Phi[i]
    theta = Theta[i]
    
    if s == 1:
        # Translation case
        Xd = Walk[:, :, i-1] + r0 * np.ones((N_step, 1)) * np.array([sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)])
        
        if np.max(np.sum(Xd[I_ss,:]**2, axis=1)) <= R**2:  ###
            Walk[:, :, i] = Xd
        continue
    
    elif s < int(N_step/2):
        Xd = Walk[:s-1, :, i-1] - np.ones((s-1, 1)) * Walk[s-1, :, i-1]  # Python is 0-indexed
    else:
        Xd = Walk[s-1:, :, i-1] - np.ones((N_step - s + 1, 1)) * Walk[s-2, :, i-1]  # Python is 0-indexed
    
    # Rotation matrices
    Ry_theta = np.array([
        [cos(theta), 0, sin(theta)],
        [0, 1, 0],
        [-sin(theta), 0, cos(theta)]
    ])
    
    Rz_phi = np.array([
        [cos(phi), -sin(phi), 0],
        [sin(phi), cos(phi), 0],
        [0, 0, 1]
    ])
    
    flag = 1
    
    if s < int(N_step/2):
        # Apply rotation to first part of walk
        Walk[:s-1, :, i] = np.ones((s-1, 1)) * Walk[s-1, :, i-1] + Xd @ Rz_phi.T @ Ry_theta.T
        
        # Check for overlaps
        for j in range(s-1, N_step):
            if np.min(np.sum((np.ones((s-1, 1)) * Walk[j, :, i] - Walk[:s-1, :, i])**2, axis=1)) < r**2:
                flag = 0
                break
        
        I_intersect = list(set(I_ss).intersection(range(0, s-1))) ####
        if flag == 0 or np.max(np.sum(Walk[I_intersect, :, i]**2, axis=1)) > R**2:
            Walk[:s-1, :, i] = Walk[:s-1, :, i-1]
    
    else:
        # Apply rotation to latter part of walk
        Walk[s-1:, :, i] = np.ones((N_step - s + 1, 1)) * Walk[s-2, :, i-1] + Xd @ Rz_phi.T @ Ry_theta.T
        
        # Check for overlaps
        for j in range(s-1):
            if np.min(np.sum((np.ones((N_step - s + 1, 1)) * Walk[j, :, i] - Walk[s-1:, :, i])**2, axis=1)) < r**2:
                flag = 0
                break
        
        I_intersect = list(set(I_ss).intersection(range(s-1, N_step))) 
        if flag == 0 or np.max(np.sum(Walk[I_intersect, :, i]**2, axis=1)) > R**2:
            Walk[s-1:, :, i] = Walk[s-1:, :, i-1]

end_time = time.time()    
# Show elapsed time
time_cost=end_time-start_time
print("The run is over!")
print("Elapsed Time:", time_cost)

# Compute radius of gyration (Rg)
N_run2=N_run
Rg = np.zeros(N_run2)
for i in range(N_run2):
    walk = Walk[:, :, i]
    X0 = walk.mean(0)
    Rg[i] = math.sqrt(((walk - np.ones((N_step, 1)) * X0) ** 2).sum(1).mean())

# Plot Radius of gyration (Rg)
plt.figure()
plt.plot(range(1, N_run2 + 1), Rg, '.-')
plt.xlabel('Steps')
plt.ylabel('Radius of gyration (Rg)')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# Plot the last chain (walk) conformation 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Walk[:, 0, -1], Walk[:, 1, -1], Walk[:, 2, -1], '.-', markersize=20)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(True)
ax.set_box_aspect([np.ptp(coord) for coord in [Walk[:, 0, -1], Walk[:, 1, -1], Walk[:, 2, -1]]])
plt.show()

# Plot end position distribution
n = 30
n_bin = 2 * n + 1
End = Walk[-1, :, :] - Walk[0, :, :]
R_max = np.max(np.abs(End))
fig, axs = plt.subplots(figsize=(5,4))
mk = ['o-', '^-', 's-']
for i in range(3):
    X = End[i, :]
    bins = np.linspace(-R_max, R_max, 21)
    fx, x = np.histogram(X, bins)
    px = fx / sum(fx) / (bins[1] - bins[0])
    axs.plot(x[:-1], px, mk[i])
axs.legend(['x', 'y', 'z'])
axs.set_ylim([0, np.max(px) + 0.1])
plt.show()

# Generate polar coordinates (theta, phi) of the end
Phi1 = []
Theta1 = []
for i in range(End.shape[1]):
    A1 = End[:, i]
    phi1 = math.atan2(A1[1], A1[0])
    theta1 = math.acos(A1[2] / math.sqrt(sum(A1**2)))
    Theta1.append(theta1)
    Phi1.append(phi1)
I = np.where(np.array(Phi1) < 0)[0]
for idx in I:
    Phi1[idx] += 2 * np.pi

# Plot polar coordinates (theta, phi) distribution
fig, axs = plt.subplots(2,1,figsize=(5,7))
for i in range(2):
    if i == 0:
        X = Theta1
        bins = np.linspace(0, math.pi, 31)
        xstr = r'$\theta$'
    else:
        X = Phi1
        bins = np.linspace(0, 2 * math.pi, 101)
        xstr = r'$\phi$'
    fx, x = np.histogram(X, bins)
    px = fx / sum(fx) / (bins[1] - bins[0])
    axs[i].plot(x[:-1], px, mk[i])
    axs[i].set_xlabel(xstr)
    axs[i].set_ylabel('PDF')
    if i == 1:
        axs[i].set_ylim([0, 0.3])
    plt.show()
###    