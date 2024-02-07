# -*- coding: utf-8 -*-
# This is the python code for the pivot simulation of the 2D SAW model

import numpy as np
import matplotlib.pyplot as plt
import time
import math

# Simulation parameters
r = 1 # bead size
N_run = int(2e+5) # Total walk number in a run
N_step = 401  # Total number of beads in the chain 
n_gap=10  # gap for selecting walks to save and analyze
N_run2 = N_run//n_gap # the number of walks that are selected to be saved in a run

# chain conformation ensemble obtained from random walks
Walk = np.zeros((N_step, 3, N_run2))  

## generate the initial (walk) conformation
# set the random number seed for repeatability of the initial conformation
repeat = 1
np.random.seed(repeat)
# Produce some random conformations and choose one as the initial conformation 
n_c=50 # the number of random conformations
Rg_0 = np.zeros(n_c)  # radius of gyration (Rg)
EED_0 = np.zeros(n_c) # average end-to-end distance (EED)
Walk2 = [None] * n_c
for h in range(n_c):
    print(h)
    walk0 = np.zeros((N_step, 3))
    Phi = np.random.rand(N_step * int(1e+4)) * 2 * math.pi

    p = 0
    s = 1
    Flag = np.zeros(N_step)
    while s < N_step:
        flag = 1
        Flag[s] += 1
        
        # go back to an earlier step if the walk is trapped for a period
        if Flag[s] > 100:  
            Flag[s:] = 0
            s = max(s - 50, 1)

        while flag > 0:
            # add a new random bead (step)
            p += 1
            phi = Phi[p]
            Xd = r * np.array([math.cos(phi), math.sin(phi), 0])
            Xd2 = Xd + walk0[s - 1, :]
            
            # see if the new bead overlapps with the previous ones in space: yes, go ahead; no, return to last step
            if (((walk0[:s, :] - np.ones((s, 1)) * Xd2) ** 2).sum(1)).min() >= r**2:
                flag = 0
                walk0[s, :] = Xd2
                s += 1
            else:
                flag += 1
                if flag > 10:
                    flag = 0
                    s -= 1
    Walk2[h] = walk0
    X0 = np.mean(walk0, axis=0)
    Rg_0[h] = np.sqrt(np.mean(np.sum((walk0 - np.ones((N_step, 1)) * X0) ** 2, axis=1)))
    EED_0[h] = np.sqrt(np.sum((walk0[-1, :] - walk0[0, :]) ** 2))

# Find the index with the minimum distance from the mean
dd = np.sqrt((Rg_0 - np.mean(Rg_0))**2 + (EED_0 - np.mean(EED_0))**2)
I = np.argmin(dd)

# Plot EED vs. Rg
plt.figure()
plt.plot(Rg_0, EED_0, '*', label='Data Points')
plt.plot(np.mean(Rg_0), np.mean(EED_0), 'og', label='Mean Point')
plt.plot(Rg_0[I], EED_0[I], '*r', label='Min Distance Point')
plt.xlabel('Rg')
plt.ylabel('EED')
plt.legend()
plt.show()

# Select the initial conformation
Walk[:, :, 0] = Walk2[I]

# Clear variables to free up space
del Rg_0, EED_0, Walk2

# Test if the initial conformation is self-avoiding 
Dis = 1e+8
for i in range(N_step):
    for j in range(i + 1, N_step):
        d = np.sqrt(np.sum((Walk[i, :, 0] - Walk[j, :, 0]) ** 2))
        if d < Dis:
            Dis = d
if Dis >= 1:
    print('The initial conformation is self-avoiding!')
# Plot the initial conformation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(Walk[:, 0, 0], Walk[:, 1, 0], Walk[:, 2, 0], '.-', markersize=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(True)
ax.view_init(elev=90, azim=0)
plt.show()

# Generate the random walks of the chain 
# Random number seed for repeatability of the walks
np.random.seed(repeat)
# Random numbers for pivot beads and rotation angles
R1 = np.random.rand(1, N_run)
Phi = np.random.rand(1, N_run) * 2 * math.pi

walk_last = Walk[:, :, 0]
walk_ongoing = np.copy(walk_last)
# Time the "for" loop
start_time = time.time()
for i in range(1, N_run):
    s = int(N_step * R1[0, i]) + 1

    if i % n_gap == 0:
        Walk[:, :,(i+1) // n_gap] = np.copy(walk_ongoing)

    phi = Phi[0, i]

    if s == 1:
        continue
    elif s < N_step // 2:
        Xd = walk_last[0:s - 1, :] - np.ones((s - 1, 1)) * walk_last[s-1, :]
    else:
        Xd = walk_last[s - 1:, :] - np.ones((N_step - s + 1, 1)) * walk_last[s - 2, :]

    R = np.sqrt((Xd ** 2).sum())

    Rz_phi = np.array([[math.cos(phi), -math.sin(phi), 0],
                       [math.sin(phi), math.cos(phi), 0],
                       [0, 0, 1]])

    flag = 1
    if s == 1:
        pass  
    elif s < N_step // 2:
        walk_ongoing[0:s - 1, :] = np.ones((s - 1, 1)) * walk_last[s-1, :] + Xd.dot(Rz_phi.T)#np.dot(Xd, Rz_phi.T)
        
        for j in range(s-1, N_step):
            if (((np.ones((s - 1, 1)) * walk_ongoing[j, :] - walk_ongoing[0:s - 1, :]) ** 2).sum(1)).min() < r ** 2:
                flag = 0
                break
        if flag == 0:
            walk_ongoing[0:s - 1, :] = np.copy(walk_last[0:s - 1, :])
    else:
        walk_ongoing[s - 1:, :] = np.ones((N_step - s + 1, 1)) * walk_last[s - 2, :] + Xd.dot(Rz_phi.T)#np.dot(Xd, Rz_phi.T)
        for j in range(0, s - 1):
            if (((np.ones((N_step - s + 1, 1)) * walk_ongoing[j, :] - walk_ongoing[s - 1:, :]) ** 2).sum(1)).min() < r ** 2:
                flag = 0
                break
        if flag == 0:
            walk_ongoing[s - 1:, :] = np.copy(walk_last[s - 1:, :])
    walk_last = np.copy(walk_ongoing)

# Print elapsed time
end_time = time.time()
print("The run is over!")
print("Elapsed Time:", end_time - start_time)

# Compute radius of gyration (Rg)
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

# Plot the last walk conformation 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Walk[:, 0, -1], Walk[:, 1, -1], Walk[:, 2, -1], '.-', markersize=20)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(True)
# ax.set_box_aspect([np.ptp(coord) for coord in [Walk[:, 0, -1], Walk[:, 1, -1], Walk[:, 2, -1]]])
ax.view_init(elev=90, azim=0)
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
axs.set_xlabel('Coordinate')
axs.set_ylabel('PDF')
axs.set_ylim([0, 0.07])
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