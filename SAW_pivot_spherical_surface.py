# -*- coding: utf-8 -*-
"""
# This is the python code to simulate a SAW chain travelling on the surface of a sphere using the pivot algorithm.
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from mpl_toolkits.mplot3d import Axes3D

# Simulation parameters
r = 1 # bead size
N_run = int(2e+5) # Total walk number in a run
N_step = 401  # Total number of beads in the chain 
n_gap=1  # gap for selecting walks to save and analyze
N_run2 = N_run//n_gap # the number of walks that are selected to be saved in a run
R = 10000   # confining spherical radius
theta = 2 * np.arcsin(r / (2 * R)) # constant angle corresponding to one step 

# creat 3D array for chain conformation ensemble obtained from random walks
Walk = np.zeros((N_step, 3, N_run))

"## generate the initial (walk) conformation"
# set the random number seed for repeatability of the initial conformation
repeat = 1
np.random.seed(repeat)

# generate first two steps (beads' position) 
Walk[0, 2, 0] = R
Walk[1, 0, 0] = R * np.sin(theta)
Walk[1, 2, 0] = R * np.cos(theta)

# random numbers for rotation angles
Phi = np.random.rand(N_step * int(1e5)) * 2 * np.pi

p = 0
s = 1
while s < N_step:
    A1 = Walk[s - 1, :, 0]
    phi1 = math.atan2(A1[1], A1[0])
    theta1 = math.acos(A1[2] / R)
    Ry_theta1 = np.array([[math.cos(theta1), 0, math.sin(theta1)],
                          [0, 1, 0],
                          [-math.sin(theta1), 0, math.cos(theta1)]])
    Rz_phi1 = np.array([[math.cos(phi1), -math.sin(phi1), 0],
                        [math.sin(phi1), math.cos(phi1), 0],
                        [0, 0, 1]])

    flag = 1
    while flag > 0:
        p = p + 1
        phi = Phi[p]
        # add a new random bead (step)
        Xd = R * np.array([math.sin(theta) * math.cos(phi),
                           math.sin(theta) * math.sin(phi),
                           math.cos(theta)])
        Xd2 = Xd.dot((Ry_theta1.T.dot(Rz_phi1.T)))
        # see if the new bead overlapps with the previous ones in space: yes, go ahead; no, return to last step
        if (((Walk[:s , :, 0] - Xd2) ** 2).sum(1)).min() >= r**2:
            flag = 0
            Walk[s, :, 0] = Xd2
            s = s + 1
        else:
            flag = flag + 1
            if flag > 5:
                flag = 0
                s -= 1
# Test if the initial conformation is self-avoiding 
Dis = 100
for i in range(N_step):
    for j in range(i + 1, N_step):
        d = np.sqrt(np.sum((Walk[i, :, 0] - Walk[j, :, 0]) ** 2))
        if d < Dis:
            Dis = d
if Dis >= 1:
    print('The initial conformation is self-avoiding!!!')

# Plot the initial conformation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Walk[:, 0, 0], Walk[:, 1, 0], Walk[:, 2, 0], '.r', markersize=15)
# Plot the sphere
theta_sphere, phi_sphere = np.mgrid[0.0:np.pi:1200j, 0.0:2.0 * np.pi:1200j]
x_sphere = R * np.sin(theta_sphere) * np.cos(phi_sphere)
y_sphere = R * np.sin(theta_sphere) * np.sin(phi_sphere)
z_sphere = R * np.cos(theta_sphere)
ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.5, color='b')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(True)
# ax.set_box_aspect([np.ptp(Walk[:, 0, 0]), np.ptp(Walk[:, 1, 0]), np.ptp(Walk[:, 2, 0])])
plt.axis([min(Walk[:,0,0]), max(Walk[:,0,0]), min(Walk[:,1,0]), max(Walk[:,1,0])])
ax.set_zlim([min(Walk[:,2,0]), max(Walk[:,2,0])])
plt.show()

"# Generate the random walks of the chain "
# Set random number seed for repeatability of the walks
np.random.seed(repeat)

# Random numbers for pivot beads and rotation angles
R1 = np.random.rand(N_run)
R2 = np.random.rand(N_run)
Phi = R2 * 2 * math.pi
del R2
# Time the "for" loop
start_time = time.time()
for i in range(1, N_run):
    s = int(N_step * R1[i]) + 1
    Walk[:, :, i] = Walk[:, :, i - 1]
    phi = Phi[i]

    if s == 1:
        continue
    elif s < N_step // 2:
        Xd = Walk[0:s - 1, :, i - 1]
        A1 = Walk[s-1, :, i - 1]
    else:
        Xd = Walk[s-1:, :, i - 1]
        A1 = Walk[s - 2, :, i - 1]
        
    phi1 = math.atan2(A1[1], A1[0])
    theta1 = math.acos(A1[2] / R)
       
    Ry_theta1 = np.array([[math.cos(theta1), 0, math.sin(theta1)],
                         [0, 1, 0],
                         [-math.sin(theta1), 0, math.cos(theta1)]])
    Rz_phi1 = np.array([[math.cos(phi1), -math.sin(phi1), 0],
                       [math.sin(phi1), math.cos(phi1), 0],
                       [0, 0, 1]])

    Rz_phi = np.array([[math.cos(phi), -math.sin(phi), 0],
                      [math.sin(phi), math.cos(phi), 0],
                      [0, 0, 1]])

    flag = 1
    if s == 1:
        continue
    elif s < N_step // 2:
        Walk[0:s - 1, :, i] = Xd.dot(Rz_phi1).dot(Ry_theta1).dot(Rz_phi).dot(Ry_theta1.T).dot(Rz_phi1.T)
       
        for j in range(s-1, N_step):
            if (((np.ones((s - 1, 1)) * Walk[j, :, i] - Walk[0:s - 1, :, i]) ** 2).sum(1)).min() < r ** 2:
                flag = 0
                break
        if flag == 0:
            Walk[0:s - 1, :, i] = Walk[0:s - 1, :, i - 1]
    else:
        Walk[s-1:, :, i] = Xd.dot(Rz_phi1).dot(Ry_theta1).dot(Rz_phi).dot(Ry_theta1.T).dot(Rz_phi1.T)
      
        for j in range(0, s-1):
            if (((np.ones((N_step - s + 1, 1)) * Walk[j, :, i] - Walk[s-1:, :, i]) ** 2).sum(1)).min() < r ** 2:
                flag = 0
                break
        if flag == 0:
            Walk[s-1:, :, i] = Walk[s-1:, :, i - 1]
end_time = time.time()    
# Show elapsed time
time_cost=end_time-start_time
print('Random walks have been produced!!!')
print("Elapsed Time:", time_cost)

# Radius of Gyration
Rg = np.zeros(N_run)
for i in range(N_run):
    walk = Walk[:, :, i * 1]
    X0 = np.mean(walk, axis=0)
    Rg[i] = np.sqrt(np.mean(np.sum((walk - np.ones((N_step, 1)) * X0) ** 2, axis=1), axis=0))

# Plotting Radius of Gyration
plt.figure()
plt.plot(range(1, N_run + 1), Rg, '.-')
plt.xlabel('Steps')
plt.ylabel('Radius of gyration (Rg)')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# Plot the final chain conformation 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Walk[:, 0, -1], Walk[:, 1, -1], Walk[:, 2, -1], marker='.', s=20)
# Plot the confining sphere
phi, theta = np.mgrid[0.0:2.0 * np.pi:2400j, 0.0:np.pi:1200j]
x_sphere = R * np.sin(theta) * np.cos(phi)
y_sphere = R * np.sin(theta) * np.sin(phi)
z_sphere = R * np.cos(theta)
ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.5, color='r', linewidth=0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(True)
plt.axis([min(Walk[:,0,-1]), max(Walk[:,0,-1]), min(Walk[:,1,-1]), max(Walk[:,1,-1])])
ax.set_zlim([min(Walk[:,2,-1]), max(Walk[:,2,-1])])
plt.show()

# Plot end position distribution
n = 30
n_bin = 2 * n + 1
End = Walk[-1, :, :] - Walk[0, :, :]
R_max = np.max(np.abs(End))
mk = ['o-', '^-', 's-']
plt.figure()
px_max=0
for i in range(3):
    X = End[i, :]
    bins = np.linspace(-R_max, R_max, 21)
    fx, x = np.histogram(X, bins)
    px = fx / np.sum(fx) / (bins[1] - bins[0])
    plt.plot(x[:-1], px, mk[i])
    px_max=max(px_max,max(px))
    # plt.hold(True)
plt.legend(['x', 'y', 'z'])
plt.ylim([0, px_max*1.05])
plt.xlabel('Coordinate')
plt.ylabel('PDF')
plt.show()

# Generate polar coordinates (theta, phi) of the end
Phi1 = []
Theta1 = []
for i in range(End.shape[1]):
    A1 = End[:, i]
    phi1 = math.atan2(A1[1], A1[0])
    theta1 = np.arccos(A1[2] / np.sqrt(np.sum(A1 ** 2)))
    Theta1.append(theta1)
    Phi1.append(phi1)
Phi1 = np.array(Phi1)
Theta1 = np.array(Theta1)
I = np.where(Phi1 < 0)
Phi1[I] += 2 * np.pi
# Plotting Theta and Phi distributions
px_max=0
fig, axs = plt.subplots(2,1,figsize=(5,7))
for i in range(2):
    if i == 0:
        X = Theta1
        bins = np.linspace(0, math.pi, 51)
        xstr = r'$\theta$'
    else:
        X = Phi1
        bins = np.linspace(0, 2 * math.pi, 51)
        xstr = r'$\phi$'
    fx, x = np.histogram(X, bins)
    px = fx / np.sum(fx) / (bins[1] - bins[0])
    axs[i].plot(x[:-1], px, mk[i])
    axs[i].set_xlabel(xstr)
    axs[i].set_ylabel('PDF')
    # px_max=max(px_max,max(px))
    # if i == 1:
    axs[i].set_ylim([0, max(px)*1.05])
    plt.show()