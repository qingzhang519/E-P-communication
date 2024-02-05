# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import math

np.random.seed(0)

r = 1
N_run = int(1e+4)
#N_step = 801
N_step = 401
N_run2=N_run//20;
Walk = np.zeros((N_step, 3, N_run2))
#Walk = np.zeros((N_step, 3, N_run))

repeat = 1
np.random.rand((repeat-1)*int(1.1e+8)+1)

Walk[0, :, 0] = 0
Phi = np.random.rand(N_step*int(1e+3)) * 2 * np.pi
Theta = np.arccos(1 - 2 * np.random.rand(N_step*int(1e+3)))

p = 0
s = 1
while s < N_step:
    flag = 1
    while flag > 0:
        p += 1
        phi = Phi[p]
        theta = Theta[p]
        Xd = r * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        Xd2 = Xd + Walk[s-1, :, 0]
        if np.min(np.sum((Walk[0:s, :, 0] - np.ones((s, 1)) * Xd2)**2, axis=1)) >= r**2:
            flag = 0
            Walk[s, :, 0] = Xd2
            s += 1
        else:
            flag += 1
            if flag > 5:
                flag = 0
                s -= 1

Dis = 100
for i in range(N_step):
    for j in range(i+1, N_step):
        d = np.sqrt(np.sum((Walk[i, :, 0] - Walk[j, :, 0])**2))
        if d < Dis:
            Dis = d

if Dis >= 1:
    print('The initial conformation is self-avoiding!!!')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Walk[:, 0, 0], Walk[:, 1, 0], Walk[:, 2, 0], '.-', markersize=20)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(True)
plt.axis('equal')
plt.show()

np.random.seed(repeat)
R1 = np.random.rand(N_run)
R2 = np.random.rand(N_run)
Phi = R2 * 2 * np.pi
R3 = np.random.rand(N_run)
Theta = np.arccos(1 - 2 * R3)
del R2,R3
walk_last = Walk[:, :, 0]
walk_ongoing = walk_last.copy()
start_time = time.time()

n_select=N_run // N_run2
for i in range(1, N_run):
   # s = int(N_step * np.fix(N_run * R1[i - 1])) + 1
    s = int(N_step * R1[i - 1])+1 

    if i % n_select == 0:
        Walk[:, :, int((i+1) // n_select)] = walk_ongoing.copy()

    phi = Phi[i - 1]
    theta = Theta[i - 1]

    if s == 1:
        continue
    
    # Xd = np.zeros((N_step, 3))
    if s < N_step // 2:
        Xd = walk_last[0:s-1, :] - np.ones((s-1, 1)) * walk_last[s-1, :]
    else:
        Xd = walk_last[s-1:, :] - np.ones((N_step - s + 1, 1)) * walk_last[s - 2, :]
    
    # if s < N_step // 2:
    #     Xd[0:s-1, :] = walk_last[0:s-1, :] - np.ones((s-1, 1)) * walk_last[s-1, :]
    # else:
    #     Xd[s-1:, :] = walk_last[s-1:, :] - np.ones((N_step - s + 1, 1)) * walk_last[s - 2, :] 


    Ry_theta = np.array([[math.cos(theta), 0, math.sin(theta)],
                         [0, 1, 0],
                         [-math.sin(theta), 0, math.cos(theta)]])
    Rz_phi = np.array([[math.cos(phi), -math.sin(phi), 0],
                       [math.sin(phi), math.cos(phi), 0],
                       [0, 0, 1]])

    flag = 1
    # if s == 1:
    #     # Walk[s-1:, :, i - 1] = np.dot(Xd, np.dot(Rz_phi.T, Ry_theta.T))
    #     1
    # transformed_Xd = np.dot(Xd, np.dot(Rz_phi.T, Ry_theta.T))
    # transformed_Xd = Xd.dot(Rz_phi.T.dot(Ry_theta.T))
    if s < N_step // 2:
        walk_ongoing[0:s-1, :] = np.ones((s-1, 1)) * walk_last[s-1, :] + np.dot(Xd, np.dot(Rz_phi.T, Ry_theta.T))
        # walk_ongoing[0:s-1, :] = np.ones((s-1, 1)) * walk_last[s-1, :] + transformed_Xd#[0:s-1, :]

        for j in range(s-1, N_step):
            if ((np.ones((s-1, 1)) * walk_ongoing[j, :] - walk_ongoing[0:s-1, :]) ** 2).sum(1).min() < r ** 2:
                flag = 0
                break

        if flag == 0:
            walk_ongoing[0:s-1, :] = walk_last[0:s-1, :]
    else:
        walk_ongoing[s-1:, :] = np.ones((N_step - s + 1, 1)) * walk_last[s - 2, :] + np.dot(Xd, np.dot(Rz_phi.T, Ry_theta.T))
        # walk_ongoing[s-1:, :] = np.ones((N_step - s + 1, 1)) * walk_last[s - 2, :] + transformed_Xd#[s-1:, :]
        
        for j in range(0, s-1):
            if ((np.ones((N_step - s + 1, 1)) * walk_ongoing[j, :] - walk_ongoing[s-1:, :]) ** 2).sum(1).min() < r ** 2:
                flag = 0
                break

        if flag == 0:
            walk_ongoing[s-1:, :] = walk_last[s-1:, :]
    walk_last = walk_ongoing.copy()
end_time = time.time()    
time_cost=end_time-start_time
print('time_cost (min)=')
print(time_cost/60)
print('last time cost=6.817 min')

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

# Plot 3D walk
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Walk[:, 0, -1], Walk[:, 1, -1], Walk[:, 2, -1], '.-', markersize=20)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(True)
ax.set_box_aspect([np.ptp(coord) for coord in [Walk[:, 0, -1], Walk[:, 1, -1], Walk[:, 2, -1]]])
plt.show()

mk = ['o-', '^-', 's-']

# end position distribution
n = 30
n_bin = 2 * n + 1

End = Walk[-1, :, :] - Walk[0, :, :]
#End2 = End#np.reshape(End, (End.shape[0], End.shape[1]))
R_max = np.max(np.abs(End))
fig, axs = plt.subplots()
for i in range(3):
    X = End[i, :]
    bins = np.linspace(-R_max, R_max, 21)
    fx, x = np.histogram(X, bins)
    px = fx / sum(fx) / (bins[1] - bins[0])
    axs.plot(x[:-1], px, mk[i])

axs.legend(['x', 'y', 'z'])
axs.set_ylim([0, np.max(px) + 0.1])
plt.show()

# Compute Phi1 and Theta1
Phi1 = []
Theta1 = []
for i in range(End.shape[1]):
    A1 = End[:, i]
    # phi1 = np.arctan2(A1[1], A1[0])
    phi1 = math.atan2(A1[1], A1[0])
    # phi1 = np.arctan(A1[1]/A1[0])
    # if A1[0] == 0:
    #     if A1[1] >= 0:
    #         phi1 = 0
    #     else:
    #         phi1 = math.pi
    # theta1 = np.arccos(A1[2] / np.sqrt(np.sum(A1**2)))
    theta1 = math.acos(A1[2] / math.sqrt(sum(A1**2)))
    # if A1[0] / np.sqrt(np.sum(A1**2)) / np.sin(theta1) < 0:
    #     phi1 = phi1 + np.pi
    Theta1.append(theta1)
    Phi1.append(phi1)

I = np.where(np.array(Phi1) < 0)[0]
for idx in I:
    Phi1[idx] += 2 * np.pi

# Plot Phi1 and Theta1
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
    fig, axs = plt.subplots()
    axs.plot(x[:-1], px, mk[i])
    axs.set_xlabel(xstr)
    axs.set_ylabel('PDF')
    if i == 1:
        axs.set_ylim([0, 0.3])
    plt.show()