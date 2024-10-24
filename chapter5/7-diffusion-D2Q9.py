# Chapter 5: Diffusion
# 2-D diffusion equation with D2Q5a

import numpy as np
import matplotlib.pyplot as plt

m = 51
n = 51
x1 = 1
y1 = 1
dx = x1 / (m - 1)
dy = y1 / (n - 1)
c2 = 1 / 3
dx = 1

alpha = 0.25
omega = 1 / (3 * alpha + 0.5)
twall = 1
nstep = 400

w = np.zeros(9)
w[0] = 4 / 9

for i in range(1, 5):
    w[i] = 1 / 9

for i in range(5, 9):
    w[i] = 1 / 36

directions = np.zeros((9, 2), dtype=int)
directions[1] = [1, 0]
directions[2] = [0, 1]
directions[3] = [-1, 0]
directions[4] = [0, -1]
directions[5] = [1, 1]
directions[6] = [-1, 1]
directions[7] = [-1, -1]
directions[8] = [1, -1]

rho = np.zeros((m, n))
f = np.zeros((m, n, 9))
feq = np.zeros((m, n, 9))
x = np.zeros(m)
y = np.zeros(n)
Tm = np.zeros(m)

for i in range(m - 1):
    x[i + 1] = x[i] + dx

for j in range(n - 1):
    y[j + 1] = y[j] + dy

for _ in range(nstep):
    # Collision
    feq = np.tensordot(rho, w, axes=0)
    f = (1 - omega) * f + omega * feq

    # Streaming
    for i, direction in enumerate(directions):
        f[:, :, i] = np.roll(f[:, :, i], direction, axis=(0, 1))

    # Boundary conditions
    # Left boundary, fixed at hot temperature
    f[0, :, 1] = (w[1] + w[3]) * twall - f[0, :, 3]
    f[0, :, 5] = (w[5] + w[7]) * twall - f[0, :, 7]
    f[0, :, 8] = (w[8] + w[6]) * twall - f[0, :, 6]

    # Bottom boundary, adiabatic bounce-back
    f[:, 0, 2] = f[:, 1, 2]
    f[:, 0, 5] = f[:, 1, 5]
    f[:, 0, 6] = f[:, 1, 6]

    # Top boundary, fixed at cold temperature
    f[:, -1, 7] = -f[:, -1, 5]
    f[:, -1, 4] = -f[:, -1, 2]
    f[:, -1, 8] = -f[:, -1, 6]

    # Right boundary, fixed at cold temperature
    f[-1, :, 3] = -f[-1, :, 1]
    f[-1, :, 7] = -f[-1, :, 5]
    f[-1, :, 6] = -f[-1, :, 8]

    rho = np.sum(f, axis=2)

Tm = rho[:, int((n - 1) / 2)]

plt.figure(1)
plt.plot(x, Tm)
plt.title('Temperature, 400 time steps')
plt.xlabel('X')
plt.ylabel('T')

plt.figure(2)
plt.title('Contour plot of temperature')
plt.contourf(rho.transpose())

plt.show()