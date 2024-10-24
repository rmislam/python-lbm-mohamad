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
w0 = 2 / 6
w = 1 / 6
c2 = 1 / 3
dx = 1

rho = np.zeros((m, n))
f0 = np.zeros((m, n))
f1 = np.zeros((m, n))
f2 = np.zeros((m, n))
f3 = np.zeros((m, n))
f4 = np.zeros((m, n))
x = np.zeros(m)
y = np.zeros(n)
Tm = np.zeros(m)

alpha = 0.25
omega = 1 / (3 * alpha + 0.5)
twall = 1
nstep = 400

for i in range(m - 1):
    x[i + 1] = x[i] + dx

for j in range(n - 1):
    y[j + 1] = y[j] + dy

f0 = w0 * rho
f1 = f2 = f3 = f4 = w * rho

for _ in range(nstep):
    # Collision
    feq0 = w0 * rho
    feq = w * rho
    f0 = (1 - omega) * f0 + omega * feq0
    f1 = (1 - omega) * f1 + omega * feq
    f2 = (1 - omega) * f2 + omega * feq
    f3 = (1 - omega) * f3 + omega * feq
    f4 = (1 - omega) * f4 + omega * feq

    # Streaming
    f1 = np.roll(f1, 1, axis=0)
    f2 = np.roll(f2, -1, axis=0)
    f3 = np.roll(f3, 1, axis=1)
    f4 = np.roll(f4, -1, axis=1)

    # Boundary conditions
    f1[0, :] = twall - f0[0, :] - f2[0, :] - f3[0, :] - f4[0, :]
    f2[m - 1, :] = -f0[m - 1, :] - f1[m - 1, :] - f3[m - 1, :] - f4[m - 1, :]
 
    f3[:, 0] = f3[:, 1]
    f4[:, n - 1] = -f0[:, n - 1] - f1[:, n - 1] - f2[:, n - 1] - f3[:, n - 1]

    rho = f0 + f1 + f2 + f3 + f4

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