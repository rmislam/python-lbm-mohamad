# Chapter 5: Diffusion
# 1-D diffusion equation with variable alpha with D1Q3

import numpy as np
import matplotlib.pyplot as plt

m = 101
w0 = 4 / 6
w1 = 1 / 6
c2 = 1 / 3
w2 = w1
dx = 1

rho = np.zeros(m)
f0 = np.zeros(m)
f1 = np.zeros(m)
f2 = np.zeros(m)
x = np.zeros(m)
fluxq = np.zeros(m)
flux = np.zeros(m)

tk = np.zeros(m)
dtkdx = np.zeros(m)
cpr = np.zeros(m)
alpha = np.zeros(m)
omega = np.zeros(m)

for i in range(m - 1):
    x[i + 1] = x[i] + dx

for i in range(m):
    tk[i] = 20 + 30 / (2 * x[i] + 1)
    alpha[i] = tk[i] / 100
    omega[i] = 1 / (3 * alpha[i] + 0.5)

twall = 1
nstep = 1500

f0 = w0 * rho
f1 = w1 * rho
f2 = w2 * rho

for _ in range(nstep):
    # Collision
    feq0 = w0 * rho
    feq = w1 * rho
    f0 = (1 - omega) * f0 + omega * feq0
    f1 = (1 - omega) * f1 + omega * feq
    f2 = (1 - omega) * f2 + omega * feq

    # Streaming
    f1 = np.roll(f1, 1)
    f2 = np.roll(f2, -1)

    # Boundary conditions
    f1[0] = twall - f0[0] - f2[0]
    f1[m - 1] = f1[m - 2]
    f2[m - 1] = f2[m - 2]
    f0[m - 1] = f0[m - 2]

    rho = f0 + f1 + f2

# Flux
flux = tk * omega * (f1 - f2) / c2
fluxq[:m - 1] = tk[:m - 1] * (rho[:m - 1] - rho[1:])
fluxq[m - 1] = fluxq[m - 2]

plt.figure(1)
plt.plot(x,rho)
plt.title('Temperature, 200 time steps')
plt.xlabel('X')
plt.ylabel('T')

plt.figure(2)
plt.plot(x, flux, 'o', x, fluxq, 'x')
plt.title('Flux, 200 time steps')
plt.xlabel('X')
plt.ylabel('Flux')

plt.show()