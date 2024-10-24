# Chapter 5: Diffusion
# 1-D diffusion equation with source term with D1Q3

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

for i in range(m - 1):
    x[i + 1] = x[i] + dx

rcp = 200
qs = 1
qsr = qs / rcp
alpha = 0.25
omega = 1 / (3 * alpha + 0.5)
twall = 1
tk = alpha * rcp
nstep = 200

f0 = w0 * rho
f1 = w1 * rho
f2 = w2 * rho

for _ in range(nstep):
    # Collision
    feq0 = w0 * rho
    feq = w1 * rho
    f0 = (1 - omega) * f0 + omega * feq0 + w0 * qsr
    f1 = (1 - omega) * f1 + omega * feq + w1 * qsr
    f2 = (1 - omega) * f2 + omega * feq + w2 * qsr

    # Streaming
    f1 = np.roll(f1, 1)
    f2 = np.roll(f2, -1)

    # Boundary conditions
    f1[0] = twall - f0[0] - f2[0]
    f2[m - 1] = f2[m - 2]
    f0[m - 1] = f0[m - 2]

    rho = f0 + f1 + f2

# Flux
flux = omega * (f1 - f2) / c2
fluxq[:m - 1] = rho[:m - 1] - rho[1:]
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