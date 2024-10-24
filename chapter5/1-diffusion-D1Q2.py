# Chapter 5: Diffusion
# 1-D diffusion equation with D1Q2

import numpy as np
import matplotlib.pyplot as plt

m = 101              # number of grid points
dx = 1.0             # spatial step
rho = np.zeros(m)    # nondimensionalized temperature (all zeros at initial time)
f1 = np.zeros(m)     # right distribution function
f2 = np.zeros(m)     # left distribution function
flux = np.zeros(m)
x = np.zeros(m)      # grid points

# Create x grid
for i in range(m - 1):
    x[i + 1] = x[i] + dx

alpha = 0.25  # diffusivity
omega = 1 / (alpha + 0.5)
twall = 1.0  # (nondimensionalized) temperature of hot wall
nstep = 200

# Initialize f distribution
for i in range(m):
    f1[i] = 0.5 * rho[i]
    f2[i] = 0.5 * rho[i]

# Main simulation loop
for _ in range(nstep):
    # Collision
    feq = 0.5 * rho
    f1 = (1 - omega) * f1 + omega * feq
    f2 = (1 - omega) * f2 + omega * feq

    # Streaming
    f1 = np.roll(f1, 1)
    f2 = np.roll(f2, -1)

    # Boundary conditions
    f1[0] = twall - f2[0]
    f2[m - 1] = f2[m - 2]

    rho = f1 + f2

# Flux
flux = omega * (f1 - f2)

# Plots
plt.figure(1)
plt.plot(x, rho)
plt.title('Temperature, 200 time steps')
plt.xlabel('X')
plt.ylabel('T')

plt.figure(2)
plt.plot(x, flux, 'o')
plt.title('Flux, 200 time steps')
plt.xlabel('X')
plt.ylabel('Flux')

plt.show()