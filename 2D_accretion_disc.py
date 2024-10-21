import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # m^3 kg^-1 s^-2, gravitational constant
c = 299792458    # m/s, speed of light
sigma = 5.670374419e-8  # W m^-2 K^-4, Stefan-Boltzmann constant
M_sun = 1.98847e30  # kg, mass of the sun

# Black hole parameters
M_BH = 1e8 * M_sun  # kg, mass of the black hole (e.g., 10^8 solar masses)
dot_M = 1 * M_sun / (365.25 * 24 * 3600)  # kg/s, mass accretion rate (e.g., 1 solar mass per year)

# Radial grid parameters
N_R = 500  # Number of radial grid points
R_out_factor = 10  # Outer radius factor (R_out = R_out_factor * R_ISCO)

# Angular grid parameters
N_theta = 500  # Number of angular grid points

# Calculate inner and outer radii
R_ISCO = 6 * G * M_BH / c**2  # ISCO radius for a Schwarzschild black hole
R_in = R_ISCO
R_out = R_out_factor * R_ISCO

# Radial and angular grids
radii = np.linspace(R_in, R_out, N_R)  # Radial positions
theta_coords = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)  # Angular positions

# Meshgrid for polar coordinates
R_grid, Theta_grid = np.meshgrid(radii, theta_coords, indexing='ij')

# Base temperature profile functions
def T_N4(R):
    return (3 * G * M_BH * dot_M) / (8 * np.pi * R**3 * sigma)

def T_base(R):
    relativistic_correction = np.sqrt(np.maximum(1 - (3 * G * M_BH) / (2 * R * c**2), 0))
    T_N4_val = np.maximum(T_N4(R), 1e-3)
    return 1.1 * relativistic_correction * T_N4_val**(1/4)

# Calculate base temperature across the disk
temperature_base = T_base(R_grid)

# Initialize temperature fluctuations due to flares
temperature_fluctuations = np.zeros_like(temperature_base)

# Flare parameters
R_flare = R_in + (R_out - R_in) / 2  # Radial position of the flare
theta_flare = np.pi  # Angular position of the flare
Delta_R_flare = (R_out - R_in) / 10  # Radial extent of the flare
Delta_theta_flare = np.pi / 10  # Angular extent of the flare
Delta_T_flare = 1e5  # Temperature increase due to the flare in Kelvin

# Determine affected grid cells by the flare
Delta_R_grid = np.abs(R_grid - R_flare)
Delta_theta_grid = np.abs(Theta_grid - theta_flare)
Delta_theta_grid = np.minimum(Delta_theta_grid, 2 * np.pi - Delta_theta_grid)

flare_mask = (Delta_R_grid <= Delta_R_flare / 2) & (Delta_theta_grid <= Delta_theta_flare / 2)

# Apply the flare temperature increase
temperature_fluctuations[flare_mask] += Delta_T_flare

# Total temperature at each grid cell
temperature_total = temperature_base + temperature_fluctuations

# Convert polar to Cartesian coordinates for plotting
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)

# Plotting the accretion disk temperature profile
plt.figure(figsize=(10, 10))
plt.pcolormesh(X / R_ISCO, Y / R_ISCO, temperature_base, shading='auto', cmap='inferno')
plt.colorbar(label='Temperature (K)')
plt.xlabel('X [$R_{ISCO}$]')
plt.ylabel('Y [$R_{ISCO}$]')
plt.title('Accretion Disk Temperature Profile with Flare')
plt.axis('equal')
plt.show()
