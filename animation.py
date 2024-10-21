# accretion_disk_animation.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from scipy.interpolate import interp1d

# Constants
G = 6.67430e-11     # Gravitational constant (m^3 kg^-1 s^-2)
c = 2.998e8         # Speed of light (m/s)
kB = 1.380649e-23   # Boltzmann constant (J/K)
M_sol = 2e30        # Solar mass (kg)
sigma = 5.67e-8     # Stefan-Boltzmann constant (W/m^2/K^4)
h = 6.626e-34       # Planck constant (J·s)

# Black Hole and Accretion Parameters
M_BH = M_sol * 7e8  # Black hole mass (kg)
dot_M = 1e23        # Accretion rate (kg/s)

# Calculate ISCO radius for a non-rotating Black hole (meters)
R_ISCO = 6 * G * M_BH / c**2

# Grid Setup
num_points_anim = 200               # Number of radial grid points (adjusted for animation)
num_angular_sections_anim = 100     # Number of angular grid points (adjusted for animation)

# Radial Range
R_in = R_ISCO                    # Inner radius (meters)
R_out = 50 * R_ISCO              # Outer radius (meters) reduced for visualization
radii_anim = np.linspace(R_in, R_out, num_points_anim)   # Radial positions (meters)
dr_anim = radii_anim[1] - radii_anim[0]                  # Radial grid spacing (meters)
theta_coords_anim = np.linspace(0, 2 * np.pi, num_angular_sections_anim, endpoint=False)  # Angular positions (radians)
dtheta_anim = theta_coords_anim[1] - theta_coords_anim[0]  # Angular grid spacing (radians)

# Temperature Profile Function
def T_GR(R):
    """
    Computes the temperature profile T_GR(R) based on gravitational radius scaling.
    Ensures temperature is above a minimum threshold.
    """
    T_N4 = (3 * G * M_BH * dot_M) / (8 * np.pi * R**3 * sigma)
    T_eff = 1.1 * ((1 - (3 * G * M_BH) / (2 * R * c**2))**0.5) * np.maximum(T_N4, 1e-3)**0.25
    return T_eff

# Base Temperature Profile
temperature_anim = T_GR(radii_anim)
temperature_anim = np.maximum(temperature_anim, 273)  # Set minimum temperature to 273 K

# Time Grid (Reduced Time Points for Animation)
warm_up_time = 0                # Warm-up time (seconds)
total_simulation_time = 10000   # Total simulation time (seconds)
time_points_anim = 10000          # Number of time points (adjusted for animation)
times_anim = np.linspace(-warm_up_time, total_simulation_time, time_points_anim)
dt_anim = times_anim[1] - times_anim[0]

# Initialize Fluctuating Temperature Array
fluctuating_temperature_anim = np.zeros((num_points_anim, num_angular_sections_anim, len(times_anim)), dtype=np.float32)

# Flare Parameters
num_flares = 1000  # Number of flares

# Flare Temperature Increases
flare_temp_increase_min = 1     # Minimum temperature increase factor
flare_temp_increase_max = 1.9   # Maximum temperature increase factor
flare_temp_increase_power_law_index = 1.5  # Power-law index
temperature_increases = flare_temp_increase_min + (flare_temp_increase_max - flare_temp_increase_min) * np.random.power(a=flare_temp_increase_power_law_index, size=num_flares)

# Randomly Assign Flare Times, Radii, and Angles
# Set replace=True to allow sampling with replacement
flare_times = np.random.choice(times_anim, size=num_flares, replace=True)
flare_radii = np.random.choice(radii_anim, size=num_flares, replace=True)
flare_angles = np.random.randint(0, num_angular_sections_anim, size=num_flares)

# Flare Durations
flare_rise_time_min = 10    # Minimum flare rise time (seconds)
flare_rise_time_max = 50    # Maximum flare rise time (seconds)
flare_e_folding_time_min = 20  # Minimum e-folding time (seconds)
flare_e_folding_time_max = 100  # Maximum e-folding time (seconds)
flare_decay_e_folding_multiplicator = 5   # Number of e-folding times in decay phase

# Flare Rise Times (seconds)
flare_rise_times = np.random.randint(flare_rise_time_min, flare_rise_time_max + 1, size=num_flares)

# E-folding Times for Exponential Decay (seconds)
flare_e_folding_times = np.random.randint(flare_e_folding_time_min, flare_e_folding_time_max + 1, size=num_flares)

# Decay Durations
flare_decay_times = flare_decay_e_folding_multiplicator * flare_e_folding_times

# Total Flare Durations
total_flare_durations = flare_rise_times + flare_decay_times

# Flare Sizes (Radial and Angular)
min_flare_size_r = 1e8      # Minimum flare radial size (meters)
max_flare_size_r = 1e9     # Maximum flare radial size (meters)
min_flare_size_theta = np.pi / 100  # Minimum flare angular size (radians)
max_flare_size_theta = np.pi / 10   # Maximum flare angular size (radians)
flare_size_r_grid_multiplier = 10  # Minimum radial grid cells for flare size
flare_size_theta_grid_multiplier = 10  # Minimum angular grid cells for flare size

# Flare Sizes in Physical Units
flare_size_r = np.random.uniform(min_flare_size_r, max_flare_size_r, size=num_flares)
flare_size_theta = np.random.uniform(min_flare_size_theta, max_flare_size_theta, size=num_flares)

# Ensure Flare Sizes are Larger Than Grid Spacing
flare_size_r = np.maximum(flare_size_r, flare_size_r_grid_multiplier * dr_anim)
flare_size_theta = np.maximum(flare_size_theta, flare_size_theta_grid_multiplier * dtheta_anim)

# Apply Flares to the Temperature Grid
for i in range(num_flares):
    flare_r = flare_radii[i]
    flare_theta = flare_angles[i] * dtheta_anim
    flare_size_r_i = flare_size_r[i]
    flare_size_theta_i = flare_size_theta[i]

    # Time Indices
    t_start = np.argmin(np.abs(times_anim - flare_times[i]))
    t_rise = flare_rise_times[i]
    t_decay = flare_decay_times[i]
    t_end = min(t_start + t_rise + t_decay, len(times_anim))
    flare_time_indices = np.arange(t_start, t_end)

    # Time Since Flare Start
    time_since_start = times_anim[flare_time_indices] - times_anim[t_start]

    # Rising and Decay Phases
    rise_indices = time_since_start <= t_rise
    decay_indices = time_since_start > t_rise

    # Radial and Angular Indices
    r_indices = np.where(np.abs(radii_anim - flare_r) <= flare_size_r_i / 2)[0]
    angular_diffs = np.abs(theta_coords_anim - flare_theta)
    angular_diffs = np.minimum(angular_diffs, 2 * np.pi - angular_diffs)
    theta_indices = np.where(angular_diffs <= flare_size_theta_i / 2)[0]

    # Skip if No Cells are Affected
    if len(r_indices) == 0 or len(theta_indices) == 0:
        continue

    # Meshgrid of Affected Indices
    r_grid, theta_grid = np.meshgrid(r_indices, theta_indices, indexing='ij')

    # Temperature Increase Array
    delta_temperature_cells = np.zeros((len(r_indices), len(theta_indices), len(flare_time_indices)))

    # Rising Phase (Linear Rise)
    if np.any(rise_indices):
        rise_times = time_since_start[rise_indices]
        rise_fraction = rise_times / t_rise
        delta_temperature_rise = ((temperature_increases[i] - 1) * temperature_anim[r_indices][:, np.newaxis, np.newaxis]) * rise_fraction[np.newaxis, np.newaxis, :]
        delta_temperature_cells[:, :, rise_indices] = delta_temperature_rise

    # Decay Phase (Exponential Decay)
    if np.any(decay_indices):
        decay_times = time_since_start[decay_indices] - t_rise
        tau = flare_e_folding_times[i]
        decay_fraction = np.exp(-decay_times / tau)
        delta_temperature_decay = ((temperature_increases[i] - 1) * temperature_anim[r_indices][:, np.newaxis, np.newaxis]) * decay_fraction[np.newaxis, np.newaxis, :]
        delta_temperature_cells[:, :, decay_indices] = delta_temperature_decay

    # Update Fluctuating Temperature
    fluctuating_temperature_anim[r_grid[:, :, np.newaxis], theta_grid[:, :, np.newaxis], flare_time_indices] += delta_temperature_cells

# Compute Total Temperature
total_temperature_anim = temperature_anim[:, np.newaxis, np.newaxis] + fluctuating_temperature_anim

# Visualization Setup
# Create a meshgrid for plotting
Theta_anim, Radii_anim = np.meshgrid(theta_coords_anim, radii_anim)

# Convert polar coordinates to Cartesian for plotting
X_anim = Radii_anim * np.cos(Theta_anim)
Y_anim = Radii_anim * np.sin(Theta_anim)

# Prepare figure and axis for animation
fig, ax = plt.subplots(figsize=(8, 8))
norm = colors.LogNorm(vmin=temperature_anim.min(), vmax=temperature_anim.max() * 1.5)
cmap = plt.get_cmap('magma')

# Initial plot
c = ax.pcolormesh(X_anim, Y_anim, total_temperature_anim[:, :, 0], norm=norm, cmap=cmap, shading='auto')
ax.set_aspect('equal')
ax.set_xlim(-R_out, R_out)
ax.set_ylim(-R_out, R_out)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Accretion Disk Temperature Distribution')
fig.colorbar(c, ax=ax, label='Temperature (K)')

# Animation function
def animate(i):
    c.set_array(total_temperature_anim[:, :, i].ravel())
    ax.set_title(f'Accretion Disk Temperature Distribution at t = {times_anim[i]:.1f} s')
    return c,

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=len(times_anim), interval=50, blit=False)

# To save the animation, uncomment the following lines and ensure you have the necessary writers installed
# anim.save('accretion_disk_animation.mp4', writer='ffmpeg', fps=20)
# anim.save('accretion_disk_animation.gif', writer='imagemagick', fps=20)

# Display the animation
plt.show()
