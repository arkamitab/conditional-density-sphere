import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d import Axes3D

# Parameters
n_theta = 100
n_phi = 100
n_frames = 30
alpha = 2
R = 1.0

# Angular grid
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2*np.pi, n_phi)
theta_grid, phi_grid = np.meshgrid(theta, phi)

# Inter-electron distance on sphere
def r12(theta1, phi1, theta2, phi2, R=1.0):
    cos_gamma = (np.cos(theta1)*np.cos(theta2) +
                 np.sin(theta1)*np.sin(theta2)*np.cos(phi1 - phi2))
    return 2 * R * np.sqrt(0.5 * (1 - cos_gamma))

# Generate conditional densities
theta1_samples = np.linspace(0, np.pi, n_frames)
phi1_fixed = 0
frame_data = []

for theta1_fixed in theta1_samples:
    r = r12(theta1_fixed, phi1_fixed, theta_grid, phi_grid)
    psi_cond = np.exp(-alpha * r)
    prob_cond = psi_cond**2
    prob_cond /= np.sum(prob_cond) * (np.pi/n_theta) * (2*np.pi/n_phi)
    frame_data.append(prob_cond)

# Convert spherical to Cartesian coordinates
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

x, y, z = spherical_to_cartesian(R, theta_grid, phi_grid)

# Setup colormap and normalization
cmap = get_cmap('viridis')
norm = Normalize(vmin=0, vmax=np.max(frame_data))

# Setup figure and axis
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

def update_3d(frame):
    ax.clear()
    ax.set_title(f'Electron 1 fixed at θ = {theta1_samples[frame]:.2f}, φ = 0')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_axis_off()
    colors = cmap(norm(frame_data[frame]))
    ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)

    # Fixed electron 1 location
    theta1 = theta1_samples[frame]
    phi1 = 0
    x1, y1, z1 = spherical_to_cartesian(1.1, theta1, phi1)
    ax.scatter([x1], [y1], [z1], color='red', s=100)

# Create animation
ani_3d = animation.FuncAnimation(fig, update_3d, frames=n_frames, interval=200)

# Save as GIF
ani_3d.save("conditional_density_3d_animation.gif", writer='pillow', fps=5)

