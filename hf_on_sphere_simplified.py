
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm

# Parameters
l_max = 1
R = 1.0
n_theta = 50
n_phi = 50

# Angular grid
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2*np.pi, n_phi)
theta_grid, phi_grid = np.meshgrid(theta, phi)
dtheta = np.pi / n_theta
dphi = 2 * np.pi / n_phi
dA = np.sin(theta_grid) * dtheta * dphi  # Area element on the sphere

# Real spherical harmonics basis functions
Y00 = np.real(sph_harm(0, 0, phi_grid, theta_grid))
Y10 = np.real(sph_harm(0, 1, phi_grid, theta_grid))
basis = np.array([Y00, Y10])

# Normalize basis
for i in range(len(basis)):
    norm = np.sqrt(np.sum(basis[i]**2 * dA))
    basis[i] /= norm

# Initialize random coefficients
coeffs = np.random.rand(2)
coeffs /= np.linalg.norm(coeffs)

# Get orbital from coefficients
def get_orbital(coeffs, basis):
    return coeffs[0] * basis[0] + coeffs[1] * basis[1]

# Hartree–Fock loop
def hartree_fock(coeffs, basis, max_iter=20, tol=1e-6):
    for it in range(max_iter):
        psi = get_orbital(coeffs, basis)
        rho = psi**2

        V_h = np.zeros_like(rho)
        for i in range(n_phi):
            for j in range(n_theta):
                r1 = np.array([
                    np.sin(theta_grid[i, j]) * np.cos(phi_grid[i, j]),
                    np.sin(theta_grid[i, j]) * np.sin(phi_grid[i, j]),
                    np.cos(theta_grid[i, j])
                ])
                dot_r1r2 = (np.sin(theta_grid) * np.cos(phi_grid) * r1[0] +
                            np.sin(theta_grid) * np.sin(phi_grid) * r1[1] +
                            np.cos(theta_grid) * r1[2])
                dot_r1r2 = np.clip(dot_r1r2, -1.0, 1.0)  # Fix added
                dist = np.sqrt(2 * (1 - dot_r1r2))
                dist[dist < 1e-4] = 1e-4
                V_h[i, j] = np.sum(rho / dist * dA)

        F = np.zeros((2, 2))
        for a in range(2):
            for b in range(2):
                F[a, b] = np.sum(basis[a] * V_h * basis[b] * dA)

        eigvals, eigvecs = np.linalg.eigh(F)
        coeffs_new = eigvecs[:, 0]
        coeffs_new /= np.linalg.norm(coeffs_new)

        if np.linalg.norm(coeffs_new - coeffs) < tol:
            print(f"Converged in {it+1} iterations.")
            break

        coeffs = coeffs_new

    return coeffs, get_orbital(coeffs, basis)

# Run HF
coeffs_HF, psi_HF = hartree_fock(coeffs, basis)

# Plot HF orbital density
plt.figure(figsize=(6, 5))
plt.contourf(phi_grid, theta_grid, psi_HF**2, levels=100, cmap='viridis')
plt.title("Simplified HF Orbital Density on Sphere (Fixed Code)")
plt.xlabel(r"$\phi$")
plt.ylabel(r"$\theta$")
plt.colorbar(label="Electron density")
plt.show()
