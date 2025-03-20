import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def kde_log_prob(samples, query_points, bandwidth=0.1):
    """Compute log density estimation via KDE."""
    dists = torch.cdist(query_points, samples)  # Compute pairwise distances
    kernel_vals = torch.exp(-0.5 * (dists / bandwidth) ** 2)  # Gaussian Kernel
    density = kernel_vals.mean(dim=1) / (bandwidth * (2 * torch.pi) ** 0.5)
    return torch.log(density + 1e-8)  # Avoid log(0)

def kl_divergence_kde(p_samples, q_samples, bandwidth=0.1):
    """Compute KL divergence between two sets of particles using KDE."""
    log_p = kde_log_prob(p_samples, p_samples, bandwidth)  # log p(x)
    log_q = kde_log_prob(q_samples, p_samples, bandwidth)  # log q(x)
    return torch.mean(log_p - log_q)  # KL divergence

def plot_kde(f_particles, f_true_particles, step):
    """Plots the KDE estimation of given particles."""

    # Convert PyTorch tensors to NumPy arrays
    f_particles = f_particles.detach().numpy()
    f_true_particles = f_true_particles.detach().numpy()

    # Define grid for KDE evaluation
    x_min, x_max = min(f_particles.min(), f_true_particles.min()), max(f_particles.max(), f_true_particles.max())
    x_vals = np.linspace(x_min, x_max, 200)

    # Estimate densities using KDE
    kde_particles = gaussian_kde(f_particles)
    kde_true_particles = gaussian_kde(f_true_particles)

    # Plot KDE curves
    plt.plot(x_vals, kde_particles(x_vals), label="Estimated Particles", linestyle="-", color="blue")
    plt.plot(x_vals, kde_true_particles(x_vals), label="True Particles", linestyle="--", color="red")

    # Formatting
    plt.title(f'Density Estimation (KDE) at Step {step}')
    plt.xlabel("Particle Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()