import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import os

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


def plot_kde(f_particles, f_extra_particles, f_true_particles, step, x_min, x_max, y_min, y_max, save_dir='Figures', model_names=None):
    """Plots the KDE estimation of given particles and saves the plot in the 'Figures' folder."""

    # Convert PyTorch tensors to NumPy arrays (not needed if already NumPy)
    f_particles = f_particles
    f_true_particles = f_true_particles
    f_extra_particles = f_extra_particles

    # Define grid for KDE evaluation using the provided global min/max
    x_vals = np.linspace(x_min, x_max, 200)

    # Estimate densities using KDE
    kde_particles = gaussian_kde(f_particles)
    kde_true_particles = gaussian_kde(f_true_particles)
    kde_extra_particles = gaussian_kde(f_extra_particles)

    # Plot KDE curves
    plt.plot(x_vals, kde_particles(x_vals), label="SVGD Particles", linestyle="-", color="blue")
    plt.plot(x_vals, kde_true_particles(x_vals), label="True Particles", linestyle="--", color="red")
    plt.plot(x_vals, kde_extra_particles(x_vals), label="NCSN Particles", linestyle="-.", color="green")

    # Formatting
    plt.title(f'Density Estimation (KDE) at Step {step}')
    plt.xlabel("Particle Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    # Set consistent y-axis range
    plt.ylim(y_min, y_max)

    # Create the 'Figures' directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot in the 'Figures' folder
    save_path = os.path.join(save_dir, f"KDE_step_{step}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    # Close the plot after saving
    plt.close()


def plot_model_history(models_hist, true_samples, model_names=None, save_dir='Figures'):
    """
    Plot particle histories and KL divergences for multiple models.

    Args:
        models_hist (Tensor or np.ndarray): Shape (num_models, n_iter, n_samples, 2)
        true_samples (Tensor): True samples of shape (n_samples, 2)
        model_names (list of str): Optional names for models, default uses Model 0, Model 1, ...
        save_dir (str): Directory to save figures.
    """

    os.makedirs(save_dir, exist_ok=True)

    if isinstance(models_hist, torch.Tensor):
        models_hist = models_hist.detach().cpu().numpy()
    if isinstance(true_samples, torch.Tensor):
        true_samples = true_samples.detach().cpu().numpy()

    num_models = models_hist.shape[0]
    n_iter = models_hist.shape[1]

    if model_names is None:
        model_names = [f"Model {k}" for k in range(num_models)]

    # Compute global axis limits
    all_samples = np.concatenate([true_samples] + [models_hist[k].reshape(-1, 2) for k in range(num_models)], axis=0)
    x_min, x_max = np.min(all_samples[:, 0]), np.max(all_samples[:, 0])
    y_min, y_max = np.min(all_samples[:, 1]), np.max(all_samples[:, 1])

    # Scatter plot for every 10th iteration
    for i in range(0, n_iter, 10):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.5, marker='o', s=5, label="True samples")
        for k in range(num_models):
            ax.scatter(models_hist[k, i, :, 0], models_hist[k, i, :, 1], alpha=0.5, marker='o', s=5, label=model_names[k])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Samples at iteration {i}")
        ax.legend()
        fig.savefig(f'{save_dir}/plot_{i}.png')
        plt.close(fig)

    # KL divergence plot
    plt.figure(figsize=(8, 5))
    for k in range(num_models):
        KL_list = [kl_divergence_kde(torch.tensor(models_hist[k, i, :, :]).unsqueeze(1),
                                     torch.tensor(true_samples).unsqueeze(1)).detach().numpy()
                   for i in range(n_iter)]
        plt.plot(KL_list, label=model_names[k], linestyle="-", marker="o", markersize=4)
    plt.xlabel("Iteration")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence over Iterations")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f'{save_dir}/KL_divergence_plot.png')
    plt.close()

    # KDE plotting
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals_all = []
    for k in range(num_models):
        kde = gaussian_kde(models_hist[k].reshape(-1))
        y_vals_all.append(kde(x_vals))
    kde_true = gaussian_kde(true_samples[:, 0])
    y_vals_true = kde_true(x_vals)

    y_min_kde = min(min(y.min() for y in y_vals_all), y_vals_true.min())
    y_max_kde = max(max(y.max() for y in y_vals_all), y_vals_true.max())

    for i in range(0, n_iter, 10):
        samples_list = [models_hist[k, i, :, 0] for k in range(num_models)]
        plot_kde(*samples_list, true_samples[:, 0], i, x_min, x_max, y_min_kde, y_max_kde, model_names=model_names, save_dir=save_dir)