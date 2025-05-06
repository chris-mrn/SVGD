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


def plot_kde_multi_model(particle_lists, true_particles, step, x_min, x_max, y_min, y_max,
                          model_names=None, save_dir='Figures'):
    """
    Plots KDE curves for multiple model outputs at a given step.

    Args:
        particle_lists (list of np.ndarray): List of 1D arrays (e.g., all x-coordinates of particles at a step).
        true_particles (np.ndarray): 1D array of true particle values (e.g., x-coordinates).
        step (int): Current timestep.
        x_min, x_max, y_min, y_max (float): Axis limits.
        model_names (list of str): Names for models.
        save_dir (str): Directory to save plot.

    Returns:
        None
    """
    assert all(p.ndim == 1 for p in particle_lists), "All input particles must be 1D arrays"

    if model_names is None:
        model_names = [f"Model {i}" for i in range(len(particle_lists))]

    x_vals = np.linspace(x_min, x_max, 200)

    # KDE for true particles
    kde_true = gaussian_kde(true_particles)
    plt.plot(x_vals, kde_true(x_vals), label="True Particles", linestyle="--", color="black")

    # KDE for models
    colors = plt.cm.viridis(np.linspace(0, 1, len(particle_lists)))
    for i, (samples, name) in enumerate(zip(particle_lists, model_names)):
        kde = gaussian_kde(samples)
        plt.plot(x_vals, kde(x_vals), label=name, linestyle="-", color=colors[i])

    plt.title(f'Density Estimation (KDE) at Step {step}')
    plt.xlabel("Particle Value")
    plt.ylabel("Density")
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"KDE_step_{step}.png")
    plt.savefig(save_path)
    print(f"Saved KDE plot to {save_path}")
    plt.close()


def plot_kde_history(models_hist, X1, model_names=None, save_dir='Figures'):
    """
    Plot only the KDE plots of the first dimension of particle histories for multiple models over iterations.

    Args:
        models_hist (Tensor or np.ndarray): Shape (num_models, n_iter, n_samples, 2)
        X1 (Tensor or np.ndarray): True samples of shape (n_samples, 2)
        model_names (list of str): Optional names for models.
        save_dir (str): Directory to save figures.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Convert to NumPy
    if isinstance(models_hist, torch.Tensor):
        models_hist = models_hist.detach().cpu().numpy()
    if isinstance(X1, torch.Tensor):
        X1 = X1.detach().cpu().numpy()

    num_models, n_iter = models_hist.shape[0], models_hist.shape[1]

    if model_names is None:
        model_names = [f"Model {k}" for k in range(num_models)]

    # Global KDE axis limits from all particles
    all_samples = np.concatenate([X1[:, 0]] + [models_hist[k, :, :, 0].reshape(-1) for k in range(num_models)])
    x_min, x_max = np.min(all_samples), np.max(all_samples)

    # Compute KDE y-axis limits
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals_all = []
    for k in range(num_models):
        kde = gaussian_kde(models_hist[k, :, :, 0].reshape(-1))
        y_vals_all.append(kde(x_vals))
    kde_true = gaussian_kde(X1[:, 0])
    y_vals_true = kde_true(x_vals)

    y_min_kde = min(min(y.min() for y in y_vals_all), y_vals_true.min())
    y_max_kde = max(max(y.max() for y in y_vals_all), y_vals_true.max())

    # Plot KDE every 10 iterations
    for i in range(0, n_iter, 10):
        samples_list = [models_hist[k, i, :, 0] for k in range(num_models)]
        plot_kde_multi_model(samples_list, X1[:, 0], i, x_min, x_max, y_min_kde, y_max_kde,
                             model_names=model_names, save_dir=save_dir)



def plot_model_samples(sample_list, model_names, ground_truth, figsize=(20, 5)):
    """
    Plots samples from different models and the ground truth side by side.

    Args:
        sample_list (list of torch.Tensor): List of tensors, each of shape (n_samples, 2).
        model_names (list of str): Names of the corresponding models.
        ground_truth (torch.Tensor or np.ndarray): Ground truth data, shape (n_samples, 2).
        figsize (tuple): Size of the figure.

    Returns:
        None
    """
    num_models = len(sample_list)
    fig, axs = plt.subplots(1, num_models + 1, figsize=figsize)

    for i, (samples, name) in enumerate(zip(sample_list, model_names)):
        axs[i].scatter(samples.detach()[:, 0], samples.detach()[:, 1], s=1)
        axs[i].set_title(f'{name} Samples')

    axs[-1].scatter(ground_truth[:, 0], ground_truth[:, 1], s=1)
    axs[-1].set_title('Ground Truth Samples')

    plt.tight_layout()
    plt.show()


def plot_particle_trajectories(histories, model_names, X1, figsize=(20, 5), step=1, max_particles=25):
    """
    Plots particle trajectories over time for multiple models, using a subset of particles.

    Args:
        histories (list of list of torch.Tensor): Each element is a list of tensors (timesteps), each of shape (n_particles, 2).
        model_names (list of str): Names of the models.
        X1 (torch.Tensor or np.ndarray): Ground truth samples to display.
        figsize (tuple): Figure size.
        step (int): Plot every `step`-th time step to reduce clutter.
        max_particles (int): Maximum number of particles to plot per model.

    Returns:
        None
    """
    num_models = len(histories)
    fig, axs = plt.subplots(1, num_models + 1, figsize=figsize)

    for i, (hist, name) in enumerate(zip(histories, model_names)):
        hist = [h.detach().cpu() for h in hist]
        n_particles = hist[0].shape[0]

        # Sample a subset of particles
        indices = np.random.choice(n_particles, size=min(max_particles, n_particles), replace=False)

        for j in indices:
            traj = [hist[t][j] for t in range(0, len(hist), step)]
            traj = torch.stack(traj)
            axs[i].plot(traj[:, 0], traj[:, 1], lw=0.5)

        axs[i].set_title(f'{name} Trajectories')
        axs[i].scatter(hist[0][indices, 0], hist[0][indices, 1], s=2, c='green', label='Start')
        axs[i].scatter(hist[-1][indices, 0], hist[-1][indices, 1], s=2, c='red', label='End')
        axs[i].legend()

    axs[-1].scatter(X1[:, 0], X1[:, 1], s=1)
    axs[-1].set_title('Ground Truth')

    plt.tight_layout()
    plt.show()
