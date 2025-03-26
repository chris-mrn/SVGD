import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Normal, Categorical, Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from scipy.stats import gaussian_kde

from models.SVGD import SVGD
from models.NCSN import NCSN
from models.utils import MLP2D
from utils import kl_divergence_kde, plot_kde


# Set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# dataset
means = torch.tensor([[-2., -3.], [5., 4.]])
stds = torch.rand(2, 2)
mix_param = torch.tensor([1, 0.8])

# Define mixture model
mix = Categorical(mix_param)
comp = Independent(Normal(means, stds), 1)
gmm = MixtureSameFamily(mix, comp)

# Generate true samples
N_samples = 1000
true_samples = gmm.sample((N_samples,))

# Initialize models
n_iter = 100
model_SVGD = SVGD(n_iter=n_iter, step_size=1e-1)
net = MLP2D(hidden_dim=32, num_layers=4)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
model_NCSN = NCSN(net, L=n_iter)
model_NCSN.train(optimizer, 150, true_samples)

# Generate samples
gen_SVGD_samples, hist_SVGD = model_SVGD.sample(gmm, n=N_samples)
gen_NCSN_samples, hist_NCSN = model_NCSN.sample(n=N_samples)

# Save generated samples
gen_SVGD_samples = gen_SVGD_samples.detach().numpy()

# Create 'Figures' directory if not exists
os.makedirs('Figures', exist_ok=True)

# Determine global axis limits based on true_samples, hist_NCSN, and hist_SVGD
all_samples = np.concatenate([
    true_samples,
    hist_NCSN.reshape(-1, 2),
    hist_SVGD.reshape(-1, 2).detach().numpy()
], axis=0)

x_min, x_max = np.min(all_samples[:, 0]), np.max(all_samples[:, 0])
y_min, y_max = np.min(all_samples[:, 1]), np.max(all_samples[:, 1])

# Save scatter plots for every 10th iteration
for i in range(0, hist_SVGD.shape[0], 10):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.5, marker='o', s=5)
    ax.scatter(hist_NCSN[i, :, 0], hist_NCSN[i, :, 1], alpha=0.5, marker='o', s=5)
    ax.scatter(hist_SVGD[i, :, 0].detach().numpy(), hist_SVGD[i, :, 1].detach().numpy(), alpha=0.5, marker='o', s=5)

    # Set the same axis limits for each plot
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title("True and generated samples")
    ax.legend(['True samples', 'NCSN samples', 'SVGD samples'])

    fig.savefig(f'Figures/plot_{i}.png')
    plt.close(fig)

# Calculate KL divergence for each model
KL_list_SVGD = [kl_divergence_kde(hist_SVGD[i, :, :].unsqueeze(1), true_samples.unsqueeze(1)).detach().numpy() for i in range(hist_SVGD.shape[0])]
KL_list_NCSN = [kl_divergence_kde(hist_NCSN[i, :, :].unsqueeze(1), true_samples.unsqueeze(1)) for i in range(hist_NCSN.shape[0])]

# Plot KL divergence over iterations
plt.figure(figsize=(8, 5))
plt.plot(KL_list_SVGD, label="SVGD", linestyle="-", marker="o", markersize=4)
plt.plot(KL_list_NCSN, label="NCSN", linestyle="-", marker="s", markersize=4)
plt.xlabel("Iteration")
plt.ylabel("KL Divergence")
plt.title("KL Divergence over Iterations")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig('Figures/KL_divergence_plot.png')
plt.close()


# Detach and plot KDE for every 10th iteration
hist_SVGD = hist_SVGD.detach().numpy()
hist_NCSN = hist_NCSN.detach().numpy()

# Compute global min/max for consistent axis range across all plots
all_particles = np.concatenate([
    hist_SVGD.reshape(-1, 1),
    hist_NCSN.reshape(-1, 1),
    true_samples[:, 0].reshape(-1, 1)
], axis=0)

x_min, x_max = np.min(all_particles), np.max(all_particles)

# Compute global y-axis range by evaluating densities at a common set of x-values
x_vals_for_kde = np.linspace(x_min, x_max, 200)
kde_particles = gaussian_kde(hist_SVGD.flatten())
kde_true_particles = gaussian_kde(true_samples[:, 0])
kde_extra_particles = gaussian_kde(hist_NCSN.flatten())

y_vals_particles = kde_particles(x_vals_for_kde)
y_vals_true_particles = kde_true_particles(x_vals_for_kde)
y_vals_extra_particles = kde_extra_particles(x_vals_for_kde)

y_min, y_max = min(np.min(y_vals_particles), np.min(y_vals_true_particles), np.min(y_vals_extra_particles)), \
               max(np.max(y_vals_particles), np.max(y_vals_true_particles), np.max(y_vals_extra_particles))

# Plot for every 10th iteration
for i in range(0, hist_SVGD.shape[0], 10):
    plot_kde(hist_SVGD[i, :, 0], hist_NCSN[i, :, 0], true_samples[:, 0], i, x_min, x_max, y_min, y_max)
