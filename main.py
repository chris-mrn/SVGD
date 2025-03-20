import os
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal, Categorical, Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from models import SVGD, NCSN, MLP2D
from utils import kl_divergence_kde, plot_kde

# Set device and dataset parameters
device = 'mps'
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

# Save scatter plots for every 10th iteration
for i in range(0, hist_SVGD.shape[0], 10):
    fig, ax = plt.subplots()
    ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.5, marker='o', s=5)
    ax.scatter(hist_NCSN[i, :, 0], hist_NCSN[i, :, 1], alpha=0.5, marker='o', s=5)
    ax.scatter(hist_SVGD[i, :, 0].detach().numpy(), hist_SVGD[i, :, 1].detach().numpy(), alpha=0.5, marker='o', s=5)
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
for i in range(0, hist_SVGD.shape[0], 10):
    plot_kde(hist_SVGD[i, :, 0], hist_NCSN[i, :, 0], true_samples[:, 0], i)
