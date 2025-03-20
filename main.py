from models import SVGD, NCSN, MLP2D
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal, Categorical, Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from utils import kl_divergence_kde


device = 'mps'

# Dataset

means = torch.tensor([[0., 0.], [5., 5.]])
stds = torch.rand(2, 2)

mix = Categorical(torch.ones(2,))
comp = Independent(Normal(means, stds), 1)
gmm = MixtureSameFamily(mix, comp)

N_samples = 1000
true_samples = gmm.sample((N_samples,)) # generate  samples

# Model

# SVGD
model_SVGD = SVGD(n_iter=100)

# NCSN
net = MLP2D(hidden_dim=32, num_layers=4)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

model_NCSN = NCSN(net, L=100)
model_NCSN.train(optimizer, 150, true_samples)

# Generate samples
gen_SVGD_samples, hist_SVGD = model_SVGD.sample(gmm)
gen_NCSN_samples, hist_NCSN = model_NCSN.sample()

# detach
gen_SVGD_samples = gen_SVGD_samples.detach().numpy()
hist_SVGD = hist_SVGD


KL_list_SVGD = [kl_divergence_kde(hist_SVGD[i, :, :].unsqueeze(1), true_samples.unsqueeze(1)).detach().numpy() for i in range(hist_SVGD.shape[0])]
KL_list_NCSN = [kl_divergence_kde(hist_NCSN[i, :, :].unsqueeze(1), true_samples.unsqueeze(1)) for i in range(hist_NCSN.shape[0])]


true_samples = true_samples.detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.6)
ax.scatter(gen_NCSN_samples[:, 0], gen_NCSN_samples[:, 1], alpha=0.6)
ax.scatter(gen_SVGD_samples[:, 0], gen_SVGD_samples[:, 1], alpha=0.6)
ax.grid(False)
ax.set_aspect('equal', adjustable='box')
strtitle = "True and generated samples"
ax.set_title(strtitle)
ax.legend(['True samples', 'NCSN samples', 'SVGD samples'])
plt.show()


plt.figure(figsize=(8, 5))  # Set figure size

plt.plot(KL_list_SVGD, label="SVGD", linestyle="-", marker="o", markersize=4)
plt.plot(KL_list_NCSN, label="NCSN", linestyle="-", marker="s", markersize=4)

plt.xlabel("Iteration")  # Label for x-axis
plt.ylabel("KL Divergence")  # Label for y-axis
plt.title("KL Divergence over Iterations")  # Title for the plot
plt.legend()  # Show legend
plt.grid(True, linestyle="--", alpha=0.6)  # Add grid for readability

plt.show()