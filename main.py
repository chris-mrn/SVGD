from models import SVGD, NCSN, MLP2D
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal, Categorical, Independent
from torch.distributions.mixture_same_family import MixtureSameFamily

device = 'mps'

# Dataset

means = torch.tensor([[0., 0.], [5., 5.]])
stds = torch.rand(2, 2)

mix = Categorical(torch.ones(2,))
comp = Independent(Normal(means, stds), 1)
gmm = MixtureSameFamily(mix, comp)

N_samples = 1000
true_samples = gmm.sample((N_samples,)).detach().numpy()  # generate  samples

# Model

# SVGD
model_SVGD = SVGD(n_iter=100)

# NCSN
net = MLP2D(hidden_dim=32, num_layers=4)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

model_NCSN = NCSN(net, L=20)
model_NCSN.train(optimizer, 150, true_samples)

# Generate samples
gen_SVGD_samples = model_SVGD.sample(gmm).detach().numpy()
gen_NCSN_samples = model_NCSN.sample()


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
plt.savefig('target_reference_datasets.pdf', format="pdf")
