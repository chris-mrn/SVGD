import os
import torch
import numpy as np
from torch.distributions import Normal, Categorical, Independent
from torch.distributions.mixture_same_family import MixtureSameFamily

from models.svgd import SVGD
from models.ncsn import NCSN
from models.utils import MLP2D
from utils import plot_model_history


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


models_hist = torch.stack([hist_SVGD, hist_NCSN], dim=0)


plot_model_history(models_hist, true_samples)