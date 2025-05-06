import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Normal, Categorical, Independent
from torch.distributions.mixture_same_family import MixtureSameFamily

from models.svgd import SVGD
from models.ncsn import NCSN
from models.fm import GaussFlowMatching_OT
from models.utils import MLP2D, FMnet
from utils import plot_model_samples, plot_particle_trajectories, plot_kde_history


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
dim = 2
X0 = torch.randn(N_samples, dim)
X1 = gmm.sample((N_samples,))

# Initialize models

# FM
h = 64
n_step = 10
net_fm = nn.Sequential(
    nn.Linear(dim + 1, h),
    nn.ELU(),
    nn.Linear(h, h),
    nn.ELU(),
    nn.Linear(h, h),
    nn.ELU(),
    nn.Linear(h, dim))

model_FM = GaussFlowMatching_OT(net_fm)
optimizer_fm = torch.optim.Adam(net_fm.parameters(), 1e-2)

# NCSN
net = MLP2D(hidden_dim=h, num_layers=4)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
model_NCSN = NCSN(net, L=10)

# SVGD
model_SVGD = SVGD(n_iter=100, step_size=1e-1)


# Training
model_NCSN.train(optimizer, 150, X1)
model_FM.train(optimizer_fm, X1, X0, n_epochs=1000)


# Generate samples
gen_SVGD_samples, hist_SVGD = model_SVGD.sample(gmm, n=N_samples)
gen_NCSN_samples, hist_NCSN = model_NCSN.sample_from(X0)
gen_FM_samples, hist_FM = model_FM.sample_from(X0)


# Plots
plot_model_samples(
    [gen_SVGD_samples, gen_NCSN_samples, gen_FM_samples],
    ['SVGD', 'NCSN', 'FM'],
    X1
)

plot_particle_trajectories(
    [hist_SVGD, hist_NCSN, hist_FM],
    ['SVGD', 'NCSN', 'FM'],
    X1
)


models_hist = torch.stack([hist_SVGD, hist_NCSN, hist_FM], dim=0)

# Plot KDE histories
plot_kde_history(
    models_hist,
    X1,
    model_names=['SVGD', 'NCSN', 'FM'],
    save_dir='kde_plots'
)