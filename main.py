from models import SVGD
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal, Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily

device = 'mps'

model = SVGD(n_iter=500)

mean =  torch.Tensor([-2, 3])
covariance_matrix = 5 * torch.Tensor([[0.2260, 0.1652],[0.1652, 0.6779]])

P_gauss = torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance_matrix)

mix = Categorical(torch.tensor([1/3, 2/3]))
comp = Normal(torch.tensor([-10., 9.]), torch.tensor([1., 1.]))
P_mix = MixtureSameFamily(mix, comp)
P = torch.distributions.Normal(torch.Tensor([3]), torch.Tensor([5]))

samples = model.sample(1000, 1, P_mix).detach().numpy()

true_samples = P_mix.sample((1000,)).detach().numpy()

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(samples, bins=20)
axs[1].hist(true_samples, bins=20)

plt.show()
