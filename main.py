from models import SVGD
import torch
import matplotlib.pyplot as plt

device = 'mps'

model = SVGD(n_iter=10)

mean =  torch.Tensor([-0.6871, 0.8010])
covariance_matrix = 5 * torch.Tensor([[0.2260, 0.1652],[0.1652, 0.6779]])

P_gauss = torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance_matrix)

samples = model.sample(100, 2, P_gauss).detach().numpy()


# Extract x and y coordinates
x = samples[:, 0]
y = samples[:, 1]

# Plot
plt.scatter(x, y, marker='o', color='b', label="Points")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("2D Point Plot")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()