import torch
import numpy as np
import torch.nn as nn


class SVGD:
    def __init__(self, n_iter=100, kernel_name='RBF', step_size=1, optimizer='SGD', batch_size=32):
        """
        Initializes the SVGD object with the number of iterations, kernel type, step size,
        optimizer type, and batch size.
        """
        self.n_iter = n_iter
        self.step_size = step_size
        self.kernel_name = kernel_name
        self.batch_size = batch_size

    def kernel(self, particles):
        """
        Computes the kernel matrix using pairwise squared Euclidean distances.
        """
        X_expanded = particles.unsqueeze(0)
        X_transposed = particles.unsqueeze(1)
        squared_diff = (X_expanded - X_transposed) ** 2
        D = squared_diff.sum(dim=2)

        n = particles.shape[0]
        h = torch.median(squared_diff)
        h_scaled = h / np.log(n)

        return torch.exp(-D / h_scaled)

    def sample(self, P, n=250, d=2):
        """
        Samples from the distribution P using SVGD optimization.
        """
        self.P = P
        self.n = n
        particles = torch.randn(n, d, requires_grad=True)

        optimizer = torch.optim.Adam([particles], lr=0.1)

        x_hist = torch.zeros(self.n_iter + 1, *particles.shape)
        x_hist[0] = particles

        for i in range(self.n_iter):
            print(f'Iteration {i}')
            optimizer.zero_grad()
            particles.grad = -self._phistar(particles)
            optimizer.step()
            particles.grad = None

            x_hist[i + 1] = particles

        return particles, x_hist

    def score(self, particles):
        """
        Computes the score function (gradient of log probability).
        """
        log_prob = self.P.log_prob(particles)
        score = torch.autograd.grad(log_prob.sum(), particles)[0]
        return score

    def _phistar(self, particles):
        """
        Computes the update term for SVGD.
        """
        kernel = self.kernel(particles)
        score = self.score(particles)
        grad_kernel = -0.5 * torch.autograd.grad(kernel.sum(), particles)[0]

        K_T = kernel.permute(*torch.arange(kernel.ndimension() - 1, -1, -1))
        phi = (torch.matmul(K_T, score) + grad_kernel) / particles.shape[0]

        return phi


class NCSN:
    """
    This class implements the Noise Conditional Score Network (NCSN), a generative model
    that learns to reverse a diffusion process. It refines noisy data towards a clean sample
    using a score model that predicts gradients of the log probability of the data at various noise levels.

    Parameters:
    - model: A neural network that predicts the score (gradient of the log probability) for perturbed data.
    - L: The number of discrete noise levels.
    - sigma_low: The lowest noise level used during sampling.
    - sigma_high: The highest noise level used during sampling.
    """

    def __init__(self, model, L=10, sigma_low=0.01, sigma_high=1):
        """
        Initializes the NCSN with the given model and noise parameters.
        """
        self.model = model  # The score model
        self.sigma = [sigma_high]  # Starting with the highest noise level
        self.L = L  # Number of steps (noise levels)

        # Generate a sequence of noise levels between sigma_high and sigma_low
        for _ in range(L-1):
            self.sigma.append(self.sigma[-1] * (sigma_low / sigma_high)**(1/(L-1)))

        self.sigma = torch.tensor(self.sigma)  # Convert noise levels to tensor

    def sample(self, n=1000, d=2, T=100, eps=2e-5):
        """
        Generates a sample by refining an initial noisy input using the learned score model.

        Parameters:
        - x_init: The initial noisy sample.
        - T: The number of denoising iterations per noise level.
        - eps: Small value for numerical stability.

        Returns:
        - x_step: The generated clean sample after denoising.
        """
        x_step = torch.randn(1000, d)
        x_hist = torch.zeros(self.L+1, *x_step.shape)
        x_hist[0] = x_step
        with torch.no_grad():
            for i in range(self.L):
                alpha_i = eps * self.sigma[i]**2 / self.sigma[-1]**2
                for _ in range(T):
                    noise_level = (self.sigma[i * torch.ones(x_step.shape[0], dtype=int), None]).to(x_step.device)
                    x_step = x_step + alpha_i / 2 * self.model(x_step, noise_level) / noise_level + np.sqrt(alpha_i) * torch.randn_like(x_step, device=x_step.device)
                x_hist[i+1] = x_step

        return x_step, x_hist

    def train(self, optimizer, epochs, data, print_interval=100):
        """
        Trains the NCSN model using score matching. The model learns to predict the score for noisy inputs
        at different noise levels.

        Parameters:
        - optimizer: The optimizer used for backpropagation.
        - epochs: Number of training epochs.
        - dataloader: DataLoader providing the training data.
        - print_interval: Frequency of loss printing.
        """
        for epoch in range(epochs):
            total_loss = 0

            x = torch.tensor(data, dtype=torch.float32)  # Convert data to tensor

            sigma_level = torch.randint(0, self.L, (x.shape[0],))  # Randomly choose noise levels for each epoch
            sigma_level = self.sigma[sigma_level, None]

            optimizer.zero_grad()

            # Add noise to the data and compute the loss
            random_noise = torch.randn_like(x, device=x.device)
            x_perturbed = x + sigma_level * random_noise

            noise_predicted = -self.model(x_perturbed, sigma_level)

            loss = ((random_noise - noise_predicted) ** 2).mean()

            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the model

            total_loss += loss.item()

            # Print average loss at each epoch
            print(f"Epoch {epoch} - Total Loss: {total_loss / x.shape[0]}")

class MLP2D(nn.Module):
    """
    Naive MLP for 2D data conditioned on noise level.
    Apply positional encoding to the inputs.
    """
    def __init__(self, hidden_dim, num_layers):
        super(MLP2D, self).__init__()
        self.linpos = nn.Linear(2, 64)
        layers = [nn.Linear(2*64,hidden_dim),nn.ReLU()]
        for _ in range(0,num_layers-2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 2))
        self.mlp = nn.Sequential(*layers)
        self.pe = PE(num_pos_feats=64)

    def forward(self, x, sigma):
        x = torch.cat([self.linpos(x), self.pe(sigma)],dim=1)
        return self.mlp(x)

class PE(nn.Module):
    """
    Positional encoding.
    """
    def __init__(self, num_pos_feats=64, temperature=10000):
        super().__init__()
        dim_t = torch.arange(num_pos_feats)
        self.register_buffer("dim_t", temperature ** (2 * (dim_t // 2) / num_pos_feats))

    def forward(self, x):
        pos_x = x[:, :, None] / self.dim_t
        pos = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3).flatten(1)
        return pos
