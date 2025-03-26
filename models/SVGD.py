import torch
import numpy as np

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

        return torch.exp(-D /(2*h_scaled))

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
