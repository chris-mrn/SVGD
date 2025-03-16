import torch

class SVGD:
    def __init__(self, n_iter=100, kernel_name='RBF', step_size=0.1):
        self.n_iter = n_iter
        self.step_size = step_size
        self.kernel_name = kernel_name

    def kernel(self, particles):
        XX = torch.nn.PairwiseDistance(p=2)(particles, particles)
        return torch.exp(-XX)

    def sample(self, n: int, d: int, P):
        self.P = P
        particles = torch.randn(n, d).requires_grad_()
        for _ in range(self.n_iter):
            particles = self.step(particles, self.step_size)
        return particles

    def step(self, particles, step_size):
        return particles + step_size * self._phistar(particles)

    def score(self, particles):
        log_prob = self.P.log_prob(particles)
        score = torch.autograd.grad(log_prob.sum(), particles)[0]
        return score

    def _phistar(self, particles):
        kernel = self.kernel(particles)
        score = self.score(particles)
        # minus due to the derivative regarding the second variable
        grad_kernel = - 0.5 * torch.autograd.grad(kernel.sum(), particles)[0]
        K_T = kernel.permute(*torch.arange(kernel.ndim - 1, -1, -1))
        phi = (torch.matmul(K_T, score) + grad_kernel)/particles.shape[0]

        return phi
