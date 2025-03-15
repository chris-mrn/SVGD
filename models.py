import torch

class SVGD:
    def __init__(self, n_iter=100, kernel_name='RBF', step_size=0.1):
        self.n_iter = n_iter
        self.step_size = step_size
        self.kernel_name = kernel_name

    def kernel(self, x, y):

        if self.kernel_name == 'RBF':
            l = 1
            return torch.exp(-torch.norm(x-y)**2/l)

    def sample(self, n : int, d : int, P):
        self.P = P
        particules = torch.randn(n, d).requires_grad_()


        for i in range(self.n_iter):
            print('iteration:', i)
            new_particules = torch.zeros_like(particules)
            c = 0
            for x in particules:

                x = self.step(x, particules, self.step_size)
                new_particules[c] = x
                c += 1

            particules = new_particules

        return particules

    def step(self, x, particules, step_size):

        return x + step_size * self._phistar(x, particules)

    def grad_kernel(self, x, y):
        return -2 * (x-y) * torch.exp(-torch.norm(x-y)**2)

    def score(self, x):
        log_prob = self.P.log_prob(x)
        score = torch.autograd.grad(log_prob, x)[0]
        return score

    def _phistar(self, x, particules):
        val = 0
        for x_j in particules:
            kernel_val = self.kernel(x_j, x)
            grad_kernel_val = self.grad_kernel(x_j, x)
            val += kernel_val * self.score(x_j) + grad_kernel_val
        return val/particules.shape[0]