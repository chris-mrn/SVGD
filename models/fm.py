import torch

class GaussFlowMatching_OT:
    def __init__(self, neural_net, L=10):
        self.net = neural_net
        self.L = L
        self.loss_fn = torch.nn.MSELoss()

    def train(self, optimizer, X1, X0, n_epochs=10):
        print("Training flow matching...")

        for i in range(n_epochs):
            t = torch.rand(len(X1), 1)
            x_t = (1 - t) * X0 + t * X1
            dx_t = X1-X0
            optimizer.zero_grad()
            loss = self.loss_fn(self.flow(x_t, t), dx_t)
            loss.backward()
            if i % 100 == 0 :
                print('Loss:', loss.item())
            optimizer.step()

    def flow(self, x_t,t):
        return self.net(torch.cat((t, x_t), -1))

    def step(self, x_t, t_start, t_end):
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        # For simplicity, using midpoint ODE solver in this example
        return x_t + (t_end - t_start) * self.flow(x_t + self.flow(x_t, t_start) * (t_end - t_start) / 2,
        t_start + (t_end - t_start) / 2)

    def sample_from(self, X0, n_steps=10):
        time_steps = torch.linspace(0, 1.0, n_steps + 1)
        x = X0
        hist = torch.zeros(n_steps+1, *X0.shape)
        hist[0] = x
        for i in range(n_steps):
            x = self.step(x, time_steps[i], time_steps[i + 1])
            hist[i+1] = x
        return x, hist

    def coupling(self,):
        # iterrator that generates data according to the coupling
        pass
