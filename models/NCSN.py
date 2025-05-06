import torch
import numpy as np

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

    def sample_from(self, X0, T=100, eps=2e-5):
        """
        Generates a sample by refining an initial noisy input using the learned score model.

        Parameters:
        - x_init: The initial noisy sample.
        - T: The number of denoising iterations per noise level.
        - eps: Small value for numerical stability.

        Returns:
        - x_step: The generated clean sample after denoising.
        """
        x_step = X0
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

    def train(self, optimizer, epochs, data):
        """
        Trains the NCSN model using score matching. The model learns to predict the score for noisy inputs
        at different noise levels.

        Parameters:
        - optimizer: The optimizer used for backpropagation.
        - epochs: Number of training epochs.
        - dataloader: DataLoader providing the training data.
        - print_interval: Frequency of loss printing.
        """
        print("Training NCSN...")
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
