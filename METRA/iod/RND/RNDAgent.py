import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RNDModule(nn.Module):
    def __init__(self, obs_shape, device, hidden_dim=256, lr=1e-4):
        super(RNDModule, self).__init__()

        # Ensure obs_shape is a tuple or int
        input_dim = int(np.prod(obs_shape)) if hasattr(obs_shape, '__iter__') else int(obs_shape)

        # Target network: fixed, not trained
        self.target_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Predictor network: learns to match target
        self.predictor_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.optimiser = optim.Adam(self.predictor_network.parameters(), lr=lr)
        self.device = device
        self.to(device)

    def calculate_intrinsic_reward(self, next_state):
        """
        Computes RND intrinsic reward: MSE between target and predictor output.
        """
        x = T.tensor(next_state, dtype=T.float32).view(1, -1).to(self.device)
        with T.no_grad():
            target_out = self.target_network(x)
        predictor_out = self.predictor_network(x)
        return T.mean((target_out - predictor_out) ** 2).item()

    def train_rnd(self, batch_of_states):
        """
        Train the predictor to match the target network using MSE loss.
        """
        states = T.tensor(batch_of_states, dtype=T.float32).view(len(batch_of_states), -1).to(self.device)
        with T.no_grad():
            target_out = self.target_network(states)
        predictor_out = self.predictor_network(states)

        loss = T.mean((target_out - predictor_out) ** 2)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item()
