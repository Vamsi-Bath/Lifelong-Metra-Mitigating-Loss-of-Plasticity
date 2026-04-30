import os
import torch
import torch.nn as nn
from torch.distributions import Normal


class LegacyContinuousActor(nn.Module):
    """
    Architecture compatible with the old PPO ActorContinuousNetwork.
    """
    def __init__(self, state_dim, action_dim, max_action=None, device="cpu", chkpt_dir=".", actor_lr=3e-4):
        super().__init__()
        self.device = device
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, "actor_continuous.pth")

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # Must match old shape: [action_dim], not [1, action_dim]
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.optimiser = torch.optim.Adam(self.parameters(), lr=actor_lr)
        self.to(device)

    def forward(self, state):
        state = state.to(self.device).float()
        if torch.isnan(state).any() or torch.isinf(state).any():
            state = torch.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)

        mean = self.actor(state)

        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6)

        std = torch.exp(self.log_std).clamp(min=1e-6, max=1.0)
        return Normal(mean, std)

    def save_checkpoint(self):
        os.makedirs(self.chkpt_dir, exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))


class LegacyContinuousCritic(nn.Module):
    """
    Architecture compatible with the old PPO CriticVectorNetwork.
    """
    def __init__(self, state_dim, device="cpu", chkpt_dir=".", critic_lr=3e-4):
        super().__init__()
        self.device = device
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, "critic_vector.pth")

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=critic_lr)
        self.to(device)

    def forward(self, state):
        return self.critic(state)

    def save_checkpoint(self):
        os.makedirs(self.chkpt_dir, exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))