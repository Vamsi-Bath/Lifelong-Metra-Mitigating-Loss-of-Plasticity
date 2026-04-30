import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np


class RunningMeanStd:
    """
    Running mean/std used by the other PPO implementation.
    """
    def __init__(self, shape, device):
        self.mean = T.zeros(shape, device=device)
        self.var = T.ones(shape, device=device)
        self.count = 1e-4
        self.device = device

    def to(self, device):
        self.device = device
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self

    def update(self, x):
        if not isinstance(x, T.Tensor):
            x = T.as_tensor(x, dtype=T.float32, device=self.device)
        else:
            x = x.to(self.device, dtype=T.float32)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.shape[0] < 2:
            return

        batch_mean = T.mean(x, dim=0)
        batch_var = T.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + T.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


def _build_checkpoint_path(chkpt_dir, model_name, suffix):
    return os.path.join(chkpt_dir, f"{model_name}_{suffix}.pth")


class ActorVectorNetwork(nn.Module):
    """
    Discrete vector actor from your original code.
    """
    def __init__(self, n_actions, input_dims, alpha, device, chkpt_dir, model_name,
                 fc1_dims=256, fc2_dims=256):
        super().__init__()
        self.chkpt_dir = chkpt_dir
        self.model_name = model_name
        self.checkpoint_file = _build_checkpoint_path(chkpt_dir, model_name, "actor")

        self.actor = nn.Sequential(
            nn.Linear(input_dims[0], fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimiser = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        probs = self.actor(state)
        return Categorical(probs)

    def save_checkpoint(self):
        os.makedirs(self.chkpt_dir, exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            raise FileNotFoundError(f"Missing actor checkpoint: {self.checkpoint_file}")
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


class CriticVectorNetwork(nn.Module):
    """
    Compatible critic for the other PPO implementation:
    tanh, 3 hidden layers, 256/256/128.
    """
    def __init__(self, input_dims, alpha, device, chkpt_dir, model_name):
        super().__init__()
        self.chkpt_dir = chkpt_dir
        self.model_name = model_name
        self.checkpoint_file = _build_checkpoint_path(chkpt_dir, model_name, "critic")

        self.fc1 = nn.Linear(input_dims[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.value_layer = nn.Linear(128, 1)

        T.nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        T.nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        T.nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        T.nn.init.orthogonal_(self.value_layer.weight, gain=1.0)

        T.nn.init.zeros_(self.fc1.bias)
        T.nn.init.zeros_(self.fc2.bias)
        T.nn.init.zeros_(self.fc3.bias)
        T.nn.init.zeros_(self.value_layer.bias)

        self.optimiser = optim.Adam(self.parameters(), lr=alpha, eps=1e-5)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        x = T.tanh(self.fc3(x))
        return self.value_layer(x)

    def save_checkpoint(self):
        os.makedirs(self.chkpt_dir, exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            raise FileNotFoundError(f"Missing critic checkpoint: {self.checkpoint_file}")
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


class ActorPixelNetwork(nn.Module):
    """
    Left here for completeness. This is your original pixel actor.
    """
    def __init__(self, n_actions, input_dims, alpha, device, chkpt_dir, model_name):
        super().__init__()
        self.chkpt_dir = chkpt_dir
        self.model_name = model_name
        self.checkpoint_file = _build_checkpoint_path(chkpt_dir, model_name, "actor")

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dims[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv_output_dim = self._calculate_conv_output(input_dims)
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_dim, 512),
            nn.ReLU()
        )
        self.pi = nn.Linear(512, n_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def _calculate_conv_output(self, input_dims):
        with T.no_grad():
            dummy_input = T.zeros(1, *input_dims)
            output = self.conv(dummy_input)
            return int(T.prod(T.tensor(output.size()[1:])))

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        logits = self.pi(features)
        return Categorical(logits=logits)

    def save_checkpoint(self):
        os.makedirs(self.chkpt_dir, exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            raise FileNotFoundError(f"Missing actor checkpoint: {self.checkpoint_file}")
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


class CriticPixelNetwork(nn.Module):
    """
    Left here for completeness. This is your original pixel critic.
    """
    def __init__(self, input_dims, alpha, device, chkpt_dir, model_name):
        super().__init__()
        self.chkpt_dir = chkpt_dir
        self.model_name = model_name
        self.checkpoint_file = _build_checkpoint_path(chkpt_dir, model_name, "critic")

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dims[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv_output_dim = self._calculate_conv_output(input_dims)
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_dim, 512),
            nn.ReLU()
        )
        self.v = nn.Linear(512, 1)

        self.optimiser = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def _calculate_conv_output(self, input_dims):
        with T.no_grad():
            dummy_input = T.zeros(1, *input_dims)
            output = self.conv(dummy_input)
            return int(T.prod(T.tensor(output.size()[1:])))

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return self.v(features)

    def save_checkpoint(self):
        os.makedirs(self.chkpt_dir, exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            raise FileNotFoundError(f"Missing critic checkpoint: {self.checkpoint_file}")
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


class ActorContinuousNetwork(nn.Module):
    """
    Compatible actor for the other PPO implementation:
    tanh, 3 hidden layers, 256/256/128, log_std shape [1, action_dim].
    """
    def __init__(self, n_actions, input_dims, alpha, device, chkpt_dir, model_name, max_action=1.0):
        super().__init__()
        self.chkpt_dir = chkpt_dir
        self.model_name = model_name
        self.checkpoint_file = _build_checkpoint_path(chkpt_dir, model_name, "actor")
        self.max_action = float(max_action)

        self.fc1 = nn.Linear(input_dims[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.mean_layer = nn.Linear(128, n_actions)

        self.log_std = nn.Parameter(T.ones(1, n_actions) * -0.5)

        T.nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        T.nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        T.nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        T.nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)

        T.nn.init.zeros_(self.fc1.bias)
        T.nn.init.zeros_(self.fc2.bias)
        T.nn.init.zeros_(self.fc3.bias)
        T.nn.init.zeros_(self.mean_layer.bias)

        self.optimiser = optim.Adam(self.parameters(), lr=alpha, eps=1e-5)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device).float()
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        x = T.tanh(self.fc3(x))

        mean = self.max_action * T.tanh(self.mean_layer(x))
        log_std = T.clamp(self.log_std, -20, 2).expand_as(mean)
        std = T.exp(log_std)
        return Normal(mean, std)

    def save_checkpoint(self):
        os.makedirs(self.chkpt_dir, exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            raise FileNotFoundError(f"Missing actor checkpoint: {self.checkpoint_file}")
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))