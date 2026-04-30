import torch
import torch.nn as nn

class ContextEncoder(nn.Module):
    def __init__(self, obs_dim, act_dim, z_dim, hidden=256):
        super().__init__()
        in_dim = obs_dim + act_dim + 1 + obs_dim
        self.pre = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, z_dim)
        self.logstd = nn.Linear(hidden, z_dim)

    def forward(self, s, a, r, sp):
        x = torch.cat([s, a, r, sp], dim=-1)
        x = self.pre(x)
        _, h = self.gru(x)
        h = h.squeeze(0)
        mu = self.mu(h)
        logstd = torch.clamp(self.logstd(h), -5, 2)
        return mu, logstd


class ContextPrior(nn.Module):
    def __init__(self, z_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logstd = nn.Linear(hidden, z_dim)

    def forward(self, z_prev):
        h = self.net(z_prev)
        mu = self.mu(h)
        logstd = torch.clamp(self.logstd(h), -5, 2)
        return mu, logstd


class ContextDecoder(nn.Module):
    def __init__(self, obs_dim, act_dim, z_dim, hidden=256):
        super().__init__()
        in_dim = obs_dim + act_dim + z_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.next_obs_head = nn.Linear(hidden, obs_dim)
        self.reward_head = nn.Linear(hidden, 1)

    def forward(self, s, a, c):
        
        x = torch.cat([s, a, c], dim=-1)
        h = self.net(x)
        pred_next_obs = self.next_obs_head(h)
        pred_reward = self.reward_head(h)
        return pred_next_obs, pred_reward


def reparameterize(mu, logstd):
    std = torch.exp(logstd)
    eps = torch.randn_like(std)
    return mu + eps * std


def gaussian_kl(mu_q, logstd_q, mu_p, logstd_p):
    var_q = torch.exp(2 * logstd_q)
    var_p = torch.exp(2 * logstd_p)
    kl = (logstd_p - logstd_q) + (var_q + (mu_q - mu_p) ** 2) / (2 * var_p) - 0.5
    return kl.sum(dim=-1).mean()