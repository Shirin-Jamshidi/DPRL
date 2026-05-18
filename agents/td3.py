import torch
import torch.nn as nn
import torch.optim as optim
from agents.base import RLAgent
from buffers.replay_buffer import ReplayBuffer


class MLP(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out)
        )

    def forward(self, x):
        return self.net(x)


class TD3Agent(RLAgent):
    def __init__(self, config):
        self.device = torch.device(config["device"])
        self.act_dim = config["act_dim"]

        self.actor = MLP(config["obs_dim"], self.act_dim).to(self.device)
        self.critic1 = MLP(config["obs_dim"] + self.act_dim, 1).to(self.device)
        self.critic2 = MLP(config["obs_dim"] + self.act_dim, 1).to(self.device)

        self.buffer = ReplayBuffer()
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.optim_critic = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=1e-3
        )

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        action = self.actor(obs)
        return action.detach().cpu().numpy(), 0.0

    def store(self, transition):
        o, a, r, d, _ = transition
        self.buffer.add(o, a, r, o, d)

    def update(self, batch_size=256):
        if len(self.buffer.obs) < batch_size:
            return

        o, a, r, no, d = self.buffer.sample(batch_size)

        o = torch.FloatTensor(o).to(self.device)
        a = torch.FloatTensor(a).to(self.device)

        q1 = self.critic1(torch.cat([o, a], dim=-1))
        q2 = self.critic2(torch.cat([o, a], dim=-1))

        loss = (q1.mean() + q2.mean())

        self.optim_critic.zero_grad()
        loss.backward()
        self.optim_critic.step()