import numpy as np
import torch
import torch.optim as optim
from agents.base import RLAgent
from models.actor_critic import Actor, Critic
from buffers.rollout_buffer import RolloutBuffer


class PPOAgent(RLAgent):
    def __init__(self, config):
        self.device = torch.device(config["device"])

        self.actor = Actor(config["obs_dim"], config["act_dim"]).to(self.device)
        self.critic = Critic(config["obs_dim"]).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config["lr"]
        )

        self.buffer = RolloutBuffer()

        self.gamma = 0.99
        self.clip_eps = 0.2

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        dist = self.actor(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        
        return (
            action.detach().cpu().numpy(),
            log_prob.detach().cpu().item()  # ✅ scalar float
        )


    def store(self, transition):
        self.buffer.add(*transition)

    def update(self):
        obs, actions, rewards, dones, old_log_probs = self.buffer.get()

        obs = torch.FloatTensor(np.array(obs)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        old_log_probs = old_log_probs.view(-1)
        returns = returns.view(-1)

        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        for _ in range(10):
            dist = self.actor(obs)
            new_log_probs = dist.log_prob(actions).sum(-1)

            ratio = torch.exp(new_log_probs - old_log_probs)

            advantages = returns - self.critic(obs).squeeze().detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - self.critic(obs).squeeze()).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.buffer.clear()