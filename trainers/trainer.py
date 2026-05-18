import gym
from agents.factory import build_agent


class Trainer:
    def __init__(self, config):
        self.env = gym.make(config["env_name"])

        config["obs_dim"] = self.env.observation_space.shape[0]
        config["act_dim"] = self.env.action_space.shape[0]

        self.agent = build_agent(config)

        self.config = config

    def train(self):
        obs = self.env.reset()[0]

        for step in range(self.config["total_steps"]):
            action, log_prob = self.agent.select_action(obs)

            next_obs, reward, done, _, _ = self.env.step(action)

            self.agent.store((obs, action, reward, done, log_prob))

            obs = next_obs

            if done:
                obs = self.env.reset()[0]

            if step % self.config["update_freq"] == 0:
                self.agent.update()
