import gymnasium as gym
from agents.factory import build_agent


class Trainer:
    def __init__(self, config):
        self.env = gym.make(config["env_name"])

        config["obs_dim"] = self.env.observation_space.shape[0]
        config["act_dim"] = self.env.action_space.shape[0]

        self.agent = build_agent(config)

        self.config = config

    def train(self):
        obs, _ = self.env.reset()

        episode_reward = 0
        episode = 0

        for step in range(self.config["total_steps"]):
            action, log_prob = self.agent.select_action(obs)

            action = action.clip(
                self.env.action_space.low,
                self.env.action_space.high
            )

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.agent.store((obs, action, reward, done, log_prob))

            obs = next_obs
            episode_reward += reward

            if done:
                print(f"Episode {episode} | Reward: {episode_reward:.2f}")
                obs, _ = self.env.reset()
                episode_reward = 0
                episode += 1

            if step % self.config["update_freq"] == 0:
                self.agent.update()

            if step % 5000 == 0:
                print(f"Step: {step}")