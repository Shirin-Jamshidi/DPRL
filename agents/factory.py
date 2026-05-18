from agents.ppo import PPOAgent
from agents.td3 import TD3Agent


def build_agent(config):
    if config["algo"] == "ppo":
        return PPOAgent(config)
    elif config["algo"] == "td3":
        return TD3Agent(config)
    else:
        raise ValueError("Unknown algorithm")