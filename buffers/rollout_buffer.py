class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, obs, action, reward, done, log_prob):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def get(self):
        return (
            self.obs,
            self.actions,
            self.rewards,
            self.dones,
            self.log_probs
        )

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []