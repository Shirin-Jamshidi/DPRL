import numpy as np


class ReplayBuffer:
    def __init__(self, size=1_000_000):
        self.size = size
        self.ptr = 0
        self.full = False

        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.dones = []

    def add(self, o, a, r, no, d):
        if len(self.obs) < self.size:
            self.obs.append(o)
            self.actions.append(a)
            self.rewards.append(r)
            self.next_obs.append(no)
            self.dones.append(d)
        else:
            self.obs[self.ptr] = o
            self.actions[self.ptr] = a
            self.rewards[self.ptr] = r
            self.next_obs[self.ptr] = no
            self.dones[self.ptr] = d
            self.full = True

        self.ptr = (self.ptr + 1) % self.size

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self.obs), size=batch_size)
        return (
            np.array([self.obs[i] for i in idxs]),
            np.array([self.actions[i] for i in idxs]),
            np.array([self.rewards[i] for i in idxs]),
            np.array([self.next_obs[i] for i in idxs]),
            np.array([self.dones[i] for i in idxs]),
        )