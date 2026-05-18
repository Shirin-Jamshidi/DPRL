class RLAgent:
    def select_action(self, obs):
        raise NotImplementedError

    def store(self, *args):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError