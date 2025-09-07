from dataclasses import dataclass


@dataclass
class PPOConfig:
    total_timesteps: int = 1_000_000


class PPO:
    def __init__(self, policy, config: PPOConfig):
        self.policy = policy
        self.config = config

    def learn(self):
        print("PPO learn called. Replace with real training loop.")
