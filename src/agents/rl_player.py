from poke_env.player import Player

from src.envs.wrappers import encode_state_and_mask


class RLPlayer(Player):
    def __init__(self, policy=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def choose_move(self, battle):
        obs, mask = encode_state_and_mask(battle)
        return self.choose_random_move(battle)
