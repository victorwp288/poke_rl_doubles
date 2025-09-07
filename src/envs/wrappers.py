import numpy as np

FEATURE_SIZE = 128  # placeholder


def encode_state_and_mask(battle) -> tuple[np.ndarray, np.ndarray]:
    obs = np.zeros((FEATURE_SIZE,), dtype=np.float32)
    mask = np.ones((16,), dtype=np.float32)  # replace with real legality mask
    return obs, mask
