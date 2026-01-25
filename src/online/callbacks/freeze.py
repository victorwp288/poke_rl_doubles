from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class FreezeSharedCallback(BaseCallback):
    """
    Freeze policy/value trunks for an initial step window, then unfreeze.
    Useful for warmstarts to keep the policy close to BC early in training.
    """

    def __init__(self, freeze_steps: int):
        super().__init__()
        self.freeze_steps = int(freeze_steps)
        self._frozen = False
        self._unfrozen = False

    def _set_requires_grad(self, flag: bool):
        policy: Any = self.model.policy
        modules = [
            policy.features_extractor,
            policy.mlp_extractor.policy_net,
            policy.mlp_extractor.value_net,
        ]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = flag

    def _freeze(self):
        if self._frozen or self.freeze_steps <= 0:
            return
        self._set_requires_grad(False)
        self._frozen = True
        print(f"[freeze] frozen trunks for first {self.freeze_steps} steps", flush=True)

    def _unfreeze(self):
        if self._unfrozen:
            return
        self._set_requires_grad(True)
        self._unfrozen = True
        print("[freeze] trunks unfrozen", flush=True)

    def _on_training_start(self) -> None:
        self._freeze()

    def _on_step(self) -> bool:
        if self.freeze_steps <= 0 or self._unfrozen:
            return True
        if self.num_timesteps >= self.freeze_steps:
            self._unfreeze()
        return True


__all__ = ["FreezeSharedCallback"]
