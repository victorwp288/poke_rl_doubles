# PPO variant with KL regularization against a frozen reference policy.
import copy

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO
from stable_baselines3.common.utils import explained_variance


class KLRegularizedMaskablePPO(MaskablePPO):
    def __init__(
        self,
        policy,
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        tensorboard_log=None,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
        kl_coef_start=0.0,
        kl_coef_final=0.0,
        kl_anneal_steps=0,
    ):
        self.kl_coef_start = float(kl_coef_start)
        self.kl_coef_final = float(kl_coef_final)
        self.kl_anneal_steps = max(int(kl_anneal_steps), 0)
        self._kl_updates = 0
        self.kl_reference_policy = None
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def set_reference_policy(self, policy):
        if policy is None:
            self.kl_reference_policy = None
            return
        reference = copy.deepcopy(policy).to(self.device)
        reference.eval()
        for param in reference.parameters():
            param.requires_grad_(False)
        self.kl_reference_policy = reference

    def set_reference_from_current(self):
        self.set_reference_policy(self.policy)

    def _current_kl_coef(self):
        if self.kl_reference_policy is None:
            return 0.0
        if self.kl_anneal_steps <= 0:
            return self.kl_coef_final
        progress = min(self._kl_updates / float(self.kl_anneal_steps), 1.0)
        span = self.kl_coef_final - self.kl_coef_start
        return self.kl_coef_start + span * progress

    def _kl_penalty(self, rollout_data, log_prob):
        if self.kl_reference_policy is None:
            return torch.zeros(1, device=self.device)
        with torch.no_grad():
            _, ref_log_prob, _ = self.kl_reference_policy.evaluate_actions(
                rollout_data.observations,
                self._prepared_actions(rollout_data),
                action_masks=getattr(rollout_data, "action_masks", None),
            )
        ref_log_prob = ref_log_prob.view_as(log_prob)
        return (log_prob - ref_log_prob).mean()

    def _maybe_kl_term(self, rollout_data, log_prob, kl_coef):
        if kl_coef <= 0:
            return torch.zeros(1, device=self.device), None
        # KL penalty compares to frozen reference; coef is the pull-back strength.
        kl_term = self._kl_penalty(rollout_data, log_prob)
        return kl_term, kl_term.item()

    def _prepared_actions(self, rollout_data):
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            return actions.long().flatten()
        if actions.dtype != torch.long:
            return actions.long()
        return actions

    def _normalized_advantages(self, advantages):
        if not self.normalize_advantage:
            return advantages
        mean = advantages.mean()
        std = advantages.std()
        return (advantages - mean) / (std + 1e-8)

    def _value_prediction(self, values, old_values, clip_range_vf):
        if clip_range_vf is None:
            return values
        delta = values - old_values
        clipped = torch.clamp(delta, -clip_range_vf, clip_range_vf)
        return old_values + clipped

    def _log_mean(self, name, values):
        self.logger.record(name, float(np.mean(values)) if values else 0.0)

    def train(self):
        no_reference = self.kl_reference_policy is None
        no_start = self.kl_coef_start == 0.0
        no_final = self.kl_coef_final == 0.0
        if no_reference and no_start and no_final:
            return super().train()

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (
            self.clip_range_vf(self._current_progress_remaining)
            if self.clip_range_vf is not None
            else None
        )
        ent_coef_value = (
            self.ent_coef(self._current_progress_remaining)
            if callable(self.ent_coef)
            else self.ent_coef
        )

        stats = {
            "entropy": [],
            "policy": [],
            "value": [],
            "clip": [],
            "kl": [],
        }

        continue_training = True
        loss = torch.zeros(1, device=self.device)
        approx_kl_divs = []

        for epoch in range(self.n_epochs):
            if not continue_training:
                break
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = self._prepared_actions(rollout_data)
                action_masks = getattr(rollout_data, "action_masks", None)
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=action_masks,
                )

                values = values.flatten()
                advantages = self._normalized_advantages(rollout_data.advantages)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_term = advantages * ratio
                clipped_term = advantages * clipped_ratio
                policy_loss = -torch.min(policy_term, clipped_term).mean()
                stats["policy"].append(policy_loss.item())

                clip_fraction = (torch.abs(ratio - 1) > clip_range).float().mean().item()
                stats["clip"].append(clip_fraction)

                values_pred = self._value_prediction(values, rollout_data.old_values, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                stats["value"].append(value_loss.item())

                if entropy is None:
                    entropy_loss = torch.zeros(1, device=self.device)
                else:
                    entropy_loss = -torch.mean(entropy)
                stats["entropy"].append(entropy_loss.item())

                kl_coef = self._current_kl_coef()
                kl_term, kl_value = self._maybe_kl_term(rollout_data, log_prob, kl_coef)
                if kl_value is not None:
                    stats["kl"].append(kl_value)

                loss = (
                    policy_loss
                    + ent_coef_value * entropy_loss
                    + self.vf_coef * value_loss
                    + kl_coef * kl_term
                )

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            if not continue_training:
                break

        self._kl_updates += 1
        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        self._log_mean("train/entropy_loss", stats["entropy"])
        self._log_mean("train/policy_gradient_loss", stats["policy"])
        self._log_mean("train/value_loss", stats["value"])
        self._log_mean("train/approx_kl", approx_kl_divs)
        self._log_mean("train/clip_fraction", stats["clip"])
        if stats["kl"]:
            self._log_mean("train/kl_penalty", stats["kl"])
        self.logger.record("train/loss", float(loss.item()))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
