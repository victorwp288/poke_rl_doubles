# Sanitize/repair pipeline that turns raw actions into legal orders.
import numpy as np
from poke_env.player.battle_order import DefaultBattleOrder, DoubleBattleOrder, SingleBattleOrder


class MaskableEnvRepairMixin:
    def _order_is_default(self, order):
        if isinstance(order, DefaultBattleOrder):
            return True
        if isinstance(order, str):
            return order.strip() == "/choose default"
        command = getattr(order, "order", None)
        if isinstance(command, str):
            return command.strip() == "/choose default"
        return False

    def _maps_to_default(self, vector, strict=False):
        battle = self._battle_state()
        if battle is None:
            return False
        order = self._action_to_order(vector, battle, strict)
        return self._order_is_default(order) if order is not None else False

    def _is_default_action(self, vector):
        battle = self._battle_state()
        if battle is None:
            return False
        order = self._action_to_order(vector, battle, True)
        return self._order_is_default(order) if order is not None else False

    def _joint_candidates(self, slot_masks):
        battle = self._battle_state()
        if battle is None:
            return []
        orders_a = self._legal_orders_safe(battle, 0)
        orders_b = self._legal_orders_safe(battle, 1)
        try:
            joined = DoubleBattleOrder.join_orders(orders_a, orders_b)
        except Exception:
            joined = []
        masked_candidates = []
        unmasked_candidates = []
        for order in joined:
            vector = self._order_to_vector(battle, order)
            if vector is None or not self._within_action_space(vector):
                continue
            if self._mask_allows_action(vector, slot_masks):
                masked_candidates.append(vector)
            else:
                unmasked_candidates.append(vector)
        if masked_candidates:
            return masked_candidates
        if unmasked_candidates:
            return unmasked_candidates
        forced_vectors = self._forced_switch_vectors(battle, slot_masks=slot_masks)
        if forced_vectors:
            return forced_vectors
        nvec = self._action_nvec()
        if nvec:
            zeros = np.zeros(len(nvec), dtype=int)
            if self._mask_allows_action(zeros, slot_masks):
                return [zeros]
        return []

    def _forced_switch_vectors(self, battle, slot_masks=None):
        force_flags = getattr(battle, "force_switch", None)
        if not isinstance(force_flags, list | tuple):
            return []
        if not any(force_flags):
            return []
        slot_orders = []
        for idx in range(2):
            forced = bool(force_flags[idx]) if idx < len(force_flags) else False
            if forced:
                switches = (
                    battle.available_switches[idx]
                    if idx < len(getattr(battle, "available_switches", []))
                    else []
                )
                orders = [SingleBattleOrder(switch) for switch in switches or []]
                if not orders:
                    return []
            else:
                orders = self._legal_orders_safe(battle, idx)
                if not orders:
                    return []
            slot_orders.append(orders)
        while len(slot_orders) < 2:
            slot_orders.append(self._legal_orders_safe(battle, len(slot_orders)))
        if len(slot_orders) < 2 or not slot_orders[0] or not slot_orders[1]:
            return []
        try:
            joined = DoubleBattleOrder.join_orders(slot_orders[0], slot_orders[1])
        except Exception:
            joined = []
        vectors = []
        for order in joined:
            vector = self._order_to_vector(battle, order)
            if vector is None or not self._within_action_space(vector):
                continue
            if slot_masks is not None and not self._mask_allows_action(vector, slot_masks):
                continue
            if self._maps_to_default(vector, strict=True):
                continue
            vectors.append(vector)
        return vectors

    def _repair_action(self, action, slot_masks):
        battle = self._battle_state()
        space = self._action_space()
        if battle is None or space is None:
            return None
        nvec = self._action_nvec()
        vector = self._action_vector(action)
        if vector.size < len(nvec):
            vector = np.pad(vector, (0, len(nvec) - vector.size), constant_values=0)
        vector = vector[: len(nvec)]
        for idx, limit in enumerate(nvec):
            if limit <= 0:
                vector[idx] = 0
            else:
                vector[idx] = max(0, min(vector[idx], limit - 1))
        sanitized_default = False
        if self._mask_allows_action(vector, slot_masks):
            order = self._action_to_order(vector, battle, True)
            if order is not None:
                sanitized_default = self._order_is_default(order)
                if not sanitized_default:
                    return self._reshape_like_space(vector)
                for forced in self._forced_switch_vectors(battle, slot_masks=slot_masks):
                    if self._maps_to_default(forced, strict=True):
                        continue
                    return self._reshape_like_space(forced)
        joint_candidates = list(self._joint_candidates(slot_masks))
        for candidate in joint_candidates:
            order = self._action_to_order(candidate, battle, True)
            if order is None:
                continue
            if not self._order_is_default(order):
                return self._reshape_like_space(candidate)
        for forced in self._forced_switch_vectors(battle, slot_masks=slot_masks):
            if self._maps_to_default(forced, strict=True):
                continue
            return self._reshape_like_space(forced)
        self._log_repair_failure(vector, slot_masks, joint_candidates)
        if sanitized_default and self._mask_allows_action(vector, slot_masks):
            return self._reshape_like_space(vector)
        return None

    def _default_fallback(self, slot_masks):
        space = self._action_space()
        if space is None:
            return None
        nvec = self._action_nvec()
        if not nvec:
            return None
        zeros = np.zeros(len(nvec), dtype=int)
        if self._mask_allows_action(zeros, slot_masks):
            return self._reshape_like_space(zeros)
        return None

    def _safe_action_fallback(self, slot_masks, candidate_vectors=None):
        space = self._action_space()
        if space is None:
            return None
        vectors = candidate_vectors
        if vectors is None:
            vectors = list(self._joint_candidates(slot_masks))
        for candidate in vectors:
            if self._maps_to_default(candidate, strict=True):
                continue
            return self._reshape_like_space(candidate)
        fallback = self._default_fallback(slot_masks)
        if fallback is not None and not self._maps_to_default(fallback, strict=True):
            return fallback
        return None

    def _candidate_queue(self, action):
        # Sanitize clamps raw action into space/mask; repair searches a legal non-default alternative.
        # Info flags distinguish sanitized_action vs repaired_action downstream.
        sanitized, sanitized_changed = self._sanitize_action(action)
        slot_masks = self._slot_masks()
        queue = []
        sanitized_is_default = self._is_default_action(sanitized)
        default_candidate = None
        if sanitized_is_default:
            default_candidate = (sanitized, True, False)
        else:
            queue.append((sanitized, bool(sanitized_changed), False))
        repaired = self._repair_action(sanitized, slot_masks)
        if (
            repaired is not None
            and not self._same_action_vector(repaired, sanitized)
            and not self._maps_to_default(repaired, strict=True)
        ):
            queue.append((repaired, True, True))
        fallback = self._default_fallback(slot_masks)
        if fallback is not None and all(
            not self._same_action_vector(fallback, existing[0]) for existing in queue
        ):
            if not self._maps_to_default(fallback, strict=True):
                queue.append((fallback, True, True))
            elif default_candidate is None:
                default_candidate = (fallback, True, True)
        if default_candidate is not None:
            queue.append(default_candidate)
        return queue

    def _first_non_default_candidate(self):
        if self._action_space() is None:
            return None
        slot_masks = self._slot_masks()
        for vector in self._joint_candidates(slot_masks):
            if not self._maps_to_default(vector, strict=True):
                return self._reshape_like_space(vector)
        return None
