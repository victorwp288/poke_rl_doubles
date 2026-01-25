def _drop_sample(payload, filters):
    if not filters:
        return False
    if filters.get("drop_timeouts") and payload.get("timeout"):
        return True
    if filters.get("drop_errors") and payload.get("error"):
        return True
    stats = payload.get("stats") if isinstance(payload, dict) else {}
    reward = payload.get("reward")
    if reward is None and isinstance(stats, dict):
        reward = stats.get("reward")
    if reward is not None:
        min_reward = filters.get("min_reward")
        if min_reward is not None and reward < min_reward:
            return True
    turn = None
    if isinstance(stats, dict):
        turn = stats.get("turn")
    if turn is None:
        turn = payload.get("turn")
    if turn is not None:
        min_turns = filters.get("min_turns")
        if min_turns is not None and turn < min_turns:
            return True
    result = payload.get("result") or (stats.get("result") if isinstance(stats, dict) else None)
    bad_results = filters.get("drop_results") or []
    return result in bad_results


def _sample_weight(payload, cfg):
    if not cfg:
        return 1.0
    weight = 1.0
    stats = payload.get("stats") if isinstance(payload, dict) else {}
    result = payload.get("result") or (stats.get("result") if isinstance(stats, dict) else None)
    if result:
        weight *= float(cfg.get("by_result", {}).get(result, 1.0))
    opponent = payload.get("opponent")
    if opponent:
        weight *= float(cfg.get("by_opponent", {}).get(opponent, 1.0))
    reward = payload.get("reward")
    if reward is None and isinstance(stats, dict):
        reward = stats.get("reward")
    reward_scale = cfg.get("reward_scale")
    if reward is not None and reward_scale:
        weight += float(reward_scale) * float(reward)
    clamp_cfg = cfg.get("clamp") or {}
    w_min = clamp_cfg.get("min")
    w_max = clamp_cfg.get("max")
    if w_min is not None:
        weight = max(float(w_min), weight)
    if w_max is not None:
        weight = min(float(w_max), weight)
    return float(weight)


__all__ = ["_drop_sample", "_sample_weight"]
