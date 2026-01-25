from src.config import section

_METRIC_KEYS = {
    "faint": "faint_diff",
    "team_hp": "team_hp_advantage",
    "active_hp": "active_hp_advantage",
    "status": "status_advantage",
    "side_condition": "side_condition_advantage",
}


def _default_rewards():
    rewards = {
        "win": 1.0,
        "loss": -1.0,
        "draw": -0.1,
        "faint": 0.1,
        "team_hp": 0.05,
        "active_hp": 0.1,
        "status": 0.05,
        "side_condition": 0.05,
    }
    config = section("online")
    base = config.get("base_rewards", {})
    if isinstance(base, dict):
        for key, value in base.items():
            if key in rewards and value is not None:
                rewards[key] = float(value)
    return rewards


def _reward_stats(rewards):
    return {
        "win_reward": rewards["win"],
        "loss_penalty": rewards["loss"],
        "draw_penalty": rewards["draw"],
        "faint_reward": rewards["faint"],
        "team_hp_reward": rewards["team_hp"],
        "active_hp_reward": rewards["active_hp"],
        "status_reward": rewards["status"],
        "side_condition_reward": rewards["side_condition"],
    }


def _team_hp_fraction(team):
    total = 0.0
    count = 0
    for mon in team:
        if mon is None:
            continue
        max_hp = getattr(mon, "max_hp", None)
        current_hp = getattr(mon, "current_hp", None)
        if max_hp:
            numer = current_hp if current_hp is not None else max_hp
            total += float(numer) / float(max_hp)
            count += 1
            continue
        fraction = getattr(mon, "current_hp_fraction", None)
        if fraction is not None:
            total += float(fraction)
            count += 1
            continue
        total += 1.0
        count += 1
    return total / max(1.0, float(count))


def _active_hp_fraction(slots):
    total = 0.0
    count = 0
    for mon in slots:
        if mon is None:
            continue
        max_hp = getattr(mon, "max_hp", None)
        current_hp = getattr(mon, "current_hp", None)
        if max_hp:
            numer = current_hp if current_hp is not None else max_hp
            total += float(numer) / float(max_hp)
            count += 1
            continue
        fraction = getattr(mon, "current_hp_fraction", None)
        if fraction is not None:
            total += float(fraction)
            count += 1
            continue
        total += 1.0
        count += 1
    return total / max(1.0, float(count))


def _status_count(team):
    count = 0
    for mon in team:
        if mon is None:
            continue
        if getattr(mon, "status", None):
            count += 1
    return count


def _reward_metrics(battle, knocked_out_me, knocked_out_opp):
    return {
        "faint_diff": float(knocked_out_opp - knocked_out_me),
        "team_hp_advantage": _team_hp_fraction(battle.team.values())
        - _team_hp_fraction(battle.opponent_team.values()),
        "active_hp_advantage": _active_hp_fraction(battle.active_pokemon)
        - _active_hp_fraction(battle.opponent_active_pokemon),
        "status_advantage": float(
            _status_count(battle.opponent_team.values()) - _status_count(battle.team.values())
        ),
        "side_condition_advantage": float(
            len(getattr(battle, "opponent_side_conditions", {}))
            - len(getattr(battle, "side_conditions", {}))
        ),
    }


def _score_from_metrics(rewards, metrics):
    total = 0.0
    for reward_key, metric_key in _METRIC_KEYS.items():
        total += rewards[reward_key] * metrics[metric_key]
    return total


__all__ = [
    "_active_hp_fraction",
    "_default_rewards",
    "_reward_metrics",
    "_reward_stats",
    "_score_from_metrics",
    "_status_count",
    "_team_hp_fraction",
]
