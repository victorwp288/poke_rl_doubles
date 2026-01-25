from poke_env.player.baselines import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from src.utils.teambuilders import (
    RotatingTeambuilder,
    constant_team_from_text,
    load_showdown_teams_from_dir,
)

from .teachers import PolicyTeacherPlayer, RecordingHeuristics


def _make_player(kind, *, record=False, **kwargs):
    kwargs = dict(kwargs)
    model_path = kwargs.pop("model_path", None)
    kind = kind.lower()
    if kind in {"simple", "heuristic", "simpleheuristics"}:
        if record:
            recorder = kwargs.pop("recorder")
            act_size = kwargs.pop("act_size")
            return RecordingHeuristics(
                recorder=recorder,
                act_size=act_size,
                teacher_name="SimpleHeuristicsPlayer",
                **kwargs,
            )
        return SimpleHeuristicsPlayer(**kwargs)
    if kind in {"maxbp", "maxbasepower"}:
        return MaxBasePowerPlayer(**kwargs)
    if kind == "random":
        return RandomPlayer(**kwargs)
    if kind == "policy":
        if model_path is None:
            raise ValueError("policy teacher_kind requires model_path (teacher_path)")
        recorder = kwargs.pop("recorder")
        act_size = kwargs.pop("act_size")
        return PolicyTeacherPlayer(
            recorder=recorder, act_size=act_size, model_path=model_path, **kwargs
        )
    raise ValueError(f"unknown player kind: {kind}")


def _opponent_pool(settings, our_team_text):
    pool = [
        text
        for text in load_showdown_teams_from_dir(settings.opponent_teams_dir)
        if text.strip() and text.strip() != our_team_text.strip()
    ]
    return pool or [our_team_text]


def _build_teacher(*, settings, recorder, act_size, server_cfg, our_team_text):
    return _make_player(
        settings.teacher_kind,
        record=True,
        recorder=recorder,
        act_size=act_size,
        model_path=settings.teacher_path,
        battle_format=settings.battle_format,
        team=constant_team_from_text(our_team_text),
        server_configuration=server_cfg,
        max_concurrent_battles=settings.max_concurrent_battles,
    )


def _build_opponent(*, kind, settings, opponent_pool, server_cfg):
    return _make_player(
        kind,
        battle_format=settings.battle_format,
        team=RotatingTeambuilder(opponent_pool),
        server_configuration=server_cfg,
        max_concurrent_battles=settings.opponent_concurrency,
    )


__all__ = ["_build_opponent", "_build_teacher", "_make_player", "_opponent_pool"]
