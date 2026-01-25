from pathlib import Path

from src.config import section


class Settings:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not hasattr(self, "teacher_path"):
            self.teacher_path = None
        if not hasattr(self, "max_concurrent_battles"):
            self.max_concurrent_battles = 1
        if not hasattr(self, "opponent_concurrency"):
            self.opponent_concurrency = self.max_concurrent_battles
        if not hasattr(self, "seed"):
            self.seed = 0
        if hasattr(self, "opponents") and not hasattr(self, "opponents_kinds"):
            self.opponents_kinds = list(self.opponents)
        if not hasattr(self, "opponents_kinds"):
            self.opponents_kinds = ["simple", "maxbp", "random"]


def load_settings():
    config = section("imitation_collect")
    settings = Settings(
        n_battles=int(config.get("n_battles", 50)),
        server_url=config.get("server_url", "http://localhost:8000"),
        battle_format=config.get("battle_format", "gen9doublesou"),
        our_team_path=Path(config.get("our_team_path", "teams/gen9dou_fixed.txt")),
        opponent_teams_dir=Path(config.get("opponent_teams_dir", "teams")),
        teacher_kind=config.get("teacher_kind", "simple"),
        teacher_path=config.get("teacher_path"),
        opponents_kinds=list(config.get("opponents", ["simple", "maxbp", "random"])),
        out_path=Path(config.get("out_path", "data/processed/imitation.jsonl")),
        battle_timeout=float(config.get("battle_timeout", 60.0)),
        rotate_every=(int(config["rotate_every"]) if config.get("rotate_every") else None),
        max_concurrent_battles=int(config.get("max_concurrent_battles", 1)),
        opponent_concurrency=int(
            config.get("opponent_concurrency", config.get("max_concurrent_battles", 1))
        ),
        seed=int(config.get("seed", 0)),
    )
    return settings


__all__ = ["Settings", "load_settings"]
