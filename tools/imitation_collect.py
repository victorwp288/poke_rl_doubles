#!/usr/bin/env python3
# Collect imitation tuples from heuristic self-play in Gen 9 Doubles.
import argparse
import asyncio
import contextlib
import json
import random
import sys
import uuid
from pathlib import Path

from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
    ShowdownServerConfiguration,
)
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.baselines import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import section  # noqa: E402
from src.core.env import action_space_size  # noqa: E402
from src.core.features import encode_observation, slot_action_mask  # noqa: E402
from src.utils.teambuilders import (  # noqa: E402
    RotatingTeambuilder,
    constant_team_from_text,
    load_showdown_teams_from_dir,
    read_showdown_team,
)


def _resolve_server_configuration(url):
    token = url.strip()
    lowered = token.lower()
    if lowered in {"showdown", "https://play.pokemonshowdown.com"}:
        return ShowdownServerConfiguration
    if lowered in {"local", "localhost", "http://localhost:8000"}:
        return LocalhostServerConfiguration

    websocket_url = token
    if websocket_url.startswith("http://"):
        websocket_url = "ws://" + websocket_url[len("http://") :]
    elif websocket_url.startswith("https://"):
        websocket_url = "wss://" + websocket_url[len("https://") :]
    elif not websocket_url.startswith(("ws://", "wss://")):
        websocket_url = f"ws://{websocket_url}"

    if not websocket_url.endswith("/websocket"):
        websocket_url = websocket_url.rstrip("/") + "/websocket"

    auth_url = ShowdownServerConfiguration.authentication_url
    return ServerConfiguration(websocket_url, auth_url)


def _probe_action_space_size(battle_format, server_cfg):
    env = None
    try:
        account = AccountConfiguration(f"Recorder{uuid.uuid4().hex[:6]}", None)
        env = DoublesEnv(
            account_configuration1=account,
            battle_format=battle_format,
            server_configuration=server_cfg,
            start_listening=False,
            fake=True,
        )
        agents = getattr(env, "possible_agents", None) or []
        agent = agents[0] if agents else getattr(env, "agent", None)
        if agent is None:
            return None
        space = env.action_space(agent)
        nvec = getattr(space, "nvec", None)
        if nvec is None or len(nvec) == 0:
            return None
        return int(nvec[0])
    except Exception as exc:
        print(f"[warn] failed to probe action space size via DoublesEnv: {exc}")
        return None
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()


def _resolve_action_space_size(settings, server_cfg):
    inferred = _probe_action_space_size(settings.battle_format, server_cfg)
    return inferred or action_space_size(settings.battle_format)


def _action_to_tuple(order, battle):
    raw = DoublesEnv.order_to_action(order, battle, fake=True, strict=False)
    first, second = (int(raw[0]), int(raw[1]))
    return first, second


class Recorder:
    def __init__(self, out_path, *, rotate_every=None):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._base_path = out_path
        self._path = out_path
        self._handle = out_path.open("a", encoding="utf-8")
        self._written = 0
        self._rotate_every = rotate_every
        self._since_rotate = 0

    @property
    def written(self):
        return self._written

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def write(self, payload):
        self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._handle.flush()
        self._written += 1
        self._since_rotate += 1
        if self._rotate_every and self._since_rotate >= self._rotate_every:
            self.rotate()

    def close(self):
        with contextlib.suppress(Exception):
            self._handle.close()

    def rotate(self):
        self.close()
        suffix = self._base_path.suffix or ".jsonl"
        stem_path = self._base_path.with_suffix("")
        rotated = stem_path.with_name(f"{stem_path.name}.{self._written:07d}{suffix}")
        with contextlib.suppress(FileNotFoundError):
            self._base_path.rename(rotated)
        self._handle = self._base_path.open("a", encoding="utf-8")
        self._since_rotate = 0


class RecordingHeuristics(SimpleHeuristicsPlayer):
    def __init__(self, recorder, act_size, teacher_name, **kwargs):
        self._recorder = recorder
        self._act_size = act_size
        self._teacher_name = teacher_name
        super().__init__(**kwargs)

    def choose_move(self, battle):
        order = super().choose_move(battle)
        mask0 = slot_action_mask(battle, 0, self._act_size)
        mask1 = slot_action_mask(battle, 1, self._act_size)
        first, second = _action_to_tuple(order, battle)
        record = {
            "battle_tag": battle.battle_tag,
            "turn": battle.turn,
            "teacher": self._teacher_name,
            "format": battle.format,
            "observation": encode_observation(battle),
            "action": [first, second],
            "mask": [mask0, mask1],
        }
        self._recorder.write(record)
        return order


def _make_player(kind, *, record=False, **kwargs):
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
    raise ValueError(f"unknown player kind: {kind}")


class Settings:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
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


async def play_dataset(settings):
    random.seed(settings.seed)
    server_cfg = _resolve_server_configuration(settings.server_url)
    act_size = _resolve_action_space_size(settings, server_cfg)

    our_team_text = read_showdown_team(settings.our_team_path)
    opponent_pool = _opponent_pool(settings, our_team_text)

    def _cleanup(player):
        with contextlib.suppress(Exception):
            player.reset_battles()
        with contextlib.suppress(Exception):
            close_fn = getattr(player, "close", None)
            if callable(close_fn):
                close_fn()

    with Recorder(settings.out_path, rotate_every=settings.rotate_every) as recorder:
        teacher = _build_teacher(
            settings=settings,
            recorder=recorder,
            act_size=act_size,
            server_cfg=server_cfg,
            our_team_text=our_team_text,
        )

        stats = {"wins": 0, "losses": 0, "draws": 0}
        for battle_idx in range(settings.n_battles):
            opponent_kind = random.choice(settings.opponents_kinds)
            opponent = _build_opponent(
                kind=opponent_kind,
                settings=settings,
                opponent_pool=opponent_pool,
                server_cfg=server_cfg,
            )
            print(
                f"[info] starting battle {battle_idx + 1}/{settings.n_battles} vs {opponent_kind}",
                flush=True,
            )
            try:
                wins_before = teacher.n_won_battles
                losses_before = teacher.n_lost_battles
                await asyncio.wait_for(
                    teacher.battle_against(opponent, n_battles=1),
                    timeout=settings.battle_timeout,
                )
                outcome = "draw"
                if teacher.n_won_battles > wins_before:
                    stats["wins"] += 1
                    outcome = "win"
                elif teacher.n_lost_battles > losses_before:
                    stats["losses"] += 1
                    outcome = "loss"
                else:
                    stats["draws"] += 1
                print(
                    f"[info] finished battle {battle_idx + 1}/{settings.n_battles} -> {outcome}",
                    flush=True,
                )
            except TimeoutError:
                print(f"[warn] battle {battle_idx + 1}: timeout (likely rejected team)")
            finally:
                _cleanup(opponent)
        print(
            f"[info] teacher results -> wins={stats['wins']} losses={stats['losses']} draws={stats['draws']}",
            flush=True,
        )
        _cleanup(teacher)

    print(
        f"Collected {settings.n_battles} battles at {settings.out_path}. "
        f"Teacher={settings.teacher_kind} format={settings.battle_format} act_size={act_size}"
    )


def load_settings():
    config = section("imitation_collect")
    settings = Settings(
        n_battles=int(config.get("n_battles", 50)),
        server_url=config.get("server_url", "http://localhost:8000"),
        battle_format=config.get("battle_format", "gen9doublesou"),
        our_team_path=Path(config.get("our_team_path", "teams/gen9dou_fixed.txt")),
        opponent_teams_dir=Path(config.get("opponent_teams_dir", "teams")),
        teacher_kind=config.get("teacher_kind", "simple"),
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


def collect_imitation(settings=None):
    settings = settings or load_settings()
    if settings.rotate_every is not None and settings.rotate_every <= 0:
        raise ValueError("rotate_every must be positive when provided")
    asyncio.run(play_dataset(settings))


def main():
    # collect_imitation(load_settings())
    parser = argparse.ArgumentParser(
        description="Collect imitation tuples from heuristic self-play in Gen 9 Doubles."
    )

    parser.add_argument("--n-battles", type=int)
    parser.add_argument("--server-url", type=str)
    parser.add_argument("--battle-format", type=str)
    parser.add_argument("--our-team-path", type=str)
    parser.add_argument("--opponent-teams-dir", type=str)
    parser.add_argument("--teacher-kind", type=str)
    parser.add_argument("--opponents", nargs="*")
    parser.add_argument("--out-path", type=str)
    parser.add_argument("--battle-timeout", type=float)
    parser.add_argument("--rotate-every", type=int)
    parser.add_argument("--max-concurrent-battles", type=int)
    parser.add_argument("--opponent-concurrency", type=int)
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    base = load_settings()

    if args.n_battles is not None:
        base.n_battles = args.n_battles
    if args.server_url is not None:
        base.server_url = args.server_url
    if args.battle_format is not None:
        base.battle_format = args.battle_format
    if args.our_team_path is not None:
        base.our_team_path = Path(args.our_team_path)
    if args.opponent_teams_dir is not None:
        base.opponent_teams_dir = Path(args.opponent_teams_dir)
    if args.teacher_kind is not None:
        base.teacher_kind = args.teacher_kind
    if args.opponents is not None:
        base.opponents_kinds = args.opponents
    if args.out_path is not None:
        base.out_path = Path(args.out_path)
    if args.battle_timeout is not None:
        base.battle_timeout = args.battle_timeout
    if args.rotate_every is not None:
        base.rotate_every = args.rotate_every
    if args.max_concurrent_battles is not None:
        base.max_concurrent_battles = args.max_concurrent_battles
    if args.opponent_concurrency is not None:
        base.opponent_concurrency = args.opponent_concurrency
    if args.seed is not None:
        base.seed = args.seed

    collect_imitation(base)


if __name__ == "__main__":
    main()


__all__ = ["Settings", "collect_imitation", "play_dataset", "load_settings"]
