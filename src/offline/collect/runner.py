import asyncio
import contextlib
import random

from src.utils.teambuilders import read_showdown_team

from .config import load_settings
from .players import _build_opponent, _build_teacher, _opponent_pool
from .recording import Recorder, _battle_summary
from .utils import _resolve_action_space_size, _resolve_server_configuration


async def play_dataset(settings):
    random.seed(settings.seed)
    server_cfg = _resolve_server_configuration(settings.server_url)
    act_size = _resolve_action_space_size(settings, server_cfg)

    our_team_text = read_showdown_team(settings.our_team_path)
    opponent_pool = _opponent_pool(settings, our_team_text)
    seen_tags: set[str] = set()

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
            outcome = None
            timeout = False
            err_msg = None
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
                timeout = True
                err_msg = "timeout"
                print(f"[warn] battle {battle_idx + 1}: timeout (likely rejected team)")
            finally:
                new_tags = [tag for tag in teacher.battles if tag not in seen_tags]
                for tag in new_tags:
                    battle_obj = teacher.battles.get(tag)
                    if battle_obj is None:
                        continue
                    summary = _battle_summary(
                        battle_obj,
                        outcome=outcome,
                        opponent_kind=opponent_kind,
                        settings=settings,
                        timeout=timeout,
                        error=err_msg,
                    )
                    recorder.write(summary)
                    seen_tags.add(tag)
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


def collect_imitation(settings=None):
    settings = settings or load_settings()
    if settings.rotate_every is not None and settings.rotate_every <= 0:
        raise ValueError("rotate_every must be positive when provided")
    asyncio.run(play_dataset(settings))


__all__ = ["collect_imitation", "play_dataset"]
