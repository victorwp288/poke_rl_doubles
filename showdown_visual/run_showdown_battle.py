#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from poke_env import AccountConfiguration
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.baselines import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from src.core.action_mask import slot_action_mask
from src.core.env import action_space_size
from src.core.observation import encode_observation
from src.offline.model import BehaviorCloningPolicy
from src.online.environment import _resolve_server_configuration
from src.online.opponents import _unique_username
from src.online.policy.load import load_maskable_policy
from src.utils.teambuilders import read_showdown_team


def _spectator_base(server_url: str | None) -> str:
    if not server_url:
        return "http://localhost:8000"
    token = server_url.strip()
    lowered = token.lower()
    if lowered in {"local", "localhost", "http://localhost:8000"}:
        return "http://localhost:8000"
    if token.startswith("ws://"):
        return "http://" + token[len("ws://") :].rstrip("/")
    if token.startswith("wss://"):
        return "https://" + token[len("wss://") :].rstrip("/")
    return token.rstrip("/")


def _load_bc_policy(checkpoint_path: Path, stats_path: Path | None):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        state_dict = payload.get("state_dict", payload)
        metadata = payload
    else:
        state_dict = payload
        metadata = {}

    config = metadata.get("config") or {}
    obs_dim = metadata.get("obs_dim")
    action_dim = metadata.get("action_dim")
    if obs_dim is None or action_dim is None:
        raise ValueError("BC checkpoint is missing obs_dim/action_dim metadata")

    model = BehaviorCloningPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=int(config.get("hidden_dim", 1536)),
        hidden_layers=int(config.get("hidden_layers", 6)),
        dropout=float(config.get("dropout", 0.2)),
        attn_heads=int(config.get("attn_heads", 8)),
        slot_mlp_layers=int(config.get("slot_mlp_layers", 2)),
        head_mlp_dim=int(config.get("head_mlp_dim", 512)),
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    normalization = metadata.get("normalization")
    if normalization is None and stats_path is not None and stats_path.exists():
        normalization = json.loads(stats_path.read_text(encoding="utf-8"))
    mean = None
    std = None
    if normalization:
        mean = np.asarray(normalization.get("mean", []), dtype=np.float32)
        std = np.asarray(normalization.get("std", []), dtype=np.float32)
        if mean.size == 0 or std.size == 0:
            mean = None
            std = None
        else:
            std[std <= 1e-6] = 1.0
    return model, mean, std


def _mask_logits(logits: np.ndarray, mask: Iterable[int]) -> np.ndarray:
    masked = logits.copy()
    mask_arr = np.asarray(list(mask), dtype=bool)
    if mask_arr.shape[0] != masked.shape[0]:
        raise ValueError("mask/logits dimension mismatch")
    masked[~mask_arr] = -1e9
    return masked


class _BattleLogger:
    def __init__(self, spectator_base: str, log_level: str = "off"):
        self._spectator_base = spectator_base.rstrip("/")
        self._log_level = log_level
        self._seen: set[str] = set()
        self._last_turn: dict[str, int] = {}

    def maybe_log(self, battle) -> None:
        tag = getattr(battle, "battle_tag", None)
        if not tag or tag in self._seen:
            return
        self._seen.add(tag)
        print(f"[spectate] {self._spectator_base}/{tag}", flush=True)

    def log_turn(
        self,
        battle,
        obs: np.ndarray | None,
        mask0: Iterable[int] | None,
        mask1: Iterable[int] | None,
        action: Iterable[int] | None,
        *,
        note: str | None = None,
    ) -> None:
        if self._log_level == "off":
            return
        tag = getattr(battle, "battle_tag", None)
        if not tag:
            return
        turn = int(getattr(battle, "turn", 0) or 0)
        last = self._last_turn.get(tag)
        if last is not None and turn == last:
            return
        self._last_turn[tag] = turn

        active = getattr(battle, "active_pokemon", []) or []
        active_desc = []
        for mon in list(active)[:2]:
            name = getattr(mon, "species", None) or getattr(mon, "name", None) or "?"
            hp = getattr(mon, "current_hp_fraction", None)
            if hp is None:
                active_desc.append(str(name))
            else:
                active_desc.append(f"{name}({hp:.2f})")
        active_text = ", ".join(active_desc) if active_desc else "unknown"

        action_text = None
        if action is not None:
            try:
                action_list = list(action)
                if len(action_list) >= 2:
                    action_text = f"[{action_list[0]}, {action_list[1]}]"
            except Exception:
                action_text = None

        mask_text = None
        if mask0 is not None and mask1 is not None:
            try:
                legal0 = int(np.sum(np.asarray(list(mask0), dtype=np.int8)))
                legal1 = int(np.sum(np.asarray(list(mask1), dtype=np.int8)))
                mask_text = f"legal=({legal0},{legal1})"
            except Exception:
                mask_text = None

        note_text = f" note={note}" if note else ""
        parts = [
            f"[turn] {tag} t={turn}",
            f"active={active_text}",
        ]
        if action_text:
            parts.append(f"action={action_text}")
        if mask_text:
            parts.append(mask_text)
        if self._log_level == "verbose" and obs is not None:
            try:
                parts.append(
                    f"obs(len={obs.size} mean={obs.mean():.3f} min={obs.min():.3f} max={obs.max():.3f})"
                )
            except Exception:
                parts.append(f"obs(len={obs.size})")
        line = " ".join(parts) + note_text
        print(line, flush=True)


class MaskablePPOPlayer(SimpleHeuristicsPlayer):
    def __init__(self, model_path: Path, act_size: int, logger: _BattleLogger, **kwargs):
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"policy checkpoint not found: {self._model_path}")
        self._model = load_maskable_policy(self._model_path, device="cpu")
        self._act_size = act_size
        self._logger = logger
        super().__init__(**kwargs)

    def choose_move(self, battle):
        self._logger.maybe_log(battle)
        mask0 = slot_action_mask(battle, 0, self._act_size)
        mask1 = slot_action_mask(battle, 1, self._act_size)
        concat_mask = np.concatenate([mask0, mask1]).astype(np.int8)
        obs = np.asarray(encode_observation(battle), dtype=np.float32)

        order = None
        try:
            action, _ = self._model.predict(obs, action_masks=concat_mask, deterministic=True)
            action = np.asarray(action, dtype=int)
            if action.shape[0] == 2:
                order = DoublesEnv.action_to_order(action, battle, fake=False, strict=False)
        except Exception as exc:
            print(f"[warn] policy predict failed: {exc}", flush=True)

        self._logger.log_turn(
            battle,
            obs,
            mask0,
            mask1,
            action if isinstance(action, np.ndarray) else None,
            note=None if order is not None else "fallback",
        )

        if order is None:
            order = super().choose_move(battle)
        return order


class BehaviorClonePlayer(SimpleHeuristicsPlayer):
    def __init__(
        self,
        model_path: Path,
        stats_path: Path | None,
        act_size: int,
        logger: _BattleLogger,
        **kwargs,
    ):
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"BC checkpoint not found: {self._model_path}")
        self._model, self._mean, self._std = _load_bc_policy(self._model_path, stats_path)
        self._act_size = act_size
        self._logger = logger
        self._warned_dim = False
        super().__init__(**kwargs)

    def choose_move(self, battle):
        self._logger.maybe_log(battle)
        obs = np.asarray(encode_observation(battle), dtype=np.float32)
        if self._mean is not None and self._std is not None:
            obs = (obs - self._mean) / self._std

        with torch.no_grad():
            logits = self._model(torch.from_numpy(obs).unsqueeze(0))
        logits = logits.squeeze(0).cpu().numpy()

        if logits.shape[0] != 2 or logits.shape[1] != self._act_size:
            if not self._warned_dim:
                self._warned_dim = True
                print(
                    "[warn] BC action_dim mismatch; falling back to heuristics for this battle",
                    flush=True,
                )
            self._logger.log_turn(battle, obs, None, None, None, note="fallback")
            return super().choose_move(battle)

        actions = []
        for slot in range(2):
            mask = slot_action_mask(battle, slot, self._act_size)
            try:
                masked_logits = _mask_logits(logits[slot], mask)
            except ValueError:
                self._logger.log_turn(battle, obs, None, None, None, note="fallback")
                return super().choose_move(battle)
            if np.all(masked_logits <= -1e8):
                self._logger.log_turn(battle, obs, None, None, None, note="fallback")
                return super().choose_move(battle)
            actions.append(int(np.argmax(masked_logits)))

        order = None
        try:
            order = DoublesEnv.action_to_order(np.asarray(actions), battle, fake=False, strict=False)
        except Exception:
            order = None
        if order is None:
            self._logger.log_turn(battle, obs, None, None, actions, note="fallback")
            order = super().choose_move(battle)
        else:
            mask0 = slot_action_mask(battle, 0, self._act_size)
            mask1 = slot_action_mask(battle, 1, self._act_size)
            self._logger.log_turn(battle, obs, mask0, mask1, actions)
        return order


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local Showdown battle and print a spectate URL."
    )
    parser.add_argument("--model", required=True, type=Path, help="Path to PPO .zip or BC .pt")
    parser.add_argument(
        "--model-type",
        choices=["auto", "ppo", "bc"],
        default="auto",
        help="Model type (default: auto by file extension).",
    )
    parser.add_argument("--stats", type=Path, default=None, help="BC stats JSON (optional)")
    parser.add_argument(
        "--battle-format",
        default="gen9doublesou",
        help="Showdown format (default: gen9doublesou)",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="Showdown server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--team",
        type=Path,
        default=ROOT / "teams" / "gen9dou_fixed.txt",
        help="Team file for the agent.",
    )
    parser.add_argument(
        "--opponent",
        choices=["simple", "maxbp", "random", "policy", "mirror"],
        default="simple",
        help="Opponent type (default: simple)",
    )
    parser.add_argument(
        "--opponent-model",
        type=Path,
        default=None,
        help="Opponent policy path (for opponent=policy).",
    )
    parser.add_argument(
        "--opponent-model-type",
        choices=["auto", "ppo", "bc"],
        default="auto",
        help="Opponent model type (default: auto).",
    )
    parser.add_argument(
        "--opponent-stats",
        type=Path,
        default=None,
        help="Opponent BC stats JSON (optional)",
    )
    parser.add_argument(
        "--opponent-team",
        type=Path,
        default=None,
        help="Team file for opponent (defaults to --team)",
    )
    parser.add_argument("--battles", type=int, default=1, help="Number of battles to run")
    parser.add_argument(
        "--log-level",
        choices=["off", "summary", "verbose"],
        default="off",
        help="Log what the agent sees each turn (default: off)",
    )
    parser.add_argument(
        "--player-name-prefix",
        default="RLAgent",
        help="Username prefix for the agent",
    )
    parser.add_argument(
        "--opponent-name-prefix",
        default="Opponent",
        help="Username prefix for the opponent",
    )
    return parser.parse_args()


def _infer_model_type(path: Path, override: str) -> str:
    if override != "auto":
        return override
    if path.suffix == ".zip":
        return "ppo"
    if path.suffix in {".pt", ".pth"}:
        return "bc"
    raise ValueError("Could not infer model type; use --model-type")


def _build_policy_player(
    *,
    model_path: Path,
    model_type: str,
    stats_path: Path | None,
    act_size: int,
    battle_format: str,
    server_cfg,
    team_text: str | None,
    name_prefix: str,
    logger: _BattleLogger,
):
    account = AccountConfiguration(_unique_username(name_prefix), None)
    if model_type == "ppo":
        return MaskablePPOPlayer(
            model_path=model_path,
            act_size=act_size,
            logger=logger,
            account_configuration=account,
            battle_format=battle_format,
            max_concurrent_battles=1,
            server_configuration=server_cfg,
            team=team_text,
        )
    if model_type == "bc":
        return BehaviorClonePlayer(
            model_path=model_path,
            stats_path=stats_path,
            act_size=act_size,
            logger=logger,
            account_configuration=account,
            battle_format=battle_format,
            max_concurrent_battles=1,
            server_configuration=server_cfg,
            team=team_text,
        )
    raise ValueError(f"unsupported model type: {model_type}")


def _build_opponent(
    *,
    kind: str,
    battle_format: str,
    server_cfg,
    team_text: str | None,
    name_prefix: str,
    act_size: int,
    logger: _BattleLogger,
    policy_path: Path | None,
    policy_type: str,
    policy_stats: Path | None,
    mirror_from: Path,
    mirror_type: str,
    mirror_stats: Path | None,
):
    account = AccountConfiguration(_unique_username(name_prefix), None)
    if kind == "maxbp":
        return MaxBasePowerPlayer(
            account_configuration=account,
            battle_format=battle_format,
            max_concurrent_battles=1,
            server_configuration=server_cfg,
            team=team_text,
        )
    if kind == "random":
        return RandomPlayer(
            account_configuration=account,
            battle_format=battle_format,
            max_concurrent_battles=1,
            server_configuration=server_cfg,
            team=team_text,
        )
    if kind == "policy":
        if policy_path is None:
            raise ValueError("--opponent-model is required when opponent=policy")
        opp_type = _infer_model_type(policy_path, policy_type)
        return _build_policy_player(
            model_path=policy_path,
            model_type=opp_type,
            stats_path=policy_stats,
            act_size=act_size,
            battle_format=battle_format,
            server_cfg=server_cfg,
            team_text=team_text,
            name_prefix=name_prefix,
            logger=logger,
        )
    if kind == "mirror":
        mirror_type = _infer_model_type(mirror_from, mirror_type)
        return _build_policy_player(
            model_path=mirror_from,
            model_type=mirror_type,
            stats_path=mirror_stats,
            act_size=act_size,
            battle_format=battle_format,
            server_cfg=server_cfg,
            team_text=team_text,
            name_prefix=name_prefix,
            logger=logger,
        )
    return SimpleHeuristicsPlayer(
        account_configuration=account,
        battle_format=battle_format,
        max_concurrent_battles=1,
        server_configuration=server_cfg,
        team=team_text,
    )


def main() -> int:
    args = _parse_args()
    model_type = _infer_model_type(args.model, args.model_type)
    server_cfg = _resolve_server_configuration(args.server_url)
    battle_format = args.battle_format
    act_size = action_space_size(battle_format)
    spectator_base = _spectator_base(args.server_url)

    team_text = None
    if args.team:
        team_text = read_showdown_team(Path(args.team))

    opponent_team_text = team_text
    if args.opponent_team:
        opponent_team_text = read_showdown_team(Path(args.opponent_team))

    logger = _BattleLogger(spectator_base, log_level=args.log_level)

    player = _build_policy_player(
        model_path=args.model,
        model_type=model_type,
        stats_path=args.stats,
        act_size=act_size,
        battle_format=battle_format,
        server_cfg=server_cfg,
        team_text=team_text,
        name_prefix=args.player_name_prefix,
        logger=logger,
    )

    opponent = _build_opponent(
        kind=args.opponent,
        battle_format=battle_format,
        server_cfg=server_cfg,
        team_text=opponent_team_text,
        name_prefix=args.opponent_name_prefix,
        act_size=act_size,
        logger=logger,
        policy_path=args.opponent_model,
        policy_type=args.opponent_model_type,
        policy_stats=args.opponent_stats,
        mirror_from=args.model,
        mirror_type=args.model_type,
        mirror_stats=args.stats,
    )

    async def _run():
        await player.battle_against(opponent, n_battles=args.battles)

    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
