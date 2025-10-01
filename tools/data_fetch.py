#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://replay.pokemonshowdown.com"
USER_AGENT_DEFAULT = "poke-rl/0.1 (+contact: none)"

DEFAULT_OUT_DIR = Path("data/raw/downloaded")
DEFAULT_IDS_FILE = Path("data/sources/replays/ids.txt")
DEFAULT_URLS_FILE = Path("data/sources/replays/urls.txt")
DEFAULT_IDS: list[str] = []
DEFAULT_USER = None
DEFAULT_FORMAT = "gen9doublesou"
DEFAULT_LIMIT = 200
DEFAULT_RATE = 0.5
DEFAULT_USER_AGENT = USER_AGENT_DEFAULT
DEFAULT_OVERWRITE = False
CONFIG_PATH = Path("tools/data_fetch_config.json")


def read_lines(path: Path | None) -> list[str]:
    if not path:
        return []
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_token(token: str) -> tuple[str, str]:
    token = token.strip()
    if token.startswith("http://") or token.startswith("https://"):
        parsed = urllib.parse.urlparse(token)
        replay_id = Path(parsed.path).name.split(".")[0]
        return replay_id, token
    return token, f"{BASE_URL}/{token}"


def load_robots(url: str) -> urllib.robotparser.RobotFileParser:
    robots = urllib.robotparser.RobotFileParser()
    robots.set_url(urllib.parse.urljoin(url, "/robots.txt"))
    try:
        robots.read()
    except Exception:
        robots.parse(["User-agent: *", "Allow: /"])
    return robots


def http_get(url: str, user_agent: str, timeout: float = 15.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def try_fetch_variants(replay_id: str, user_agent: str) -> tuple[bytes | None, str | None]:
    for ext in (".json", ".log", ""):
        url = f"{BASE_URL}/{replay_id}{ext}"
        try:
            blob = http_get(url, user_agent=user_agent)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            return None, None
        except Exception:
            return None, None
        if blob:
            return blob, (ext or ".html")
    return None, None


def best_effort_user_search(user: str, fmt: str, limit: int, user_agent: str) -> list[str]:
    ids: list[str] = []
    try:
        query = urllib.parse.urlencode({"user": user, "format": fmt, "output": "json"})
        payload = http_get(f"{BASE_URL}/search.json?{query}", user_agent)
        data = json.loads(payload.decode("utf-8", errors="ignore"))
        records: Iterable[dict[str, object]]
        if isinstance(data, dict) and isinstance(data.get("replays"), list):
            records = data["replays"]  # type: ignore[index]
        elif isinstance(data, list):
            records = data  # type: ignore[assignment]
        else:
            records = []
        for item in records:
            if not isinstance(item, dict):
                continue
            replay_id = item.get("id") or item.get("replayid") or item.get("uploadid")
            if isinstance(replay_id, str) and replay_id.startswith(f"{fmt}-"):
                ids.append(replay_id)
            if len(ids) >= limit:
                break
        if ids:
            return ids[:limit]
    except Exception:
        pass
    try:
        query = urllib.parse.urlencode({"user": user, "format": fmt})
        html = http_get(f"{BASE_URL}/search?{query}", user_agent).decode("utf-8", errors="ignore")
        guesses = re.findall(rf"{re.escape(fmt)}-\d+", html)
        ordered_unique = list(dict.fromkeys(guesses))
        return ordered_unique[:limit]
    except Exception:
        return []


@dataclass
class Settings:
    out_dir: Path
    ids_file: Path | None
    urls_file: Path | None
    ids: list[str]
    user: str | None
    fmt: str
    limit: int
    rate: float
    user_agent: str
    overwrite: bool


def collect_targets(settings: Settings) -> list[str]:
    raw_tokens: list[str] = []
    raw_tokens.extend(read_lines(settings.ids_file))
    raw_tokens.extend(read_lines(settings.urls_file))
    raw_tokens.extend(settings.ids)
    if settings.user:
        raw_tokens.extend(
            best_effort_user_search(
                settings.user, settings.fmt, settings.limit, settings.user_agent
            )
        )
    seen: set[str] = set()
    replay_ids: list[str] = []
    for token in raw_tokens:
        replay_id, _ = normalize_token(token)
        if replay_id not in seen:
            seen.add(replay_id)
            replay_ids.append(replay_id)
    return replay_ids


def run(settings: Settings) -> None:
    settings.out_dir.mkdir(parents=True, exist_ok=True)
    robots = load_robots(BASE_URL)
    user_agent = settings.user_agent or USER_AGENT_DEFAULT
    index_path = settings.out_dir / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    except Exception:
        index = {}

    replay_ids = collect_targets(settings)
    if not replay_ids:
        print("no targets provided. use --ids/--urls or --user to discover replays.")
        return

    delay = 1.0 / max(settings.rate, 0.1)
    fetched = 0
    for replay_id in replay_ids:
        base_path = settings.out_dir / replay_id
        if not settings.overwrite and any(
            base_path.with_suffix(suffix).exists() for suffix in (".json", ".log", ".html")
        ):
            continue
        url = f"{BASE_URL}/{replay_id}"
        if not robots.can_fetch(user_agent, url):
            print(f"robots disallow: {url}")
            continue
        blob, ext = try_fetch_variants(replay_id, user_agent)
        if not blob or not ext:
            print(f"miss: {replay_id}")
            continue
        output_path = base_path.with_suffix(ext)
        try:
            output_path.write_bytes(blob)
        except Exception as exc:
            print(f"error: {replay_id} {exc}")
            continue
        index[replay_id] = {
            "path": str(output_path),
            "ext": ext,
            "format": replay_id.split("-")[0],
        }
        fetched += 1
        if settings.rate > 0:
            time.sleep(delay)

    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"fetched {fetched} / {len(replay_ids)} replays into {settings.out_dir}")


def default_settings() -> Settings:
    config = {
        "out_dir": str(DEFAULT_OUT_DIR),
        "ids_file": str(DEFAULT_IDS_FILE),
        "urls_file": str(DEFAULT_URLS_FILE),
        "ids": list(DEFAULT_IDS),
        "user": DEFAULT_USER,
        "format": DEFAULT_FORMAT,
        "limit": DEFAULT_LIMIT,
        "rate": DEFAULT_RATE,
        "user_agent": DEFAULT_USER_AGENT,
        "overwrite": DEFAULT_OVERWRITE,
    }
    if CONFIG_PATH.exists():
        try:
            payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                config.update(payload)
        except Exception:
            pass
    ids_file = Path(config["ids_file"])
    urls_file = Path(config["urls_file"])
    return Settings(
        out_dir=Path(config["out_dir"]),
        ids_file=ids_file if ids_file.exists() else None,
        urls_file=urls_file if urls_file.exists() else None,
        ids=list(config.get("ids", [])),
        user=config.get("user"),
        fmt=config.get("format", DEFAULT_FORMAT),
        limit=int(config.get("limit", DEFAULT_LIMIT)),
        rate=float(config.get("rate", DEFAULT_RATE)),
        user_agent=config.get("user_agent", DEFAULT_USER_AGENT),
        overwrite=bool(config.get("overwrite", DEFAULT_OVERWRITE)),
    )


def main() -> None:
    settings = default_settings()
    print("data fetch settings:")
    print(f"  out_dir={settings.out_dir}")
    print(f"  ids_file={settings.ids_file}")
    print(f"  urls_file={settings.urls_file}")
    print(f"  ids={settings.ids}")
    print(f"  user={settings.user}")
    print(f"  format={settings.fmt}")
    print(f"  limit={settings.limit}")
    print(f"  rate={settings.rate}")
    print(f"  user_agent={settings.user_agent}")
    print(f"  overwrite={settings.overwrite}")
    run(settings)


if __name__ == "__main__":
    main()
