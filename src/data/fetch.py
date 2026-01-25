import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from pathlib import Path

from src.config import section

BASE_URL = "https://replay.pokemonshowdown.com"
USER_AGENT_DEFAULT = "poke-rl/0.1 (+contact: none)"


def read_lines(path):
    if not path or not Path(path).exists():
        return []
    content = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in content.splitlines() if line.strip()]


def normalize_token(token):
    token = token.strip()
    if token.startswith("http://") or token.startswith("https://"):
        parsed = urllib.parse.urlparse(token)
        replay_id = Path(parsed.path).name.split(".")[0]
        return replay_id, token
    return token, f"{BASE_URL}/{token}"


def load_robots(url):
    robots = urllib.robotparser.RobotFileParser()
    robots.set_url(urllib.parse.urljoin(url, "/robots.txt"))
    try:
        robots.read()
    except Exception:
        robots.parse(["User-agent: *", "Allow: /"])
    return robots


def http_get(url, user_agent, timeout=15.0):
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def try_fetch_variants(replay_id, user_agent):
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


def best_effort_user_search(user, fmt, limit, user_agent):
    ids = []
    try:
        query = urllib.parse.urlencode({"user": user, "format": fmt, "output": "json"})
        payload = http_get(f"{BASE_URL}/search.json?{query}", user_agent)
        data = json.loads(payload.decode("utf-8", errors="ignore"))
        records = []
        if isinstance(data, dict) and isinstance(data.get("replays"), list):
            records = data["replays"]
        elif isinstance(data, list):
            records = data
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


def collect_targets(settings):
    raw_tokens = []
    raw_tokens.extend(read_lines(settings.get("ids_file")))
    raw_tokens.extend(read_lines(settings.get("urls_file")))
    raw_tokens.extend(settings.get("ids", []))
    if settings.get("user"):
        raw_tokens.extend(
            best_effort_user_search(
                settings["user"],
                settings["format"],
                settings["limit"],
                settings["user_agent"],
            )
        )
    seen = set()
    replay_ids = []
    for token in raw_tokens:
        replay_id, _ = normalize_token(token)
        if replay_id not in seen:
            seen.add(replay_id)
            replay_ids.append(replay_id)
    return replay_ids


def run(settings):
    out_dir = Path(settings["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    robots = load_robots(BASE_URL)
    user_agent = settings.get("user_agent") or USER_AGENT_DEFAULT
    index_path = out_dir / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    except Exception:
        index = {}

    replay_ids = collect_targets(settings)
    if not replay_ids:
        print("no targets provided. use config entries to supply ids, urls, or user.")
        return

    rate = float(settings.get("rate", 0.5))
    delay = 1.0 / max(rate, 0.1)
    fetched = 0
    for replay_id in replay_ids:
        base_path = out_dir / replay_id
        if not settings.get("overwrite") and any(
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
        if rate > 0:
            time.sleep(delay)

    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"fetched {fetched} / {len(replay_ids)} replays into {out_dir}")


def load_settings():
    config = section("data_fetch")
    settings = {}
    settings["out_dir"] = Path(config.get("out_dir", "data/raw/downloaded"))
    ids_file = Path(config.get("ids_file", ""))
    settings["ids_file"] = ids_file if ids_file.exists() else None
    urls_file = Path(config.get("urls_file", ""))
    settings["urls_file"] = urls_file if urls_file.exists() else None
    settings["ids"] = list(config.get("ids", []))
    settings["user"] = config.get("user")
    settings["format"] = config.get("format", "gen9doublesou")
    settings["limit"] = int(config.get("limit", 200))
    settings["rate"] = float(config.get("rate", 0.5))
    settings["user_agent"] = config.get("user_agent") or USER_AGENT_DEFAULT
    settings["overwrite"] = bool(config.get("overwrite", False))
    return settings


def build_arg_parser(defaults):
    parser = argparse.ArgumentParser(description="Fetch Pokémon Showdown replays.")

    parser.add_argument("--out-dir", type=Path, default=defaults.get("out_dir"))
    parser.add_argument("--ids-file", type=Path, default=defaults.get("ids_file"))
    parser.add_argument("--urls-file", type=Path, default=defaults.get("urls_file"))
    parser.add_argument("--ids", nargs="*", default=defaults.get("ids"))
    parser.add_argument("--user", type=str, default=defaults.get("user"))
    parser.add_argument("--format", type=str, default=defaults.get("format"))
    parser.add_argument("--limit", type=int, default=defaults.get("limit"))
    parser.add_argument("--rate", type=float, default=defaults.get("rate"))
    parser.add_argument("--user-agent", type=str, default=defaults.get("user_agent"))
    parser.add_argument("--overwrite", action="store_true", default=defaults.get("overwrite"))

    return parser


def merge_cli_overrides(defaults, args):
    settings = defaults.copy()
    for key in [
        "out_dir",
        "ids_file",
        "urls_file",
        "ids",
        "user",
        "format",
        "limit",
        "rate",
        "user_agent",
        "overwrite",
    ]:
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    return settings


def main(argv: list[str] | None = None):
    defaults = load_settings()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)
    settings = merge_cli_overrides(defaults, args)
    run(settings)


__all__ = ["main", "run"]
