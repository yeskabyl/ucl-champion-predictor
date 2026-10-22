"""Data acquisition, normalisation and validation.

Two interchangeable sources produce the *same* normalised match table:

* ``FootballDataSource`` -- football-data.org REST API v4 (primary; needs a free API token).
* ``OpenFootballSource`` -- public-domain (CC0) openfootball text files on GitHub (no token).
  Used as the documented alternative when the API cannot provide a season, and to run the
  project end-to-end without credentials.

Design rules
------------
* Nothing is invented: missing scores stay ``NaN``; unknown seasons are reported, never substituted.
* Every row carries ``source`` and ``retrieved_at``; raw payloads are snapshotted to ``data/raw``.
* The label is the **90-minute** result (``home_goals_90``/``away_goals_90``). Extra time and
  penalties are stored separately because the tournament simulator models them separately.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, Sequence
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .utils import (
    PROJECT_ROOT,
    ProjectPaths,
    Stage,
    get_logger,
    pick_display_names,
    season_label,
    strip_country_suffix,
    team_id_from_name,
    utc_now_iso,
)

log = get_logger("ucl.data")

# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class DataSourceError(Exception):
    """Base class for every data-acquisition problem (always carries a human-readable message)."""


class MissingTokenError(DataSourceError):
    pass


class AuthError(DataSourceError):
    """HTTP 400/401: token missing or rejected."""


class RestrictedResourceError(DataSourceError):
    """HTTP 403: the account's plan does not include this competition / season."""


class NotFoundError(DataSourceError):
    """HTTP 404: unknown competition or season."""


class RateLimitError(DataSourceError):
    """HTTP 429 that persisted after all retries."""


class TransientError(DataSourceError):
    """Network failure or HTTP 5xx that persisted after all retries."""


class SchemaError(DataSourceError):
    """The response was reachable but not shaped the way the code expects."""


class DataValidationError(DataSourceError):
    """The normalised table violates a hard invariant."""


class OfficialDataUnavailable(DataSourceError):
    """Official participants / fixtures for the requested season are not in the data."""


# --------------------------------------------------------------------------------------
# Normalised schema
# --------------------------------------------------------------------------------------
MATCH_COLUMNS = [
    "match_id", "season", "utc_date", "stage", "matchday", "group",
    "home_team_id", "home_team", "away_team_id", "away_team",
    "status", "is_completed",
    "home_goals_90", "away_goals_90",      # regular-time score: the prediction label
    "home_goals_aet", "away_goals_aet",    # score after extra time, before any shoot-out
    "decided_by", "pens_home", "pens_away", "winner",
    "neutral", "source", "source_match_id", "home_source_id", "away_source_id", "retrieved_at",
]
_REQUIRED_NON_NULL = ["match_id", "season", "utc_date", "stage", "status"]

# Seasons whose QF/SF were single legs at a neutral venue (Lisbon "Final 8", COVID restart).
NEUTRAL_VENUE_OVERRIDES = {(2019, Stage.QUARTER), (2019, Stage.SEMI)}


def _empty_matches() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in MATCH_COLUMNS})


# --------------------------------------------------------------------------------------
# Token handling
# --------------------------------------------------------------------------------------
TOKEN_ENV_VAR = "FOOTBALL_DATA_API_TOKEN"


def _read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip("'\"")
    return out


def get_api_token(explicit: str | None = None, dotenv_path: Path | None = None) -> str | None:
    """Token precedence: explicit argument > environment variable > project ``.env`` file.

    Returns ``None`` when nothing is configured; the token is never logged or written anywhere.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get(TOKEN_ENV_VAR, "").strip()
    if env:
        return env
    value = _read_dotenv(dotenv_path or PROJECT_ROOT / ".env").get(TOKEN_ENV_VAR, "").strip()
    return value or None


# --------------------------------------------------------------------------------------
# Result container + source protocol
# --------------------------------------------------------------------------------------
@dataclass
class LoadResult:
    matches: pd.DataFrame
    source: str
    retrieved_at: str
    seasons_loaded: list[int] = field(default_factory=list)
    unavailable: dict[int, str] = field(default_factory=dict)  # season -> reason
    notes: list[str] = field(default_factory=list)
    from_cache: dict[int, bool] = field(default_factory=dict)


class MatchDataSource(Protocol):
    """Implement this to plug in another provider. Must return the normalised schema."""

    name: str

    def load(self, seasons: Sequence[int], refresh: bool = False) -> LoadResult: ...


# --------------------------------------------------------------------------------------
# Raw snapshots
# --------------------------------------------------------------------------------------
class RawSnapshotStore:
    """Writes untouched provider responses plus a provenance sidecar so API calls can be avoided."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, source: str, key: str, ext: str) -> Path:
        return self.root / source / f"{key}.{ext}"

    def save(self, source: str, key: str, body: str, url: str, ext: str = "json", provider: str | None = None) -> dict:
        """``source`` is the folder name; ``provider`` the human-readable origin recorded in the sidecar."""
        path = self._path(source, key, ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        meta = {
            "source": provider or source,
            "url": url,
            "retrieved_at": utc_now_iso(),
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "bytes": len(body.encode("utf-8")),
        }
        path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def load(self, source: str, key: str, ext: str = "json") -> tuple[str, dict] | None:
        path = self._path(source, key, ext)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if not path.is_file() or not meta_path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8"), json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # corrupt cache -> behave as a miss
            log.warning("ignoring unreadable snapshot %s (%s)", path, exc)
            return None

    @staticmethod
    def age_hours(meta: dict) -> float:
        retrieved = datetime.fromisoformat(meta["retrieved_at"])
        return (datetime.now(timezone.utc) - retrieved).total_seconds() / 3600.0


# --------------------------------------------------------------------------------------
# football-data.org client
# --------------------------------------------------------------------------------------
class RateLimiter:
    """Sliding-window limiter (free tier: 10 requests / minute)."""

    def __init__(self, max_calls: int = 10, period: float = 60.0,
                 monotonic: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.max_calls, self.period = max_calls, period
        self._mono, self._sleep = monotonic, sleep
        self._stamps: deque[float] = deque()

    def wait(self) -> float:
        """Block until a call is allowed. Returns the seconds slept."""
        slept = 0.0
        now = self._mono()
        while self._stamps and now - self._stamps[0] >= self.period:
            self._stamps.popleft()
        if len(self._stamps) >= self.max_calls:
            slept = max(0.0, self.period - (now - self._stamps[0])) + 0.05
            self._sleep(slept)
            now = self._mono()
            while self._stamps and now - self._stamps[0] >= self.period:
                self._stamps.popleft()
        self._stamps.append(self._mono())
        return slept


class FootballDataClient:
    """Thin, defensive wrapper around ``https://api.football-data.org/v4``.

    * timeouts on every call; retries with exponential backoff for network errors and 5xx
    * 429 handling honours the server's "Wait N seconds" hint
    * 400/401 -> ``AuthError``, 403 -> ``RestrictedResourceError`` (plan limit), 404 -> ``NotFoundError``
    """

    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self, token: str | None = None, *, base_url: str | None = None,
                 session: requests.Session | None = None, timeout: tuple[float, float] = (5.0, 30.0),
                 max_retries: int = 4, backoff_base: float = 1.5, requests_per_minute: int = 10,
                 sleep: Callable[[float], None] = time.sleep, monotonic: Callable[[], float] = time.monotonic):
        token = get_api_token(token)
        if not token:
            raise MissingTokenError(
                f"No football-data.org API token found. Register for a free token at "
                f"https://www.football-data.org/client/register and set the {TOKEN_ENV_VAR} environment "
                f"variable (or Colab secret / .env file). The token is never stored in the repository."
            )
        self._token = token
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.session = session or requests.Session()
        self.timeout, self.max_retries, self.backoff_base = timeout, max_retries, backoff_base
        self._sleep = sleep
        self._limiter = RateLimiter(requests_per_minute, 60.0, monotonic, sleep)
        self.last_headers: dict[str, str] = {}

    # ---- HTTP -----------------------------------------------------------------------
    def _get(self, path: str, params: dict | None = None) -> tuple[dict, str]:
        base = f"{self.base_url}/{path.lstrip('/')}"
        url = base + (f"?{urlencode(params)}" if params else "")
        headers = {"X-Auth-Token": self._token, "Accept": "application/json"}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._limiter.wait()
            try:
                resp = self.session.get(base, headers=headers, params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                self._backoff(attempt, f"network error ({type(exc).__name__})")
                continue
            self.last_headers = {k.lower(): v for k, v in dict(resp.headers).items()}
            code = resp.status_code
            if code == 200:
                try:
                    return resp.json(), url
                except ValueError as exc:
                    raise SchemaError(f"{url} returned HTTP 200 but the body is not valid JSON") from exc
            if code == 429:
                wait = self._retry_after(resp)
                last_error = RateLimitError(f"HTTP 429 from {url}: {self._message(resp)}")
                if attempt >= self.max_retries:
                    break
                log.warning("rate limited by football-data.org; sleeping %.0fs", wait)
                self._sleep(wait)
                continue
            if 500 <= code < 600:
                last_error = TransientError(f"HTTP {code} from {url}")
                self._backoff(attempt, f"HTTP {code}")
                continue
            msg = self._message(resp)
            if code in (400, 401):
                raise AuthError(f"HTTP {code} from {url}: {msg}. Check that the API token is valid.")
            if code == 403:
                raise RestrictedResourceError(
                    f"HTTP 403 from {url}: {msg}. The resource is not included in this token's plan "
                    f"(free tier restricts competitions/seasons)."
                )
            if code == 404:
                raise NotFoundError(f"HTTP 404 from {url}: {msg}")
            raise DataSourceError(f"Unexpected HTTP {code} from {url}: {msg}")
        if isinstance(last_error, DataSourceError):
            raise last_error
        raise TransientError(f"Giving up on {url} after {self.max_retries + 1} attempts: {last_error}")

    def _backoff(self, attempt: int, why: str) -> None:
        if attempt >= self.max_retries:
            return
        delay = self.backoff_base ** (attempt + 1)
        log.warning("%s; retrying in %.1fs (attempt %d/%d)", why, delay, attempt + 1, self.max_retries)
        self._sleep(delay)

    @staticmethod
    def _message(resp) -> str:
        try:
            body = resp.json()
            return str(body.get("message") or body)[:300]
        except Exception:  # noqa: BLE001 - best-effort diagnostics only
            return (getattr(resp, "text", "") or "")[:300]

    def _retry_after(self, resp) -> float:
        header = resp.headers.get("Retry-After") if getattr(resp, "headers", None) else None
        if header and str(header).strip().isdigit():
            return float(header) + 1.0
        m = re.search(r"(\d+)\s*seconds?", self._message(resp))
        if m:
            return float(m.group(1)) + 1.0
        return 61.0  # a full window: safe default when the server gives no hint

    # ---- endpoints ------------------------------------------------------------------
    def get_competition(self, code: str = "CL") -> dict:
        data, _ = self._get(f"competitions/{code}")
        if "code" not in data:
            raise SchemaError("competition payload has no 'code' field")
        return data

    def get_matches(self, season: int, code: str = "CL") -> tuple[dict, str]:
        """All matches of a season (``season`` is the start year: 2026 == 2026/27)."""
        data, url = self._get(f"competitions/{code}/matches", {"season": int(season)})
        if not isinstance(data.get("matches"), list):
            raise SchemaError(f"matches payload for season {season} has no 'matches' list; keys={sorted(data)}")
        return data, url

    def verify_access(self, code: str = "CL") -> dict:
        """Cheap health check: is the token valid and what does the provider say about the competition?"""
        comp = self.get_competition(code)
        seasons = comp.get("seasons") or []
        years = sorted({int(str(s["startDate"])[:4]) for s in seasons if s.get("startDate")})
        return {
            "ok": True,
            "competition": comp.get("name"),
            "code": comp.get("code"),
            "current_season": (comp.get("currentSeason") or {}).get("startDate"),
            "seasons_listed_by_provider": years,
            "requests_available_minute": self.last_headers.get("x-requests-available-minute"),
        }


# --------------------------------------------------------------------------------------
# football-data.org normalisation
# --------------------------------------------------------------------------------------
_FD_STAGE_MAP = {
    "GROUP_STAGE": Stage.GROUP, "LEAGUE_STAGE": Stage.LEAGUE, "LEAGUE_PHASE": Stage.LEAGUE,
    "PLAYOFFS": Stage.PLAYOFF, "PLAYOFF_ROUND_1": Stage.PLAYOFF, "KNOCKOUT_ROUND_PLAY_OFFS": Stage.PLAYOFF,
    "LAST_16": Stage.LAST_16, "ROUND_OF_16": Stage.LAST_16,
    "QUARTER_FINALS": Stage.QUARTER, "SEMI_FINALS": Stage.SEMI, "FINAL": Stage.FINAL,
}
_FD_STATUS_MAP = {
    "FINISHED": "FINISHED", "AWARDED": "FINISHED",
    "SCHEDULED": "SCHEDULED", "TIMED": "SCHEDULED",
    "POSTPONED": "POSTPONED", "SUSPENDED": "POSTPONED", "CANCELLED": "CANCELLED",
    "IN_PLAY": "IN_PLAY", "PAUSED": "IN_PLAY", "LIVE": "IN_PLAY", "EXTRA_TIME": "IN_PLAY",
    "PENALTY_SHOOTOUT": "IN_PLAY",
}


def _pair(node) -> tuple[float, float]:
    """(home, away) from a score node; the provider has used both 'home/away' and 'homeTeam/awayTeam'."""
    if not isinstance(node, dict):
        return np.nan, np.nan
    h = node.get("home", node.get("homeTeam"))
    a = node.get("away", node.get("awayTeam"))
    return (np.nan if h is None else float(h)), (np.nan if a is None else float(a))


def _has(pair: tuple[float, float]) -> bool:
    return not (np.isnan(pair[0]) or np.isnan(pair[1]))


def _fd_scores(score: dict | None) -> dict:
    """Decode football-data's score node into regular-time / after-ET / shoot-out components.

    Per the provider's documentation ("Dealing with scores") ``fullTime`` is the *final* score and
    already includes extra-time and shoot-out goals, whereas ``regularTime``, ``extraTime`` and
    ``penalties`` hold the individual parts. The 90-minute label therefore comes from
    ``regularTime`` when present, or ``fullTime - extraTime - penalties`` as a derived fallback.
    If neither is possible the 90-minute score is left missing -- never guessed.
    """
    nan2 = (np.nan, np.nan)
    out = {"reg": nan2, "aet": nan2, "pen": nan2, "decided_by": "REGULAR", "winner": None, "issue": None}
    if not isinstance(score, dict):
        return out
    duration = score.get("duration") or "REGULAR"
    out["decided_by"] = duration if duration in ("REGULAR", "EXTRA_TIME", "PENALTY_SHOOTOUT") else "REGULAR"
    out["winner"] = score.get("winner")
    ft, rt, et, pen = (_pair(score.get(k)) for k in ("fullTime", "regularTime", "extraTime", "penalties"))
    if out["decided_by"] == "REGULAR":
        out["reg"] = rt if _has(rt) else ft
        out["aet"] = out["reg"]
        return out
    out["pen"] = pen if out["decided_by"] == "PENALTY_SHOOTOUT" and _has(pen) else nan2
    if _has(ft):
        pen_part = out["pen"] if _has(out["pen"]) else (0.0, 0.0)
        out["aet"] = (ft[0] - pen_part[0], ft[1] - pen_part[1])
    if _has(rt):
        out["reg"] = rt
    elif _has(out["aet"]) and _has(et):
        out["reg"] = (out["aet"][0] - et[0], out["aet"][1] - et[1])
    else:
        out["issue"] = "90-minute score not derivable (no regularTime and no extraTime)"
    if _has(out["reg"]) and (out["reg"][0] < 0 or out["reg"][1] < 0):
        out["reg"] = nan2
        out["issue"] = "derived 90-minute score was negative"
    return out


def normalize_football_data_matches(payload: dict, *, season: int | None = None, retrieved_at: str | None = None,
                                    source_url: str = "") -> pd.DataFrame:
    """Raw ``/competitions/CL/matches`` JSON -> normalised table (see ``MATCH_COLUMNS``)."""
    matches = payload.get("matches")
    if not isinstance(matches, list):
        raise SchemaError("payload has no 'matches' list")
    retrieved_at = retrieved_at or utc_now_iso()
    rows, unknown_stages, score_issues = [], {}, 0
    for m in matches:
        if not isinstance(m, dict) or m.get("id") is None or not m.get("utcDate"):
            log.warning("skipping malformed match record: %s", str(m)[:120])
            continue
        stage_raw = m.get("stage") or ""
        ms = season if season is not None else _season_from_match(m)
        stage = _FD_STAGE_MAP.get(stage_raw)
        if stage is None and stage_raw == "REGULAR_SEASON" and ms is not None and ms >= 2024:
            stage = Stage.LEAGUE
        if stage is None:
            unknown_stages[stage_raw] = unknown_stages.get(stage_raw, 0) + 1
            stage = Stage.OTHER
        status_raw = m.get("status") or ""
        status = _FD_STATUS_MAP.get(status_raw, "SCHEDULED" if status_raw == "" else status_raw)
        sc = _fd_scores(m.get("score"))
        if sc["issue"]:
            score_issues += 1
        reg = sc["reg"]
        completed = status == "FINISHED" and _has(reg)
        if status == "FINISHED" and not completed:
            log.warning("match %s is FINISHED but has no usable 90-minute score (%s); treated as not completed",
                        m["id"], sc["issue"] or "fullTime missing")
        if not completed:  # never carry partial / live scores into the table
            reg, aet, pen = (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan)
        else:
            aet, pen = sc["aet"], sc["pen"]
        home, away = m.get("homeTeam") or {}, m.get("awayTeam") or {}
        hname, aname = home.get("name"), away.get("name")
        group = m.get("group")
        rows.append({
            "match_id": f"fd:{m['id']}",
            "season": ms,
            "utc_date": m["utcDate"],
            "stage": stage,
            "matchday": m.get("matchday"),
            "group": group,
            "home_team_id": team_id_from_name(hname) if hname else None,
            "home_team": hname,
            "away_team_id": team_id_from_name(aname) if aname else None,
            "away_team": aname,
            "status": status,
            "is_completed": completed,
            "home_goals_90": reg[0], "away_goals_90": reg[1],
            "home_goals_aet": aet[0], "away_goals_aet": aet[1],
            "decided_by": sc["decided_by"] if completed else None,
            "pens_home": pen[0], "pens_away": pen[1],
            "winner": sc["winner"] if completed else None,
            "neutral": stage == Stage.FINAL or (ms, stage) in NEUTRAL_VENUE_OVERRIDES,
            "source": "football-data.org",
            "source_match_id": str(m["id"]),
            "home_source_id": None if home.get("id") is None else str(home["id"]),
            "away_source_id": None if away.get("id") is None else str(away["id"]),
            "retrieved_at": retrieved_at,
        })
    if unknown_stages:
        log.warning("unmapped football-data stage values (kept as OTHER, excluded from modelling): %s", unknown_stages)
    if score_issues:
        log.warning("%d finished matches had a non-derivable 90-minute score", score_issues)
    df = pd.DataFrame(rows, columns=MATCH_COLUMNS) if rows else _empty_matches()
    return _finalise_types(df)


def _season_from_match(m: dict) -> int | None:
    start = (m.get("season") or {}).get("startDate")
    return int(str(start)[:4]) if start else None


def describe_football_data_payload(payload: dict) -> dict:
    """Schema probe for a real response: which values/fields does the provider actually use?"""
    matches = payload.get("matches") or []
    keys = lambda f: sorted({k for m in matches for k in (m.get(f) or {}).keys()})  # noqa: E731
    score_nodes = {k for m in matches for k in (m.get("score") or {}).keys()}
    return {
        "n_matches": len(matches),
        "top_level_keys": sorted(payload.keys()),
        "match_keys": sorted({k for m in matches for k in m.keys()}),
        "stages": sorted({str(m.get("stage")) for m in matches}),
        "statuses": sorted({str(m.get("status")) for m in matches}),
        "durations": sorted({str((m.get("score") or {}).get("duration")) for m in matches}),
        "score_nodes": sorted(score_nodes),
        "team_keys": keys("homeTeam"),
        "unmapped_stages": sorted({str(m.get("stage")) for m in matches} - set(_FD_STAGE_MAP)),
        "missing_required": [k for k in ("id", "utcDate", "status", "stage", "homeTeam", "awayTeam", "score")
                             if any(k not in m for m in matches)],
    }


class FootballDataSource:
    """``MatchDataSource`` backed by football-data.org with on-disk snapshots."""

    name = "football-data.org"

    def __init__(self, client: FootballDataClient | None = None, *, paths: ProjectPaths | None = None,
                 competition: str = "CL", max_age_hours: float = 6.0):
        self._client = client
        self.competition = competition
        self.store = RawSnapshotStore((paths or ProjectPaths.default()).data_raw)
        self.max_age_hours = max_age_hours

    @property
    def client(self) -> FootballDataClient:
        if self._client is None:
            self._client = FootballDataClient()  # raises MissingTokenError with setup instructions
        return self._client

    def load(self, seasons: Sequence[int], refresh: bool = False) -> LoadResult:
        frames, unavailable, from_cache, loaded = [], {}, {}, []
        newest_meta = None
        for season in sorted(set(int(s) for s in seasons)):
            key = f"{self.competition}_{season}"
            cached = None if refresh else self.store.load("football_data", key)
            payload = meta = None
            if cached is not None:
                body, meta = cached
                try:
                    payload = json.loads(body)
                except ValueError:
                    log.warning("snapshot %s is corrupt; fetching it again", key)
                    payload = None
                if payload is not None and not self._season_is_final(payload) \
                        and RawSnapshotStore.age_hours(meta) > self.max_age_hours:
                    payload = None  # live season and stale snapshot -> refetch
            if payload is not None:
                from_cache[season] = True
            else:
                try:
                    payload, url = self.client.get_matches(season, self.competition)
                except (RestrictedResourceError, NotFoundError) as exc:
                    unavailable[season] = f"{type(exc).__name__}: {exc}"
                    log.warning("season %s unavailable from football-data.org: %s", season, exc)
                    continue
                except AuthError:
                    raise
                except (RateLimitError, TransientError, SchemaError) as exc:
                    unavailable[season] = f"{type(exc).__name__}: {exc}"
                    log.warning("season %s could not be fetched: %s", season, exc)
                    continue
                meta = self.store.save("football_data", key, json.dumps(payload), url, provider=self.name)
                from_cache[season] = False
            df = normalize_football_data_matches(payload, season=season, retrieved_at=meta["retrieved_at"],
                                                 source_url=meta.get("url", ""))
            if df.empty:
                unavailable[season] = "provider returned zero matches"
                continue
            frames.append(df)
            loaded.append(season)
            newest_meta = meta
        matches = pd.concat(frames, ignore_index=True) if frames else _empty_matches()
        return LoadResult(_finalise_types(matches), self.name, (newest_meta or {}).get("retrieved_at", utc_now_iso()),
                          loaded, unavailable, from_cache=from_cache)

    @staticmethod
    def _season_is_final(payload: dict) -> bool:
        ms = payload.get("matches") or []
        return bool(ms) and all(m.get("status") in ("FINISHED", "AWARDED", "CANCELLED") for m in ms)


# --------------------------------------------------------------------------------------
# openfootball (CC0) alternative source
# --------------------------------------------------------------------------------------
OPENFOOTBALL_URL = "https://raw.githubusercontent.com/openfootball/champions-league/master/{start}-{end}/cl.txt"
_MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
_HEADER_RE = re.compile(r"^[▪»]\s*(?P<name>.+?)\s*$")
_DATE_RE = re.compile(r"^\s+(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\.?\s+)?(?P<mon>[A-Z][a-z]{2})[ /]+(?P<day>\d{1,2})(?:\s+(?P<year>\d{4}))?\s*$")
_MATCH_RE = re.compile(
    r"^\s+(?:(?P<time>\d{1,2}[:.]\d{2})\s+)?(?P<home>.+?)\s+\((?P<hc>[A-Z]{2,3})\)\s+v\s+(?P<away>.+?)\s+\((?P<ac>[A-Z]{2,3})\)(?:\s+(?P<rest>.*))?$"
)
_SCORE_RE = re.compile(
    r"^(?:(?P<pa>\d+)-(?P<pb>\d+)\s+pen\.\s+)?(?P<a>\d+)-(?P<b>\d+)(?P<aet>\s+a\.e\.t\.)?(?:\s*\((?P<paren>[^)]*)\))?"
)
_PAIR_RE = re.compile(r"(\d+)\s*-\s*(\d+)")
_LOCAL_TZ = ZoneInfo("Europe/Paris")  # openfootball lists UEFA kick-off times in CET/CEST


def _of_stage(name: str, season: int) -> tuple[str, int | None, str | None]:
    n = name.lower().replace("finals,", "").replace("-", "").strip()
    md = re.search(r"matchday\s+(\d+)", n)
    matchday = int(md.group(1)) if md else None
    if n.startswith(("group", "gruppe")):
        return Stage.GROUP, None, name.split()[-1]
    if n.startswith("league"):
        return Stage.LEAGUE, matchday, None
    if n.startswith("playoff"):
        return Stage.PLAYOFF, matchday, None
    if "round of 16" in n:
        return Stage.LAST_16, None, None
    if n.startswith("quarter"):
        return Stage.QUARTER, None, None
    if n.startswith("semi"):
        return Stage.SEMI, None, None
    if n.strip() == "final":
        return Stage.FINAL, None, None
    return Stage.OTHER, matchday, None


def parse_openfootball_text(text: str, season: int, *, retrieved_at: str | None = None,
                            source_url: str = "") -> pd.DataFrame:
    """Parse one ``cl.txt`` (Football.TXT format) into the normalised table.

    Score notation (verified against the files): ``2-1 (1-0)`` = full-time (half-time);
    ``3-2 a.e.t. (3-0, 1-0)`` = after extra time (90-minute score, half-time);
    ``4-3 pen. 1-1 a.e.t. (1-1, 0-1)`` = shoot-out, after extra time, (90 minutes, half-time).
    Lines that cannot be interpreted are collected in ``df.attrs['parse_issues']`` -- not dropped silently.
    """
    retrieved_at = retrieved_at or utc_now_iso()
    rows, issues = [], []
    stage, matchday, group = Stage.OTHER, None, None
    year, last_month, cur_date, cur_time = season, 0, None, "00:00"
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#") or line.startswith("="):
            continue
        h = _HEADER_RE.match(line)
        if h:
            stage, matchday, group = _of_stage(h.group("name"), season)
            if stage == Stage.GROUP:  # every group lists Sep..Dec again; its first date may omit the year
                year, last_month = season, 0
            continue
        d = _DATE_RE.match(line)
        if d and " v " not in line:
            mon = _MONTHS.get(d.group("mon"))
            if mon is None:
                issues.append((lineno, f"unknown month in date line: {line.strip()}"))
                continue
            if d.group("year"):
                year = int(d.group("year"))
            elif mon < last_month:
                year += 1  # calendar rolled over (Dec -> Jan)
            last_month = mon
            cur_date = (year, mon, int(d.group("day")))
            cur_time = "00:00"
            continue
        m = _MATCH_RE.match(line)
        if not m:
            issues.append((lineno, f"unrecognised line: {line.strip()[:80]}"))
            continue
        if cur_date is None:
            issues.append((lineno, "match line before any date line"))
            continue
        if m.group("time"):
            cur_time = m.group("time").replace(".", ":")
        hh, mm = (int(x) for x in cur_time.split(":"))
        local = datetime(*cur_date, hh, mm, tzinfo=_LOCAL_TZ)
        utc_dt = local.astimezone(timezone.utc).replace(tzinfo=None)
        hname, aname = strip_country_suffix(m.group("home")), strip_country_suffix(m.group("away"))
        rest = (m.group("rest") or "").strip()
        rec = {
            "season": season, "utc_date": utc_dt, "stage": stage, "matchday": matchday, "group": group,
            "home_team_id": team_id_from_name(hname), "home_team": hname,
            "away_team_id": team_id_from_name(aname), "away_team": aname,
            "status": "SCHEDULED", "is_completed": False,
            "home_goals_90": np.nan, "away_goals_90": np.nan, "home_goals_aet": np.nan, "away_goals_aet": np.nan,
            "decided_by": None, "pens_home": np.nan, "pens_away": np.nan, "winner": None,
            "neutral": stage == Stage.FINAL or (season, stage) in NEUTRAL_VENUE_OVERRIDES,
            "source": "openfootball", "home_source_id": None, "away_source_id": None, "retrieved_at": retrieved_at,
        }
        if rest:
            s = _SCORE_RE.match(rest)
            if not s:
                issues.append((lineno, f"unparseable score '{rest[:40]}' -> match kept as not completed"))
            else:
                a, b = int(s.group("a")), int(s.group("b"))
                paren = [p.strip() for p in (s.group("paren") or "").split(",") if p.strip()]
                pair = lambda t: tuple(int(x) for x in _PAIR_RE.search(t).groups()) if t and _PAIR_RE.search(t) else None  # noqa: E731
                if s.group("aet"):
                    reg = pair(paren[0]) if paren else None
                    if reg is None:
                        issues.append((lineno, "extra-time match without a 90-minute score -> kept as not completed"))
                    else:
                        rec.update(home_goals_90=reg[0], away_goals_90=reg[1], home_goals_aet=a, away_goals_aet=b,
                                   decided_by="EXTRA_TIME", status="FINISHED", is_completed=True)
                        if s.group("pa") is not None:
                            pa, pb = int(s.group("pa")), int(s.group("pb"))
                            rec.update(pens_home=pa, pens_away=pb, decided_by="PENALTY_SHOOTOUT")
                            rec["winner"] = "HOME_TEAM" if pa > pb else "AWAY_TEAM"
                        else:
                            rec["winner"] = "HOME_TEAM" if a > b else "AWAY_TEAM" if a < b else "DRAW"
                else:
                    rec.update(home_goals_90=a, away_goals_90=b, home_goals_aet=a, away_goals_aet=b,
                               decided_by="REGULAR", status="FINISHED", is_completed=True,
                               winner="HOME_TEAM" if a > b else "AWAY_TEAM" if a < b else "DRAW")
        rec["match_id"] = (f"of:{season}:{stage}:{utc_dt:%Y%m%d}:{rec['home_team_id']}:{rec['away_team_id']}")
        rec["source_match_id"] = rec["match_id"]
        rows.append(rec)
    df = pd.DataFrame(rows, columns=MATCH_COLUMNS) if rows else _empty_matches()
    df = _finalise_types(df)
    df.attrs["parse_issues"] = issues
    df.attrs["source_url"] = source_url
    return df


class OpenFootballSource:
    """``MatchDataSource`` for the openfootball/champions-league repository (CC0 / public domain)."""

    name = "openfootball"

    def __init__(self, *, paths: ProjectPaths | None = None, session: requests.Session | None = None,
                 timeout: tuple[float, float] = (5.0, 30.0), max_retries: int = 3,
                 sleep: Callable[[float], None] = time.sleep):
        self.store = RawSnapshotStore((paths or ProjectPaths.default()).data_raw)
        self.session, self.timeout, self.max_retries, self._sleep = session or requests.Session(), timeout, max_retries, sleep

    def _download(self, season: int) -> tuple[str, str]:
        url = OPENFOOTBALL_URL.format(start=season, end=str(season + 1)[-2:])
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last = exc
                self._sleep(1.5 ** (attempt + 1))
                continue
            if resp.status_code == 200:
                resp.encoding = "utf-8"
                return resp.text, url
            if resp.status_code == 404:
                raise NotFoundError(f"openfootball has no file for season {season_label(season)} ({url})")
            if resp.status_code >= 500 or resp.status_code == 429:
                last = TransientError(f"HTTP {resp.status_code} from {url}")
                self._sleep(1.5 ** (attempt + 1))
                continue
            raise DataSourceError(f"Unexpected HTTP {resp.status_code} from {url}")
        raise TransientError(f"Giving up on {url}: {last}")

    def load(self, seasons: Sequence[int], refresh: bool = False) -> LoadResult:
        frames, unavailable, from_cache, loaded, notes = [], {}, {}, [], []
        newest = None
        for season in sorted(set(int(s) for s in seasons)):
            key = f"cl_{season}-{str(season + 1)[-2:]}"
            cached = None if refresh else self.store.load("openfootball", key, ext="txt")
            if cached is not None:
                text, meta = cached
                from_cache[season] = True
            else:
                try:
                    text, url = self._download(season)
                except (NotFoundError, TransientError, DataSourceError) as exc:
                    unavailable[season] = f"{type(exc).__name__}: {exc}"
                    log.warning("season %s unavailable from openfootball: %s", season, exc)
                    continue
                meta = self.store.save("openfootball", key, text, url, ext="txt", provider="openfootball (CC0)")
                from_cache[season] = False
            df = parse_openfootball_text(text, season, retrieved_at=meta["retrieved_at"], source_url=meta["url"])
            for lineno, msg in df.attrs.get("parse_issues", []):
                notes.append(f"{season_label(season)} line {lineno}: {msg}")
                log.warning("openfootball %s line %d: %s", season_label(season), lineno, msg)
            frames.append(df)
            loaded.append(season)
            newest = meta
        matches = pd.concat(frames, ignore_index=True) if frames else _empty_matches()
        return LoadResult(_finalise_types(matches), self.name, (newest or {}).get("retrieved_at", utc_now_iso()),
                          loaded, unavailable, notes, from_cache)


# --------------------------------------------------------------------------------------
# Validation / cleaning
# --------------------------------------------------------------------------------------
@dataclass
class ValidationReport:
    n_rows: int = 0
    n_completed: int = 0
    n_fixtures: int = 0
    duplicates_dropped: int = 0
    dropped_invalid: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _finalise_types(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True).dt.tz_localize(None)
    for c in ("home_goals_90", "away_goals_90", "home_goals_aet", "away_goals_aet", "pens_home", "pens_away"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["matchday"] = pd.to_numeric(df["matchday"], errors="coerce").astype("Int64")
    df["is_completed"] = df["is_completed"].astype(bool)
    df["neutral"] = df["neutral"].astype(bool)
    return df


def clean_matches(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Enforce invariants. Removes exact duplicates and impossible rows, and *reports* everything it did."""
    report = ValidationReport(n_rows=len(df))
    missing = [c for c in MATCH_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"match table is missing required columns: {missing}")
    if df.empty:
        return df, report
    out = _finalise_types(df[MATCH_COLUMNS])

    bad = out[_REQUIRED_NON_NULL].isna().any(axis=1)
    if bad.any():
        report.warnings.append(f"dropped {int(bad.sum())} rows missing one of {_REQUIRED_NON_NULL}")
        report.dropped_invalid += int(bad.sum())
        out = out[~bad]

    same_team = (out["home_team_id"].notna() & (out["home_team_id"] == out["away_team_id"]))
    if same_team.any():
        report.warnings.append(f"dropped {int(same_team.sum())} rows where home and away team are identical")
        report.dropped_invalid += int(same_team.sum())
        out = out[~same_team]

    neg = out["is_completed"] & ((out["home_goals_90"] < 0) | (out["away_goals_90"] < 0))
    nan_scores = out["is_completed"] & (out["home_goals_90"].isna() | out["away_goals_90"].isna())
    invalid = neg | nan_scores
    if invalid.any():
        report.warnings.append(f"dropped {int(invalid.sum())} completed rows with negative/missing scores")
        report.dropped_invalid += int(invalid.sum())
        out = out[~invalid]

    n0 = len(out)
    out = out.sort_values(["utc_date", "match_id"]).drop_duplicates("match_id", keep="last")
    known = out.dropna(subset=["home_team_id", "away_team_id"]).assign(_day=lambda d: d["utc_date"].dt.date)
    pair_dupes = known.duplicated(["season", "stage", "home_team_id", "away_team_id", "_day"], keep="last")
    if pair_dupes.any():
        drop_idx = pair_dupes[pair_dupes].index
        report.warnings.append(f"dropped {len(drop_idx)} rows duplicating (season, stage, date, home, away)")
        out = out.drop(index=drop_idx)
    report.duplicates_dropped = n0 - len(out)
    if report.duplicates_dropped:
        log.warning("dropped %d duplicate match rows", report.duplicates_dropped)

    out = out.sort_values(["utc_date", "match_id"]).reset_index(drop=True)
    report.n_completed = int(out["is_completed"].sum())
    report.n_fixtures = int((~out["is_completed"]).sum())
    return out, report


_KNOCKOUT_ORDER = [Stage.PLAYOFF, Stage.LAST_16, Stage.QUARTER, Stage.SEMI, Stage.FINAL]


def undetermined_knockout_rows(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Index:
    """Index of knockout fixtures whose *participants* were not yet determined at ``cutoff``.

    Used to emulate what a data provider would have shown at the time (it lists such slots as TBD). A tie is known
    at ``cutoff`` if one of its legs was already played, or if both clubs had qualified for that round: play-off
    and last-16 seeds once the league phase is over, everyone else once their previous tie was finished.
    Only meaningful on a full historical table (original scores present).
    """
    cutoff = pd.Timestamp(cutoff)
    hidden: list = []
    for _, sdf in df.groupby("season"):
        league = sdf[sdf["stage"] == Stage.LEAGUE]
        league_end = league["utc_date"].max() if len(league) and league["is_completed"].all() else None
        prev_end: dict = {}
        for rnd in _KNOCKOUT_ORDER:
            rows = sdf[(sdf["stage"] == rnd) & sdf["home_team_id"].notna() & sdf["away_team_id"].notna()]
            ties: dict = {}
            for idx, r in rows.iterrows():
                ties.setdefault(frozenset((r["home_team_id"], r["away_team_id"])), []).append((idx, r))
            cur_end: dict = {}
            for key, legs in ties.items():
                needed = 1 if rnd == Stage.FINAL else 2
                dates = [r["utc_date"] for _, r in legs]
                complete = len(legs) >= needed and all(r["is_completed"] for _, r in legs)
                end = max(dates) if complete else None
                if rnd == Stage.PLAYOFF:
                    times = [league_end] * 2
                elif rnd == Stage.LAST_16:
                    times = [prev_end.get(t, league_end) for t in key]
                else:
                    times = [prev_end.get(t) for t in key]
                known = any(d < cutoff for d in dates) or all(tm is not None and tm < cutoff for tm in times)
                if not known:
                    hidden += [idx for idx, _ in legs]
                for t in key:
                    cur_end[t] = end
            prev_end = cur_end
    return pd.Index(hidden)


def as_of(df: pd.DataFrame, cutoff: pd.Timestamp | str, *, mask_undetermined_knockout: bool = False) -> pd.DataFrame:
    """The table as it would have looked at ``cutoff``: later results are hidden, fixtures kept.

    Matches kicking off at or after ``cutoff`` become unplayed fixtures (scores removed). This is what
    makes historical replay and leakage-safe training possible. With ``mask_undetermined_knockout`` the team names
    of knockout ties that were not yet determined at ``cutoff`` are blanked too (historical replay needs this; with
    live data the provider already shows such slots as TBD).
    """
    cutoff = pd.Timestamp(cutoff)
    out = df.copy()
    if mask_undetermined_knockout:
        idx = undetermined_knockout_rows(df, cutoff)
        out.loc[idx, ["home_team_id", "home_team", "away_team_id", "away_team"]] = None
    hide = out["utc_date"] >= cutoff
    for c in ("home_goals_90", "away_goals_90", "home_goals_aet", "away_goals_aet", "pens_home", "pens_away"):
        out.loc[hide, c] = np.nan
    out.loc[hide, ["is_completed"]] = False
    out.loc[hide & (out["status"] == "FINISHED"), "status"] = "SCHEDULED"
    out.loc[hide, ["decided_by", "winner"]] = None
    return out


def split_completed_and_fixtures(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df["is_completed"]].copy(), df[~df["is_completed"]].copy()


def load_matches(sources: Sequence[MatchDataSource], seasons: Sequence[int], *, refresh: bool = False,
                 paths: ProjectPaths | None = None) -> tuple[pd.DataFrame, dict]:
    """Load ``seasons`` from ``sources`` in priority order (one source per season, never row-mixed).

    Returns the cleaned table and a provenance dict (which source supplied which season, what was
    unavailable, validation report). Raises ``DataSourceError`` when no season could be loaded.
    """
    remaining = sorted(set(int(s) for s in seasons))
    frames, season_source, unavailable, notes, retrieved = [], {}, {}, [], {}
    for src in sources:
        if not remaining:
            break
        try:
            res = src.load(remaining, refresh=refresh)
        except MissingTokenError as exc:
            unavailable[src.name] = str(exc)
            log.warning("source %s skipped: %s", src.name, exc)
            continue
        for s in res.seasons_loaded:
            season_source[s] = src.name
        retrieved[src.name] = res.retrieved_at
        notes += res.notes
        unavailable.update({f"{src.name}:{k}": v for k, v in res.unavailable.items()})
        sub = res.matches[res.matches["season"].isin(res.seasons_loaded)] if not res.matches.empty else res.matches
        frames.append(sub)
        remaining = [s for s in remaining if s not in res.seasons_loaded]
    if not frames or all(f.empty for f in frames):
        raise DataSourceError(
            "No Champions League data could be loaded from any source. Details: "
            + json.dumps(unavailable, indent=2, default=str)
        )
    df, report = clean_matches(pd.concat(frames, ignore_index=True))
    provenance = {
        "season_source": {int(k): v for k, v in sorted(season_source.items())},
        "unavailable": unavailable,
        "seasons_not_loaded": remaining,
        "retrieved_at": retrieved,
        "validation": report.as_dict(),
        "parse_notes": notes,
    }
    return df, provenance


def build_team_directory(df: pd.DataFrame) -> pd.DataFrame:
    """team_id -> display name (most recent spelling) plus first/last season seen."""
    parts = []
    for side in ("home", "away"):
        sub = df[[f"{side}_team_id", f"{side}_team", "utc_date", "season"]].dropna(subset=[f"{side}_team_id"])
        sub.columns = ["team_id", "team_name", "utc_date", "season"]
        parts.append(sub)
    both = pd.concat(parts, ignore_index=True)
    names = pick_display_names(zip(both["team_id"], both["team_name"], both["utc_date"]))
    agg = both.groupby("team_id")["season"].agg(first_season="min", last_season="max").reset_index()
    agg["team_name"] = agg["team_id"].map(names)
    return agg[["team_id", "team_name", "first_season", "last_season"]].sort_values("team_name").reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def read_processed(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["utc_date"], dtype={"home_source_id": "object", "away_source_id": "object"})
    return _finalise_types(df)
