"""Shared helpers: paths, logging, stage/outcome constants, team-name identity, interval maths.

Nothing in here touches the network or any model, so it is safe to import from every other module.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectPaths:
    """Directory layout. Create with ``ProjectPaths.default()`` or point at a temp dir in tests."""

    root: Path

    @classmethod
    def default(cls) -> "ProjectPaths":
        return cls(PROJECT_ROOT)

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    def ensure(self) -> "ProjectPaths":
        for p in (self.data_raw, self.data_processed, self.outputs):
            p.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------------------
# Logging / reproducibility
# --------------------------------------------------------------------------------------
def get_logger(name: str = "ucl") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("UCL_LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger


def set_global_seed(seed: int) -> None:
    """Seed the stdlib and NumPy global generators (simulation code uses its own ``Generator``)."""
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------------------
# Domain constants
# --------------------------------------------------------------------------------------
CLASSES: tuple[str, str, str] = ("H", "D", "A")  # home win, draw, away win -- column order everywhere


class Stage:
    """Canonical stage names. Both data sources are mapped onto these."""

    GROUP = "GROUP_STAGE"  # 32-team format, up to 2023/24
    LEAGUE = "LEAGUE_PHASE"  # 36-team single-table format, from 2024/25
    PLAYOFF = "PLAYOFF"  # knockout phase play-offs (positions 9-24)
    LAST_16 = "LAST_16"
    QUARTER = "QUARTER_FINAL"
    SEMI = "SEMI_FINAL"
    FINAL = "FINAL"
    OTHER = "OTHER"

    KNOCKOUT = (PLAYOFF, LAST_16, QUARTER, SEMI, FINAL)
    ORDER = (GROUP, LEAGUE, PLAYOFF, LAST_16, QUARTER, SEMI, FINAL)


def outcome_label(home_goals: float, away_goals: float) -> str | None:
    """'H', 'D' or 'A' from a scoreline, ``None`` when either score is missing."""
    if home_goals is None or away_goals is None or np.isnan(home_goals) or np.isnan(away_goals):
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def season_label(start_year: int) -> str:
    """2025 -> '2025/26'."""
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def season_of(d: date | datetime) -> int:
    """Season start-year for a calendar date (UEFA seasons start in July)."""
    return d.year if d.month >= 7 else d.year - 1


# --------------------------------------------------------------------------------------
# Time / IO
# --------------------------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _JsonEncoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401 - json hook
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(path: str | Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, cls=_JsonEncoder)
    return path


def read_json(path: str | Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------
# Monte Carlo uncertainty
# --------------------------------------------------------------------------------------
def wilson_interval(successes, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion (vectorised over ``successes``).

    Used for the Monte Carlo uncertainty of a championship probability: with ``n`` independent
    simulated tournaments and ``k`` titles, it brackets the *simulation* error of ``k / n``.
    It does NOT include uncertainty about the model's match probabilities.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    k = np.asarray(successes, dtype=float)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    low = np.where(k == 0, 0.0, np.clip(centre - half, 0.0, 1.0))   # exact at the boundaries (no 1e-19 rounding noise)
    high = np.where(k == n, 1.0, np.clip(centre + half, 0.0, 1.0))
    return low, high


def binomial_std_error(successes, n: int):
    p = np.asarray(successes, dtype=float) / n
    return np.sqrt(p * (1 - p) / n)


# --------------------------------------------------------------------------------------
# Team identity
# --------------------------------------------------------------------------------------
# Words that carry no identity (legal forms / club-type prefixes). Removing them lets
# "Manchester City" and "Manchester City FC" collapse onto one key.
_STOP_TOKENS = frozenset(
    "fc afc cf sc ac as ssc ss sk fk bk kv krc kaa rsc bsc nk gnk pfc sl club clube de da do di the futbol "
    "calcio bc hsc osc cfr".split()
)

# Remaining spelling variants of the SAME club, after normalisation. Keys and values are
# *normalised* strings (see ``_normalise_key``). Reviewed by hand against every team name that
# appears in openfootball's CL files 2011/12-2025/26; extend it when a new variant shows up
# (``find_possible_duplicate_teams`` flags suspects).
TEAM_ALIASES: dict[str, str] = {
    "rb salzburg": "red bull salzburg",
    "salzburg": "red bull salzburg",
    "qarabag agdam": "qarabag",
    "internazionale milano": "inter",
    "inter milan": "inter",
    "olympiakos piraeus": "olympiakos",
    "olympiacos": "olympiakos",
    "pae olympiakos sfp": "olympiakos",
    "lazio roma": "lazio",
    "feyenoord rotterdam": "feyenoord",
    "psv eindhoven": "psv",
    "sport lisboa e benfica": "benfica",
    "sporting portugal": "sporting cp",
    "sporting braga": "braga",
    "borussia monchengladbach": "monchengladbach",
    "bor monchengladbach": "monchengladbach",
    "dinamo kiev": "dynamo kyiv",
    "dynamo kiev": "dynamo kyiv",
    "zenit st petersburg": "zenit",
    "zenit saint petersburg": "zenit",
    "stade brestois": "brest",
    "stade rennais": "rennes",
    "racing lens": "lens",
    "paris sg": "paris saint germain",
    "psg": "paris saint germain",
    "man city": "manchester city",
    "man united": "manchester united",
    "spurs": "tottenham hotspur",
    "royal antwerp": "antwerp",
    "royale union saint gilloise": "union saint gilloise",
    "union sg": "union saint gilloise",
}


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


_COUNTRY_SUFFIX = re.compile(r"\s*\([A-Z]{2,3}\)\s*$")


def strip_country_suffix(name: str) -> str:
    """'Arsenal FC (ENG)' -> 'Arsenal FC'."""
    return _COUNTRY_SUFFIX.sub("", name).strip()


def _normalise_key(name: str) -> str:
    text = _strip_accents(strip_country_suffix(name)).lower()
    text = text.replace("&", " and ")
    for src, dst in (("ø", "o"), ("đ", "d"), ("ł", "l"), ("æ", "ae"), ("œ", "oe"), ("ß", "ss")):
        text = text.replace(src, dst)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [t for t in text.split() if t not in _STOP_TOKENS and not t.isdigit()]
    return " ".join(tokens)


def canonical_team_key(name: str) -> str:
    """Stable, source-independent identity for a club name (lower-case, accent-free, aliased)."""
    if name is None or not str(name).strip():
        raise ValueError("team name is empty")
    key = _normalise_key(str(name))
    if not key:  # name consisted only of stop-tokens (e.g. 'FC'): fall back to the raw text
        key = re.sub(r"[^a-z0-9 ]+", " ", _strip_accents(str(name)).lower()).strip()
    return TEAM_ALIASES.get(key, key)


def team_id_from_name(name: str) -> str:
    """URL-safe stable identifier, e.g. 'FC Bayern München' and 'Bayern München' -> 'bayern-munchen'."""
    return canonical_team_key(name).replace(" ", "-")


def find_possible_duplicate_teams(team_ids: Iterable[str], threshold: float = 0.82) -> list[tuple[str, str, float]]:
    """Pairs of distinct team ids whose spelling is suspiciously similar (candidates for TEAM_ALIASES)."""
    from difflib import SequenceMatcher

    ids = sorted(set(team_ids))
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            ratio = SequenceMatcher(None, a, b).ratio()
            if ratio >= threshold:
                out.append((a, b, round(ratio, 3)))
    return out


def pick_display_names(rows: Iterable[tuple[str, str, object]]) -> Mapping[str, str]:
    """``rows`` = (team_id, raw_name, sort_key). The raw name with the latest sort_key wins."""
    best: dict[str, tuple[object, str]] = {}
    for team_id, raw, key in rows:
        if team_id not in best or key >= best[team_id][0]:
            best[team_id] = (key, strip_country_suffix(raw))
    return {k: v[1] for k, v in best.items()}
