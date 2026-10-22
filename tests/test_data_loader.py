"""Data acquisition tests. No test touches the network: HTTP is replaced by scripted fakes."""
import json

import numpy as np
import pandas as pd
import pytest
import requests

from src import data_loader as dl
from src.data_loader import (
    AuthError, DataSourceError, DataValidationError, FootballDataClient, FootballDataSource, MissingTokenError,
    NotFoundError, OpenFootballSource, RateLimiter, RateLimitError, RestrictedResourceError, SchemaError, TransientError,
    as_of, clean_matches, load_matches, normalize_football_data_matches, parse_openfootball_text, read_processed,
    save_processed, split_completed_and_fixtures,
)
from src.utils import ProjectPaths, Stage

from .helpers import make_matches, real_season

TOKEN = "super-secret-token-123"


# ------------------------------------------------------------------ fakes
class FakeResponse:
    def __init__(self, status=200, body=None, headers=None, text=None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")
        self.encoding = "utf-8"

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    """Plays back a script of responses / exceptions, recording every call."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(script, **kw):
    sleeps = []
    c = FootballDataClient(TOKEN, session=FakeSession(script), sleep=sleeps.append, backoff_base=2.0, **kw)
    return c, sleeps


OK_PAYLOAD = {"matches": [], "resultSet": {}}


# ------------------------------------------------------------------ token handling
def test_missing_token_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv(dl.TOKEN_ENV_VAR, raising=False)
    monkeypatch.setattr(dl, "_read_dotenv", lambda path: {})
    with pytest.raises(MissingTokenError, match="FOOTBALL_DATA_API_TOKEN"):
        FootballDataClient()


def test_token_precedence_explicit_then_env_then_dotenv(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nFOOTBALL_DATA_API_TOKEN='from-file'\nOTHER=1\n", encoding="utf-8")
    monkeypatch.delenv(dl.TOKEN_ENV_VAR, raising=False)
    assert dl.get_api_token(dotenv_path=env_file) == "from-file"
    monkeypatch.setenv(dl.TOKEN_ENV_VAR, "from-env")
    assert dl.get_api_token(dotenv_path=env_file) == "from-env"
    assert dl.get_api_token("explicit", dotenv_path=env_file) == "explicit"
    assert dl.get_api_token("   ", dotenv_path=env_file) == "from-env"  # blank explicit token is ignored


def test_the_token_is_sent_as_a_header_and_never_leaks_into_errors():
    c, _ = client([FakeResponse(403, {"message": "restricted"})])
    with pytest.raises(RestrictedResourceError) as exc:
        c.get_matches(2019)
    assert TOKEN not in str(exc.value)
    assert c.session.calls[0]["headers"]["X-Auth-Token"] == TOKEN
    assert "token" not in c.session.calls[0]["url"].lower()


# ------------------------------------------------------------------ client: success, retries, errors
def test_successful_request_sends_timeout_and_season_param():
    c, sleeps = client([FakeResponse(200, OK_PAYLOAD)])
    payload, url = c.get_matches(2026)
    assert payload == OK_PAYLOAD and url.endswith("/competitions/CL/matches?season=2026")
    call = c.session.calls[0]
    assert call["params"] == {"season": 2026} and call["timeout"] == (5.0, 30.0)
    assert sleeps == []


def test_rate_limit_429_is_retried_after_the_servers_hint():
    msg = {"message": "You reached your request limit. Wait 24 seconds.", "errorCode": 429}
    c, sleeps = client([FakeResponse(429, msg), FakeResponse(200, OK_PAYLOAD)])
    payload, _ = c.get_matches(2025)
    assert payload == OK_PAYLOAD and len(c.session.calls) == 2
    assert sleeps == [25.0]  # 24 s + 1 s margin


def test_retry_after_header_wins_over_the_message():
    c, sleeps = client([FakeResponse(429, {"message": "slow down"}, headers={"Retry-After": "7"}), FakeResponse(200, OK_PAYLOAD)])
    c.get_matches(2025)
    assert sleeps == [8.0]


def test_persistent_rate_limit_raises_a_specific_error():
    c, _ = client([FakeResponse(429, {"message": "Wait 1 seconds."})] * 3, max_retries=2)
    with pytest.raises(RateLimitError):
        c.get_matches(2025)
    assert len(c.session.calls) == 3


def test_server_errors_back_off_exponentially_then_succeed():
    c, sleeps = client([FakeResponse(503, {}), FakeResponse(500, {}), FakeResponse(200, OK_PAYLOAD)])
    c.get_matches(2025)
    assert sleeps == [2.0, 4.0]


def test_persistent_server_errors_raise_transient_error():
    c, _ = client([FakeResponse(502, {})] * 3, max_retries=2)
    with pytest.raises(TransientError, match="502"):
        c.get_matches(2025)


@pytest.mark.parametrize("exc", [requests.Timeout("slow"), requests.ConnectionError("down")])
def test_network_failures_are_retried_then_reported(exc):
    c, sleeps = client([exc, exc, FakeResponse(200, OK_PAYLOAD)])
    c.get_matches(2025)
    assert len(sleeps) == 2
    c2, _ = client([exc] * 3, max_retries=2)
    with pytest.raises(TransientError, match="Giving up"):
        c2.get_matches(2025)


@pytest.mark.parametrize("status,error", [(400, AuthError), (401, AuthError), (403, RestrictedResourceError),
                                          (404, NotFoundError), (418, DataSourceError)])
def test_client_errors_are_not_retried_and_are_typed(status, error):
    c, sleeps = client([FakeResponse(status, {"message": "nope"})])
    with pytest.raises(error, match="nope"):
        c.get_matches(2025)
    assert len(c.session.calls) == 1 and sleeps == []


def test_bad_payloads_raise_schema_errors():
    c, _ = client([FakeResponse(200, None)])  # body is not JSON
    with pytest.raises(SchemaError, match="not valid JSON"):
        c.get_matches(2025)
    c, _ = client([FakeResponse(200, {"unexpected": True})])
    with pytest.raises(SchemaError, match="no 'matches' list"):
        c.get_matches(2025)


def test_verify_access_summarises_the_competition():
    comp = {"code": "CL", "name": "UEFA Champions League", "currentSeason": {"startDate": "2026-07-01"},
            "seasons": [{"startDate": "2024-07-01"}, {"startDate": "2025-07-01"}, {"startDate": "2026-07-01"}]}
    c, _ = client([FakeResponse(200, comp, headers={"X-Requests-Available-Minute": "9"})])
    info = c.verify_access()
    assert info["ok"] and info["seasons_listed_by_provider"] == [2024, 2025, 2026] and info["requests_available_minute"] == "9"


def test_local_rate_limiter_sleeps_when_the_window_is_full():
    clock, slept = [0.0], []
    limiter = RateLimiter(max_calls=3, period=60.0, monotonic=lambda: clock[0], sleep=lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s)))
    for _ in range(3):
        assert limiter.wait() == 0.0
    waited = limiter.wait()  # 4th call inside the minute
    assert waited == pytest.approx(60.05) and slept == [waited]
    assert limiter.wait() == 0.0


# ------------------------------------------------------------------ normalisation of provider payloads
def _team(i, name):
    return {"id": i, "name": name}


def _m(i, status="FINISHED", stage="LEAGUE_STAGE", home=(1, "Alpha FC"), away=(2, "Beta FC"), score=None, date="2025-09-17T19:00:00Z", **kw):
    return {"id": i, "utcDate": date, "status": status, "stage": stage, "matchday": 1, "group": None,
            "homeTeam": _team(*home), "awayTeam": _team(*away), "score": score, **kw}


def _score(duration="REGULAR", ft=(1, 0), rt=None, et=None, pen=None, winner="HOME_TEAM", keys=("home", "away")):
    node = lambda t: None if t is None else {keys[0]: t[0], keys[1]: t[1]}  # noqa: E731
    s = {"winner": winner, "duration": duration, "fullTime": node(ft), "halfTime": node((0, 0))}
    if rt is not None:
        s["regularTime"] = node(rt)
    if et is not None:
        s["extraTime"] = node(et)
    if pen is not None:
        s["penalties"] = node(pen)
    return s


def test_normalisation_covers_the_awkward_cases():
    payload = {"matches": [
        _m(1, score=_score(ft=(2, 1))),
        _m(2, stage="LAST_16", score=_score("EXTRA_TIME", ft=(3, 2), rt=(2, 2), et=(1, 0))),
        _m(3, stage="LAST_16", score=_score("PENALTY_SHOOTOUT", ft=(6, 5), et=(0, 0), pen=(5, 4))),   # no regularTime: derive
        _m(4, stage="LAST_16", score=_score("PENALTY_SHOOTOUT", ft=(6, 5), pen=(5, 4))),              # underivable
        _m(5, score=_score(ft=None)),                                                                   # FINISHED, no score
        _m(6, status="POSTPONED", score=_score(ft=None, winner=None)),
        _m(7, status="TIMED", home=(None, None), away=(9, "Gamma FC"), score=_score(ft=None, winner=None), stage="LAST_16"),
        _m(8, status="IN_PLAY", score=_score(ft=(1, 0))),                                               # live: must not leak
        _m(9, score=_score(ft=(4, 4), keys=("homeTeam", "awayTeam"))),                                  # legacy key names
        _m(10, stage="QUALIFICATION_ROUND_1", score=_score(ft=(1, 1), winner="DRAW")),
        _m(11, stage="FINAL", score=_score(ft=(0, 1), winner="AWAY_TEAM")),
        {"utcDate": "2025-01-01T00:00:00Z"},                                                            # malformed: no id
    ]}
    df = normalize_football_data_matches(payload, season=2025, retrieved_at="2026-01-01T00:00:00+00:00")
    by = df.set_index("source_match_id")
    assert len(df) == 11 and by.loc["1", "home_goals_90"] == 2 and by.loc["1", "is_completed"]
    assert (by.loc["2", "home_goals_90"], by.loc["2", "away_goals_90"]) == (2, 2)          # regularTime, not fullTime
    assert (by.loc["2", "home_goals_aet"], by.loc["2", "away_goals_aet"]) == (3, 2)
    assert by.loc["2", "decided_by"] == "EXTRA_TIME"
    assert (by.loc["3", "home_goals_90"], by.loc["3", "away_goals_90"]) == (1, 1)          # 6-5 minus shoot-out 5-4, minus ET 0-0
    assert (by.loc["3", "home_goals_aet"], by.loc["3", "away_goals_aet"]) == (1, 1)
    assert (by.loc["3", "pens_home"], by.loc["3", "pens_away"]) == (5, 4)
    assert not by.loc["4", "is_completed"] and np.isnan(by.loc["4", "home_goals_90"])      # never guessed
    assert not by.loc["5", "is_completed"]
    assert by.loc["6", "status"] == "POSTPONED" and not by.loc["6", "is_completed"]
    assert pd.isna(by.loc["7", "home_team_id"]) and by.loc["7", "away_team_id"] == "gamma"
    assert not by.loc["8", "is_completed"] and np.isnan(by.loc["8", "home_goals_90"])
    assert by.loc["9", "home_goals_90"] == 4 and by.loc["9", "away_goals_90"] == 4
    assert by.loc["10", "stage"] == Stage.OTHER
    assert by.loc["11", "stage"] == Stage.FINAL and bool(by.loc["11", "neutral"])
    assert (df["source"] == "football-data.org").all() and (df["retrieved_at"] == "2026-01-01T00:00:00+00:00").all()


def test_regular_season_stage_means_league_phase_only_from_2024():
    payload = {"matches": [_m(1, stage="REGULAR_SEASON", score=_score())]}
    assert normalize_football_data_matches(payload, season=2025)["stage"].iloc[0] == Stage.LEAGUE
    assert normalize_football_data_matches(payload, season=2019)["stage"].iloc[0] == Stage.OTHER


def test_normalisation_rejects_a_payload_without_matches():
    with pytest.raises(SchemaError):
        normalize_football_data_matches({"nothing": 1})


def test_schema_probe_reports_unmapped_stages_and_missing_fields():
    payload = {"matches": [_m(1, stage="WEIRD_STAGE", score=_score()), {"id": 2, "utcDate": "x"}]}
    info = dl.describe_football_data_payload(payload)
    assert info["unmapped_stages"] == ["None", "WEIRD_STAGE"]
    assert "homeTeam" in info["missing_required"] and info["n_matches"] == 2


# ------------------------------------------------------------------ FootballDataSource: caching & unavailable seasons
class FakeClient:
    def __init__(self, behaviour):
        self.behaviour, self.calls = behaviour, []

    def get_matches(self, season, code="CL"):
        self.calls.append(season)
        result = self.behaviour(season)
        if isinstance(result, Exception):
            raise result
        return result, f"https://example.test/{season}"


def _season_payload(season):
    return {"matches": [_m(season * 10 + 1, score=_score(ft=(1, 0))), _m(season * 10 + 2, home=(3, "Gamma FC"), away=(4, "Delta FC"), score=_score(ft=(2, 2), winner="DRAW"))]}


def test_snapshots_avoid_repeat_requests_and_can_be_refreshed(tmp_path):
    fake = FakeClient(_season_payload)
    src = FootballDataSource(fake, paths=ProjectPaths(tmp_path))
    first = src.load([2023, 2024])
    assert fake.calls == [2023, 2024] and first.seasons_loaded == [2023, 2024] and not any(first.from_cache.values())
    second = src.load([2023, 2024])
    assert fake.calls == [2023, 2024] and all(second.from_cache.values())  # served from disk
    assert len(second.matches) == 4
    meta = json.loads((tmp_path / "data" / "raw" / "football_data" / "CL_2023.json.meta.json").read_text())
    assert meta["source"] == "football-data.org" and meta["retrieved_at"] and meta["sha256"]
    src.load([2023], refresh=True)
    assert fake.calls == [2023, 2024, 2023]


def test_unavailable_seasons_are_reported_not_silently_dropped(tmp_path):
    def behaviour(season):
        return RestrictedResourceError("403 plan") if season == 2015 else NotFoundError("404") if season == 2030 else _season_payload(season)

    res = FootballDataSource(FakeClient(behaviour), paths=ProjectPaths(tmp_path)).load([2015, 2024, 2030])
    assert res.seasons_loaded == [2024]
    assert set(res.unavailable) == {2015, 2030} and "RestrictedResourceError" in res.unavailable[2015]


def test_authentication_failure_aborts_instead_of_trying_every_season(tmp_path):
    fake = FakeClient(lambda s: AuthError("bad token"))
    with pytest.raises(AuthError):
        FootballDataSource(fake, paths=ProjectPaths(tmp_path)).load([2023, 2024])
    assert fake.calls == [2023]


def test_a_transient_failure_on_one_season_is_recorded_and_others_still_load(tmp_path):
    fake = FakeClient(lambda s: TransientError("boom") if s == 2023 else _season_payload(s))
    res = FootballDataSource(fake, paths=ProjectPaths(tmp_path)).load([2023, 2024])
    assert res.seasons_loaded == [2024] and "TransientError" in res.unavailable[2023]


def test_a_corrupt_snapshot_is_refetched_instead_of_crashing(tmp_path):
    paths = ProjectPaths(tmp_path)
    dl.RawSnapshotStore(paths.data_raw).save("football_data", "CL_2024", "{this is not json", "https://example.test")
    fake = FakeClient(_season_payload)
    res = FootballDataSource(fake, paths=paths).load([2024])
    assert fake.calls == [2024] and res.seasons_loaded == [2024] and res.from_cache == {2024: False}
    body, _ = dl.RawSnapshotStore(paths.data_raw).load("football_data", "CL_2024")
    assert json.loads(body)["matches"]  # the bad file was replaced by a good one


# ------------------------------------------------------------------ openfootball parsing
SNIPPET = """= UEFA Champions League 2024/25

▪ League, Matchday 1
  Tue Sep 17 2024
    18:45  Alpha FC (ENG)    v Beta FC (ESP)        2-1 (1-0)
           Gamma (ITA)       v Delta (GER)          0-0
    21:00  Echo (FRA)        v Foxtrot (POR)        3-3 (2-1)
  Wed Dec 11
    21:00  Beta FC (ESP)     v Echo (FRA)           1-0 (0-0)
  Wed Jan 29
    21:00  Delta (GER)       v Alpha FC (ENG)       2-2
▪ Playoffs, Matchday 2
  Tue Feb 25
    21:00  Gamma (ITA)       v Beta FC (ESP)        3-2 a.e.t. (2-2, 1-1)
▪ Finals, Final
  Sat May 31
    21:00  Alpha FC (ENG)    v Echo (FRA)           4-3 pen. 1-1 a.e.t. (1-1, 0-1)
  Sun Jun 1
    21:00  Gamma (ITA)       v Delta (GER)
"""


def _one(df, home, away, stage):
    sub = df[(df.home_team == home) & (df.away_team == away) & (df.stage == stage)]
    assert len(sub) == 1
    return sub.iloc[0]


def test_openfootball_parser_reads_every_score_notation():
    df = parse_openfootball_text(SNIPPET, 2024, retrieved_at="r")
    assert df.attrs["parse_issues"] == []
    plain = _one(df, "Alpha FC", "Beta FC", Stage.LEAGUE)
    assert (plain.home_goals_90, plain.away_goals_90, plain.decided_by, plain.winner) == (2, 1, "REGULAR", "HOME_TEAM")
    no_ht = _one(df, "Gamma", "Delta", Stage.LEAGUE)  # score without a half-time part
    assert no_ht.is_completed and (no_ht.home_goals_90, no_ht.away_goals_90, no_ht.winner) == (0, 0, "DRAW")
    et = _one(df, "Gamma", "Beta FC", Stage.PLAYOFF)
    assert (et.home_goals_90, et.away_goals_90, et.home_goals_aet, et.away_goals_aet) == (2, 2, 3, 2)  # 90' vs after ET
    assert et.decided_by == "EXTRA_TIME" and et.winner == "HOME_TEAM"
    pens = _one(df, "Alpha FC", "Echo", Stage.FINAL)
    assert (pens.home_goals_90, pens.home_goals_aet, pens.pens_home, pens.pens_away) == (1, 1, 4, 3)
    assert pens.decided_by == "PENALTY_SHOOTOUT" and pens.winner == "HOME_TEAM" and pens.neutral
    unplayed = _one(df, "Gamma", "Delta", Stage.FINAL)
    assert not unplayed.is_completed and np.isnan(unplayed.home_goals_90) and unplayed.status == "SCHEDULED"


def test_openfootball_dates_roll_over_the_new_year_and_inherit_times():
    df = parse_openfootball_text(SNIPPET, 2024)
    league = df[df.stage == Stage.LEAGUE].sort_values("utc_date")
    assert league["utc_date"].dt.year.tolist() == [2024, 2024, 2024, 2024, 2025]
    # continuation line inherits 18:45 CEST == 16:45 UTC; 21:00 CEST == 19:00 UTC; winter (CET) shifts by one hour
    gamma_delta = df[(df.home_team == "Gamma") & (df.away_team == "Delta") & (df.stage == Stage.LEAGUE)].iloc[0]
    assert gamma_delta.utc_date == pd.Timestamp("2024-09-17 16:45")
    assert df[(df.home_team == "Beta FC") & (df.stage == Stage.LEAGUE)].iloc[0].utc_date == pd.Timestamp("2024-12-11 20:00")


def test_group_sections_restart_in_september():
    text = "= X\n▪ Group A\n  Wed Sep 14 2011\n    20:45  A (ENG) v B (ESP)  1-0 (0-0)\n  Wed Dec 7\n    20:45  B (ESP) v A (ENG)  0-0\n" \
           "▪ Group B\n  Wed Sep 14\n    20:45  C (FRA) v D (GER)  2-2 (1-1)\n"
    df = parse_openfootball_text(text, 2011)
    assert df["utc_date"].dt.year.tolist() == [2011, 2011, 2011] and (df["stage"] == Stage.GROUP).all()
    assert df["group"].tolist() == ["A", "A", "B"]


def test_unparseable_lines_are_reported_not_dropped_silently():
    text = "= X\n▪ League, Matchday 1\n  Tue Sep 17 2024\n    21:00  A (ENG) v B (ESP)  banana\n    totally broken line\n"
    df = parse_openfootball_text(text, 2024)
    msgs = [m for _, m in df.attrs["parse_issues"]]
    assert any("unparseable score" in m for m in msgs) and any("unrecognised line" in m for m in msgs)
    assert len(df) == 1 and not df.iloc[0].is_completed  # kept as an unplayed fixture, not as a fake result


def test_extra_time_without_a_90_minute_score_is_not_completed():
    text = "= X\n▪ Finals, Round of 16\n  Tue Mar 10 2020\n    21:00  A (ENG) v B (ESP)  2-1 a.e.t.\n"
    df = parse_openfootball_text(text, 2019)
    assert not df.iloc[0].is_completed and df.attrs["parse_issues"]


# ------------------------------------------------------------------ real (CC0) fixtures
def test_real_2025_26_season_parses_completely():
    df = real_season(2025)
    assert len(df) == 189 and df.attrs["parse_issues"] == [] and df["is_completed"].all()
    assert df["stage"].value_counts().to_dict() == {Stage.LEAGUE: 144, Stage.PLAYOFF: 16, Stage.LAST_16: 16,
                                                    Stage.QUARTER: 8, Stage.SEMI: 4, Stage.FINAL: 1}
    final = df[df["stage"] == Stage.FINAL].iloc[0]
    assert (final.home_team, final.away_team) == ("Paris Saint-Germain FC", "Arsenal FC")
    assert (final.home_goals_90, final.away_goals_90, final.pens_home, final.pens_away) == (1, 1, 4, 3)
    # 2nd-leg extra-time match: Juventus 3-2 Galatasaray after ET, 3-0 after 90 minutes
    et = df[(df.home_team == "Juventus FC") & (df.stage == Stage.PLAYOFF)].iloc[0]
    assert (et.home_goals_90, et.away_goals_90, et.home_goals_aet, et.away_goals_aet) == (3, 0, 3, 2)


def test_every_league_team_plays_eight_matches_in_the_real_data():
    league = real_season(2024).query("stage == 'LEAGUE_PHASE'")
    teams = pd.concat([league["home_team_id"], league["away_team_id"]]).value_counts()
    assert len(teams) == 36 and (teams == 8).all()


# ------------------------------------------------------------------ as_of / leakage
def test_as_of_hides_every_result_from_the_cutoff_onwards():
    df = real_season(2025)
    cutoff = pd.Timestamp("2025-11-01")
    seen = as_of(df, cutoff)
    assert (seen.loc[seen["is_completed"], "utc_date"] < cutoff).all()
    hidden = seen[seen["utc_date"] >= cutoff]
    assert hidden[["home_goals_90", "away_goals_90", "pens_home", "winner", "decided_by"]].isna().all().all()
    assert (hidden["status"] == "SCHEDULED").all() and len(seen) == len(df)  # fixtures remain
    done, fixtures = split_completed_and_fixtures(seen)
    assert len(done) + len(fixtures) == len(df) and 0 < len(done) < len(df)


def test_as_of_does_not_modify_its_input():
    df = real_season(2025)
    before = df.copy()
    as_of(df, "2025-11-01", mask_undetermined_knockout=True)
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("cutoff,known_by_stage", [
    ("2025-09-01", {Stage.LEAGUE: 144, Stage.PLAYOFF: 0, Stage.LAST_16: 0, Stage.QUARTER: 0, Stage.SEMI: 0, Stage.FINAL: 0}),
    ("2026-01-30", {Stage.LEAGUE: 144, Stage.PLAYOFF: 16, Stage.LAST_16: 0, Stage.QUARTER: 0, Stage.SEMI: 0, Stage.FINAL: 0}),
    ("2026-03-01", {Stage.LEAGUE: 144, Stage.PLAYOFF: 16, Stage.LAST_16: 16, Stage.QUARTER: 0, Stage.SEMI: 0, Stage.FINAL: 0}),
    ("2026-04-10", {Stage.LEAGUE: 144, Stage.PLAYOFF: 16, Stage.LAST_16: 16, Stage.QUARTER: 8, Stage.SEMI: 0, Stage.FINAL: 0}),
    ("2026-05-10", {Stage.LEAGUE: 144, Stage.PLAYOFF: 16, Stage.LAST_16: 16, Stage.QUARTER: 8, Stage.SEMI: 4, Stage.FINAL: 1}),
])
def test_knockout_participants_are_hidden_until_they_were_determined(cutoff, known_by_stage):
    seen = as_of(real_season(2025), cutoff, mask_undetermined_knockout=True)
    known = seen[seen["home_team_id"].notna()].groupby("stage").size().to_dict()
    assert {k: known.get(k, 0) for k in known_by_stage} == known_by_stage


def test_league_fixtures_stay_visible_in_a_replay_because_the_draw_is_made_beforehand():
    seen = as_of(real_season(2025), "2025-09-01", mask_undetermined_knockout=True)
    assert seen[seen["stage"] == Stage.LEAGUE]["home_team_id"].notna().all()
    assert not seen["is_completed"].any()


# ------------------------------------------------------------------ cleaning: duplicates, missing data
def _dirty():
    rows = [{"home": "a", "away": "b", "hg": 1, "ag": 0, "date": "2020-01-01", "match_id": "x1"},
            {"home": "a", "away": "b", "hg": 1, "ag": 0, "date": "2020-01-01", "match_id": "x1"},          # same id twice
            {"home": "c", "away": "d", "hg": 2, "ag": 2, "date": "2020-01-02", "match_id": "x2"},
            {"home": "c", "away": "d", "hg": 2, "ag": 2, "date": "2020-01-02 15:00", "match_id": "x3"},     # same fixture, other id
            {"home": "e", "away": "e", "hg": 1, "ag": 1, "date": "2020-01-03", "match_id": "x4"},          # team vs itself
            {"home": "f", "away": "g", "hg": 1, "ag": 0, "date": "2020-01-04", "match_id": "x5"}]
    df = make_matches(rows)
    neg = make_matches([{"home": "h", "away": "i", "hg": -1, "ag": 0, "date": "2020-01-05", "match_id": "x6"}])
    miss = make_matches([{"home": "j", "away": "k", "hg": 1, "ag": 0, "date": "2020-01-06", "match_id": "x7"}])
    miss.loc[0, "home_goals_90"] = np.nan
    return pd.concat([df, neg, miss], ignore_index=True)


def test_clean_matches_removes_duplicates_and_impossible_rows_and_says_so():
    clean, report = clean_matches(_dirty())
    assert clean["match_id"].is_unique
    assert set(clean["match_id"]) == {"x1", "x3", "x5"}  # of the duplicated fixture the later row wins
    assert report.duplicates_dropped == 2 and report.dropped_invalid == 3
    assert any("identical" in w for w in report.warnings) and any("negative/missing" in w for w in report.warnings)
    assert report.n_completed == len(clean)


def test_clean_matches_rejects_a_table_missing_columns():
    with pytest.raises(DataValidationError, match="missing required columns"):
        clean_matches(pd.DataFrame({"match_id": ["a"]}))


def test_empty_input_is_handled():
    clean, report = clean_matches(pd.DataFrame(columns=dl.MATCH_COLUMNS))
    assert clean.empty and report.n_rows == 0


def test_processed_csv_roundtrip(tmp_path):
    df, _ = clean_matches(real_season(2025))
    path = save_processed(df, tmp_path / "out" / "matches.csv")
    back = read_processed(path).reset_index(drop=True)
    df = df.reset_index(drop=True)
    assert list(back.columns) == list(df.columns) and len(back) == len(df)
    for col in df.columns:  # same missing values, same present values (None vs NaN is not a difference)
        assert (df[col].isna() == back[col].isna()).all(), col
        present = df[col].notna()
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            np.testing.assert_allclose(df.loc[present, col].astype(float), back.loc[present, col].astype(float))
        else:
            assert (df.loc[present, col].astype(str) == back.loc[present, col].astype(str)).all(), col


def test_team_directory_uses_the_latest_spelling():
    df = pd.concat([real_season(2024), real_season(2025)], ignore_index=True)
    directory = dl.build_team_directory(df).set_index("team_id")
    assert directory.loc["inter", "team_name"] == "FC Internazionale Milano"
    assert directory.loc["arsenal", "first_season"] == 2024 and len(directory) >= 36


# ------------------------------------------------------------------ source orchestration
class StubSource:
    def __init__(self, name, available, token_ok=True):
        self.name, self.available, self.token_ok, self.requested = name, available, token_ok, []

    def load(self, seasons, refresh=False):
        self.requested.append(list(seasons))
        if not self.token_ok:
            raise MissingTokenError("no token")
        got = [s for s in seasons if s in self.available]
        frames = [make_matches([{"home": "a", "away": "b", "hg": 1, "ag": 0, "date": f"{s}-10-01", "season": s,
                                 "match_id": f"{self.name}:{s}"}]) for s in got]
        return dl.LoadResult(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=dl.MATCH_COLUMNS),
                             self.name, "now", got, {s: "n/a" for s in seasons if s not in got})


def test_sources_are_tried_in_priority_order_with_per_season_fallback():
    primary, fallback = StubSource("api", {2025, 2026}), StubSource("files", {2020, 2021, 2025})
    df, prov = load_matches([primary, fallback], [2020, 2021, 2025, 2026])
    assert prov["season_source"] == {2020: "files", 2021: "files", 2025: "api", 2026: "api"}
    assert fallback.requested == [[2020, 2021]]  # the fallback is only asked for what the primary could not supply
    assert prov["seasons_not_loaded"] == []
    assert set(df["season"]) == {2020, 2021, 2025, 2026}


def test_missing_token_falls_through_to_the_next_source_and_is_reported():
    df, prov = load_matches([StubSource("api", set(), token_ok=False), StubSource("files", {2024})], [2024])
    assert prov["season_source"] == {2024: "files"} and "api" in prov["unavailable"]


def test_no_data_anywhere_is_a_loud_error_not_an_empty_table():
    with pytest.raises(DataSourceError, match="No Champions League data"):
        load_matches([StubSource("api", set()), StubSource("files", set())], [2024])


def test_openfootball_source_downloads_caches_and_reports_missing_seasons(tmp_path):
    text = SNIPPET
    session = FakeSession([FakeResponse(200, None, text=text), FakeResponse(404, None)])
    src = OpenFootballSource(paths=ProjectPaths(tmp_path), session=session, sleep=lambda s: None)
    res = src.load([2024, 2026])
    assert res.seasons_loaded == [2024] and "NotFoundError" in res.unavailable[2026]
    again = OpenFootballSource(paths=ProjectPaths(tmp_path), session=FakeSession([]), sleep=lambda s: None).load([2024])
    assert again.from_cache == {2024: True} and len(again.matches) == len(res.matches)  # no HTTP needed the 2nd time


def test_openfootball_retries_server_errors(tmp_path):
    session = FakeSession([FakeResponse(503, None), requests.Timeout("t"), FakeResponse(200, None, text=SNIPPET)])
    slept = []
    res = OpenFootballSource(paths=ProjectPaths(tmp_path), session=session, sleep=slept.append).load([2024])
    assert res.seasons_loaded == [2024] and len(slept) == 2
