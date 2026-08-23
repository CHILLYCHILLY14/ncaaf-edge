"""
ESPN public data client.

ESPN exposes an undocumented but stable, keyless, free JSON API. Everything this
project needs -- the full FBS schedule, live and final scores, venue/neutral-site
flags, and pregame odds -- comes from two endpoints:

  scoreboard : schedule + scores + (for games not yet final) an `odds` array
  summary    : per-game detail, whose `pickcenter` block keeps odds around
               AFTER a game finishes, which the scoreboard drops.

That second point matters a lot. Closing lines vanish from the scoreboard the
moment a game goes final, so a pipeline that only reads the scoreboard can never
grade a bet it saw on Friday. We snapshot every line we see, every run, and fall
back to `pickcenter` when the scoreboard has moved on.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any, Iterable

import requests

SITE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ncaaf-edge/2.0; +https://github.com/)",
    "Accept": "application/json",
}


class EspnError(RuntimeError):
    pass


def _get(url: str, params: dict | None = None, tries: int = 4) -> dict:
    """GET with polite backoff. ESPN rate-limits softly; a few retries clears it."""
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except requests.RequestException as exc:  # network blip
            last = str(exc)
        time.sleep(1.5 * (attempt + 1))
    raise EspnError(f"GET {url} failed after {tries} tries: {last}")


def scoreboard(date: dt.date, group: int = 80, limit: int = 400) -> dict:
    """One calendar day of games. `group=80` is FBS."""
    return _get(
        f"{SITE}/scoreboard",
        {"dates": date.strftime("%Y%m%d"), "groups": group, "limit": limit},
    )


def summary(event_id: str) -> dict:
    return _get(f"{SITE}/summary", {"event": event_id})


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, str):
        v = v.strip()
        if v.upper() == "OFF":
            return None
        if v.upper() == "EVEN":
            return 100.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _line_num(v: Any) -> float | None:
    """Parse a line such as ``-3.5``, ``o51.5`` or ``u51.5``."""
    if isinstance(v, str):
        v = v.strip().lower()
        if v.startswith(("o", "u")):
            v = v[1:]
    return _num(v)


def _close(block: dict, market: str, side: str) -> dict:
    return ((((block.get(market) or {}).get(side) or {}).get("close")) or {})


def _pick_odds_block(odds_list: Iterable[dict], priority: list[str]) -> dict | None:
    """
    ESPN can return several providers. Prefer the ones the user actually bets
    into, in the order given in settings; otherwise take whatever came first.
    """
    blocks = list(odds_list or [])
    if not blocks:
        return None
    normalise = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())
    by_name = {}
    for b in blocks:
        name = ((b.get("provider") or {}).get("name") or "").strip()
        by_name.setdefault(normalise(name), b)
    for want in priority:
        hit = by_name.get(normalise(want))
        if hit:
            return hit
    return blocks[0]


def parse_odds(block: dict | None) -> dict:
    """
    Normalise one ESPN odds block into the fields the model needs.

    ESPN's `spread` is always stated from the HOME team's perspective
    (negative = home favoured), which is the same convention the workbook used,
    so it carries across unchanged.
    """
    if not block:
        return {}
    away = block.get("awayTeamOdds") or {}
    home = block.get("homeTeamOdds") or {}

    def _legacy_ml(side: dict) -> float | None:
        for key in ("moneyLine", "moneyline"):
            v = _num(side.get(key))
            if v is not None:
                return v
        cur = side.get("current") or {}
        return _num((cur.get("moneyLine") or {}).get("american"))

    def _legacy_spread_price(side: dict) -> float | None:
        cur = side.get("current") or {}
        return _num((cur.get("pointSpread") or {}).get("american"))

    ml_home = _num(_close(block, "moneyline", "home").get("odds"))
    ml_away = _num(_close(block, "moneyline", "away").get("odds"))
    if ml_home is None:
        ml_home = _legacy_ml(home)
    if ml_away is None:
        ml_away = _legacy_ml(away)

    spread_home = _line_num(_close(block, "pointSpread", "home").get("line"))
    if spread_home is None:
        spread_home = _num(block.get("spread"))
    spread_price_home = _num(_close(block, "pointSpread", "home").get("odds"))
    spread_price_away = _num(_close(block, "pointSpread", "away").get("odds"))
    if spread_price_home is None:
        spread_price_home = _legacy_spread_price(home)
    if spread_price_away is None:
        spread_price_away = _legacy_spread_price(away)

    total = _num(block.get("overUnder"))
    if total is None:
        total = _line_num(_close(block, "total", "over").get("line"))
    over_price = _num(_close(block, "total", "over").get("odds"))
    under_price = _num(_close(block, "total", "under").get("odds"))
    cur = block.get("current") or {}
    if over_price is None:
        over_price = _num(((cur.get("over") or {}).get("american")))
    if under_price is None:
        under_price = _num(((cur.get("under") or {}).get("american")))

    verified = []
    if spread_home is not None and spread_price_home is not None and spread_price_away is not None:
        verified.append("ATS")
    if total is not None and over_price is not None and under_price is not None:
        verified.append("TOTAL")
    if ml_home is not None and ml_away is not None:
        verified.append("ML")

    return {
        "book": ((block.get("provider") or {}).get("name") or "ESPN").strip(),
        "spread_home": spread_home,
        "spread_price_home": spread_price_home,
        "spread_price_away": spread_price_away,
        "total": total,
        "over_price": over_price,
        "under_price": under_price,
        "ml_home": ml_home,
        "ml_away": ml_away,
        "details": block.get("details"),
        "verified_markets": verified,
    }


def has_priced_market(odds: dict | None) -> bool:
    """True only when at least one market has a line and both real prices."""
    o = odds or {}
    verified = set(o.get("verified_markets") or [])
    return bool(verified & {"ML", "ATS", "TOTAL"})


def odds_health(games: Iterable[dict]) -> dict:
    """Compact integrity report for the dashboard and a flat-price tripwire."""
    rows = [g.get("odds") or {} for g in games]
    counts = {m: sum(m in set(o.get("verified_markets") or []) for o in rows)
              for m in ("ML", "ATS", "TOTAL")}
    prices = []
    for o in rows:
        markets = set(o.get("verified_markets") or [])
        if "ML" in markets:
            prices.extend((o.get("ml_home"), o.get("ml_away")))
        if "ATS" in markets:
            prices.extend((o.get("spread_price_home"), o.get("spread_price_away")))
        if "TOTAL" in markets:
            prices.extend((o.get("over_price"), o.get("under_price")))
    prices = [float(p) for p in prices if p is not None]
    unique = sorted(set(prices))
    flat = len(prices) >= 20 and len(unique) == 1
    return {
        "healthy": not flat,
        "games_checked": len(rows),
        "games_with_verified_prices": sum(has_priced_market(o) for o in rows),
        "markets": counts,
        "price_observations": len(prices),
        "unique_prices": len(unique),
        "flat_price_warning": flat,
        "message": ("All observed prices are identical; plays are disabled until the feed is checked."
                    if flat else "Only complete two-sided sportsbook prices are actionable."),
    }


def parse_event(ev: dict, odds_priority: list[str]) -> dict | None:
    """Flatten one ESPN event into our internal game record."""
    comps = ev.get("competitions") or []
    if not comps:
        return None
    c = comps[0]
    competitors = c.get("competitors") or []
    home = next((x for x in competitors if x.get("homeAway") == "home"), None)
    away = next((x for x in competitors if x.get("homeAway") == "away"), None)
    if not home or not away:
        return None

    status = ((c.get("status") or ev.get("status") or {}).get("type") or {})
    venue = c.get("venue") or {}
    addr = venue.get("address") or {}

    def team(side: dict) -> dict:
        t = side.get("team") or {}
        return {
            "id": str(t.get("id") or ""),
            "abbr": (t.get("abbreviation") or t.get("shortDisplayName") or "").strip(),
            "name": (t.get("displayName") or t.get("name") or "").strip(),
            "logo": t.get("logo"),
            "color": t.get("color"),
            "conference_id": str(side.get("conferenceId") or ""),
            "rank": (side.get("curatedRank") or {}).get("current"),
            "record": next(
                (r.get("summary") for r in (side.get("records") or []) if r.get("type") in ("total", "overall")),
                None,
            ),
        }

    def score(side: dict) -> int | None:
        v = side.get("score")
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    completed = bool(status.get("completed"))
    state = status.get("state")  # pre | in | post
    status_name = (status.get("name") or "").upper()  # e.g. STATUS_POSTPONED, STATUS_CANCELED
    postponed = "POSTPON" in status_name or "DELAY" in status_name
    canceled = "CANCEL" in status_name or "FORFEIT" in status_name

    return {
        "game_id": str(ev.get("id")),
        "date_utc": ev.get("date"),
        "season": ((ev.get("season") or {}).get("year")),
        "season_type": ((ev.get("season") or {}).get("type")),
        "week": ((ev.get("week") or {}).get("number")),
        "neutral": bool(c.get("neutralSite")),
        "conference_game": bool(c.get("conferenceCompetition")),
        "indoor": bool(venue.get("indoor")),
        "venue": venue.get("fullName"),
        "venue_city": addr.get("city"),
        "venue_state": addr.get("state"),
        "state": state,
        "status_name": status_name,
        "postponed": postponed,
        "canceled": canceled,
        "completed": completed,
        "status_detail": status.get("shortDetail"),
        "home": team(home),
        "away": team(away),
        "home_score": score(home),
        "away_score": score(away),
        "odds": parse_odds(_pick_odds_block(c.get("odds"), odds_priority)),
        "broadcast": next(
            (b.get("names", [None])[0] for b in (c.get("broadcasts") or []) if b.get("names")), None
        ),
    }


def fetch_range(start: dt.date, end: dt.date, group: int, odds_priority: list[str]) -> list[dict]:
    """Every FBS game between two dates, inclusive."""
    out: list[dict] = []
    seen: set[str] = set()
    day = start
    while day <= end:
        try:
            data = scoreboard(day, group=group)
        except EspnError as exc:
            print(f"  ! scoreboard {day}: {exc}")
            day += dt.timedelta(days=1)
            continue
        for ev in data.get("events") or []:
            g = parse_event(ev, odds_priority)
            if g and g["game_id"] not in seen:
                seen.add(g["game_id"])
                out.append(g)
        day += dt.timedelta(days=1)
    return out


def fetch_season(year: int, group: int, odds_priority: list[str]) -> list[dict]:
    """
    A whole season, walked day by day from Aug 1 through Jan 31 of the next year.

    Date-walking beats the per-week event index here: the scoreboard endpoint
    returns every game for a day in one request with scores already attached,
    where the week index hands back one $ref per game and would need hundreds of
    follow-up calls to say the same thing.
    """
    return fetch_range(dt.date(year, 8, 1), dt.date(year + 1, 1, 31), group, odds_priority)


def odds_from_summary(event_id: str, odds_priority: list[str]) -> dict:
    """
    Recover odds for a game the scoreboard has already dropped (i.e. it's final).
    `pickcenter` keeps the closing number around.
    """
    try:
        s = summary(event_id)
    except EspnError:
        return {}
    return parse_odds(_pick_odds_block(s.get("pickcenter") or [], odds_priority))
