"""
Orchestrator. Run this and the whole board rebuilds.

    python -m pipeline.build            # normal scheduled run (rolling window)
    python -m pipeline.build --full     # full-season backfill, rebuilds the cache
    python -m pipeline.build --no-bet   # price everything, log nothing

Sequence: refresh games -> snapshot odds -> re-solve ratings from results ->
project every upcoming game -> price against the market -> tier -> log qualified
bets -> grade finals -> write the JSON the site reads.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

from . import espn, ledger, model as M, predictions as P, ratings as R, store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_DATA = os.path.join(ROOT, "site", "data")


def load_cfg() -> dict:
    with open(os.path.join(ROOT, "config", "settings.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_overrides() -> dict:
    p = os.path.join(ROOT, "config", "overrides.json")
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #

def merge_games(cache: list[dict], fresh: list[dict]) -> list[dict]:
    """
    Fresh data wins, except never let a blank overwrite something we already had.

    Odds specifically: once a game is final ESPN stops returning a line, so a
    naive merge would erase the closing number we need for grading and CLV.
    """
    # Odds written before parser v2 have no verification marker. Those records
    # may contain the old invented -110 defaults, so they are not allowed to
    # survive a refresh as if they were observed sportsbook prices.
    by_id = {}
    for cached in cache:
        g = dict(cached)
        if "verified_markets" not in (g.get("odds") or {}):
            g["odds"] = {}
        by_id[g["game_id"]] = g
    for fresh_game in fresh:
        g = dict(fresh_game)
        old = by_id.get(g["game_id"])
        if old:
            # ESPN removes closing odds after a game becomes final. Preserve
            # only a close that was previously parsed from complete real prices.
            if (g.get("completed") and not espn.has_priced_market(g.get("odds"))
                    and espn.has_priced_market(old.get("odds"))):
                g["odds"] = old["odds"]
            if g.get("home_score") is None and old.get("home_score") is not None:
                g["home_score"] = old["home_score"]
                g["away_score"] = old["away_score"]
                g["completed"] = old.get("completed", g.get("completed"))
        by_id[g["game_id"]] = g
    return sorted(by_id.values(), key=lambda x: (x.get("date_utc") or "", x["game_id"]))


def is_priceable(g: dict, today: dt.date, lookahead_days: int = 10) -> bool:
    """
    Whether a game belongs on the priced board right now.

    Excludes games already final (nothing left to price), games ESPN has
    marked postponed or canceled (there is no market forming around a game
    that isn't happening as scheduled), and anything more than a day in the
    past (the rolling fetch window can carry a stale unfinished game briefly).
    """
    if g.get("completed") or g.get("postponed") or g.get("canceled"):
        return False
    d = (g.get("date_utc") or "")[:10]
    if not d:
        return False
    lo = (today - dt.timedelta(days=1)).isoformat()
    hi = (today + dt.timedelta(days=max(0, lookahead_days))).isoformat()
    return lo <= d <= hi


def build_schedule(game_rows: list[dict], fbs: set[str] | None = None) -> list[dict]:
    """
    The full season, grouped by week -- what's available, nothing invented.

    The Board only prices games inside the rolling odds window, because that's
    all the model can honestly have an opinion on. This is the other half:
    every game for the whole season, so you can see what's coming even before
    a market exists for it. A game with no posted line shows odds as null --
    never a placeholder, never an estimate. If ESPN hasn't posted a number,
    this doesn't have one either.
    """
    from collections import defaultdict
    by_week: dict[str, list[dict]] = defaultdict(list)
    for g in game_rows:
        wk = g.get("week")
        key = str(wk) if wk is not None else "Unscheduled"
        o = g.get("odds") or {}
        has_odds = espn.has_priced_market(o)
        by_week[key].append({
            "game_id": g["game_id"], "date": g.get("date"),
            "away": g["away"], "home": g["home"],
            "away_name": g.get("away_name"), "home_name": g.get("home_name"),
            "away_score": g.get("away_score"), "home_score": g.get("home_score"),
            "completed": g.get("completed"), "neutral": g.get("neutral"),
            "postponed": g.get("postponed"), "canceled": g.get("canceled"),
            "status": g.get("status"),
            "spread_home": o.get("spread_home") if has_odds else None,
            "total": o.get("total") if has_odds else None,
            "ml_home": o.get("ml_home") if has_odds else None,
            "ml_away": o.get("ml_away") if has_odds else None,
            "book": o.get("book") if has_odds else None,
            "has_odds": has_odds,
            # Non-FBS participants are shown, never hidden -- they are real games
            # on the real schedule. They are flagged because the ratings model has
            # no honest opinion on them: an FCS team never appears often enough in
            # this feed to earn a rating, so it inherits "average FBS team", which
            # is how a 54-point favourite once came out as a coin flip.
            "away_fcs": bool(fbs) and g["away"] not in fbs,
            "home_fcs": bool(fbs) and g["home"] not in fbs,
        })

    def week_sort_key(k: str):
        return (0, int(k)) if k.isdigit() else (1, k)

    out = []
    for wk in sorted(by_week, key=week_sort_key):
        rows = sorted(by_week[wk], key=lambda r: r.get("date") or "")
        for slate_no, slate in enumerate(split_slates(rows), start=1):
            multi = slate_no > 1 or len(split_slates(rows)) > 1
            rng = date_range_label(slate)
            out.append({
                "week": wk,
                "slate": slate_no,
                "label": (f"Week {wk} · {rng}" if wk.isdigit() and multi
                          else f"Week {wk}" if wk.isdigit() else wk),
                "date_range": rng,
                "games": len(slate),
                "with_odds": sum(1 for r in slate if r["has_odds"]),
                "completed": sum(1 for r in slate if r["completed"]),
                "rows": slate,
            })
    return out


def split_slates(rows: list[dict], gap_days: int = 3) -> list[list[dict]]:
    """
    Break one ESPN "week" into the separate weekends it actually contains.

    ESPN's college-football calendar is not a week. The 2026 season opens with
    a Week 1 that runs Aug 22 to Sep 7 and holds two entirely separate
    weekends -- six games on Aug 29, then 137 more on Sep 3-5. Rendered as one
    chip that is a 143-game wall, and the opening slate (what everyone calls
    "Week 0") is invisible inside it.

    Splitting on a gap of three or more days recovers the real slates without
    inventing a week numbering ESPN does not publish. Games stay labelled with
    the week ESPN assigned them; they are just no longer piled into one heap.
    """
    if not rows:
        return []
    groups, cur, prev = [], [rows[0]], _row_date(rows[0])
    for r in rows[1:]:
        d = _row_date(r)
        if prev is not None and d is not None and (d - prev).days >= gap_days:
            groups.append(cur)
            cur = []
        cur.append(r)
        prev = d if d is not None else prev
    groups.append(cur)
    return groups


def _row_date(row: dict) -> dt.date | None:
    s = (row.get("date") or "")[:10]
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        return None


def date_range_label(rows: list[dict]) -> str:
    """'Aug 29' for a single day, 'Sep 3-5' for a run, spelled out across months."""
    ds = sorted(d for d in (_row_date(r) for r in rows) if d is not None)
    if not ds:
        return ""
    lo, hi = ds[0], ds[-1]
    if lo == hi:
        return lo.strftime("%b %-d")
    if lo.month == hi.month:
        return f"{lo.strftime('%b %-d')}-{hi.strftime('%-d')}"
    return f"{lo.strftime('%b %-d')} - {hi.strftime('%b %-d')}"


def rest_days(games: list[dict]) -> dict[str, int]:
    """
    Days of rest each team brings into its next game.

    Derived from the schedule itself -- one of the workbook columns that used to
    be typed in by hand and now simply isn't, because the calendar already knows.
    """
    last: dict[str, str] = {}
    out: dict[str, int] = {}
    for g in sorted(games, key=lambda x: x.get("date_utc") or ""):
        d = (g.get("date_utc") or "")[:10]
        if not d:
            continue
        for side in ("home", "away"):
            t = g[side]["abbr"]
            prev = last.get(t)
            if prev:
                try:
                    delta = (dt.date.fromisoformat(d) - dt.date.fromisoformat(prev)).days
                    out[f'{g["game_id"]}:{side}'] = min(delta, 21)
                except ValueError:
                    pass
        if g.get("completed"):
            for side in ("home", "away"):
                last[g[side]["abbr"]] = d
    return out


def fbs_teams(games: list[dict]) -> set[str]:
    """
    Which teams this season's schedule treats as full FBS participants.

    ESPN's ``groups=80`` scoreboard filter returns any game involving at least
    one FBS team -- which correctly includes "buy games" against a smaller
    school, so an FCS opponent shows up in the data too. There's no reliable
    classification flag on the team object itself (the site API's own /teams
    endpoint ignores the groups filter and happily returns Division III
    schools), so this infers it from behaviour instead.

    The strongest available signal is how OFTEN a team appears. An FBS program
    plays a dozen games against opponents this feed can see, so it turns up a
    dozen times. An FCS school appears only for the one or two payout games it
    takes, because the rest of its season is invisible at groups=80. Counting
    beats the older "is it ever the home team" rule, which quietly promoted any
    FCS school that happened to host a buy game -- North Dakota State hosting
    an FBS visitor was enough to have it priced as an FBS peer.

    The bar adapts to how much schedule is actually cached: with only a couple
    of weeks pulled, everyone looks infrequent, so it falls back to the older
    host-based rule rather than declaring the entire sport FCS.
    """
    from collections import Counter
    seen: Counter = Counter()
    hosts: set[str] = set()
    for g in games:
        for side in ("home", "away"):
            abbr = (g.get(side) or {}).get("abbr")
            if abbr:
                seen[abbr] += 1
        if (g.get("home") or {}).get("abbr"):
            hosts.add(g["home"]["abbr"])
    if not seen:
        return set()
    counts = sorted(seen.values())
    median = counts[len(counts) // 2]
    if median < 4:
        return hosts          # too little of the season cached to separate them
    bar = max(3, median / 2.0)
    return {t for t, n in seen.items() if n >= bar}


def fcs_guard(cands: list[dict], home_abbr: str, away_abbr: str,
             fbs: set[str], cfg: dict) -> list[dict]:
    """
    Refuse to recommend either side of a game against a non-FBS opponent.

    This is the concrete case the guard exists for: an FCS team getting run off
    the field produces a market spread the model has no real basis to challenge
    -- it has almost no data on that team, and what little it has gets pulled
    toward "average FBS team" by the ratings' own regularisation, which is far
    too generous for a team that isn't FBS at all. The result is a wide,
    confident-looking "edge" that is really just the model's blind spot, not a
    disagreement worth betting into.
    """
    if not cfg["filters"].get("exclude_fcs_opponents"):
        return cands
    if home_abbr in fbs and away_abbr in fbs:
        return cands
    missing = away_abbr if home_abbr in fbs else (home_abbr if away_abbr in fbs else f"{home_abbr}/{away_abbr}")
    for c in cands:
        if c["tier"] != "PASS":
            c["tier"] = "PASS"
        c["filtered"] = f"{missing} isn't a full FBS participant this season — model doesn't rate them reliably"
    return cands


def project(g: dict, rat: dict, hfa: float, score_rat: dict, league: float,
            home_bump: float, rests: dict, ovr: dict, cfg: dict) -> dict:
    """Projected margin (home - away) and projected combined total."""
    h, a = g["home"]["abbr"], g["away"]["abbr"]
    rh, ra = rat.get(h), rat.get(a)
    known = rh is not None and ra is not None
    rh = rh if rh is not None else 0.0
    ra = ra if ra is not None else 0.0

    mu = rh - ra
    if not g.get("neutral"):
        mu += hfa

    rh_rest = rests.get(f'{g["game_id"]}:home')
    ra_rest = rests.get(f'{g["game_id"]}:away')
    if rh_rest is not None and ra_rest is not None:
        mu += (rh_rest - ra_rest) * float(cfg["model"]["rest_day_weight"])

    o = ovr.get(g["game_id"], {})
    mu += float(o.get("margin_adj", 0.0))       # injuries, suspensions, news

    so_h = score_rat.get(h) or {"off": 0.0, "def": 0.0}
    so_a = score_rat.get(a) or {"off": 0.0, "def": 0.0}
    pts_home = league + so_h["off"] - so_a["def"] + (0.0 if g.get("neutral") else home_bump)
    pts_away = league + so_a["off"] - so_h["def"]
    proj_total = pts_home + pts_away + float(o.get("total_adj", 0.0))

    return {
        "mu": round(mu, 2),
        "proj_total": round(proj_total, 1),
        "proj_home_pts": round(pts_home, 1),
        "proj_away_pts": round(pts_away, 1),
        "ratings_known": known,
    }


def snapshot_confidence(snapshots: int) -> float:
    """
    Scale confidence by how many times this line has been observed.

    Not a guess about the game -- a fact about how much the pipeline has
    actually watched this number. A line seen for the first time this run
    (snapshots <= 1) hasn't had a chance to be corrected by anything; one
    that's been stable across several runs has survived more scrutiny.
    Ranges from 0.75 (brand new) up to 1.0 (seen 4+ times), never amplifying,
    only ever tempering an otherwise-full confidence score.
    """
    return min(1.0, 0.75 + (0.25 / 3.0) * max(0, snapshots - 1))


def price_game(g: dict, proj: dict, cfg: dict, conf: float) -> list[dict]:
    """Every market on one game, priced against the book."""
    o = g.get("odds") or {}
    blend = float(cfg["model"]["market_blend"])
    sd_m = float(cfg["model"]["margin_sd"])
    sd_t = float(cfg["model"]["total_sd"])
    keys = bool(cfg["model"]["use_key_numbers"])
    mu = proj["mu"]
    out: list[dict] = []

    base = {
        "game_id": g["game_id"],
        "game_date": g.get("date_utc"),
        "week": g.get("week"),
        "matchup": f'{g["away"]["abbr"]} @ {g["home"]["abbr"]}',
        "book": o.get("book"),
        "confidence": conf,
    }

    # ---- Moneyline -------------------------------------------------------- #
    if cfg["markets"]["moneyline"] and o.get("ml_home") is not None and o.get("ml_away") is not None:
        raw = M.moneyline_probability(mu, sd_m, keys)
        be_h = M.american_to_prob(float(o["ml_home"]))
        be_a = M.american_to_prob(float(o["ml_away"]))
        fair_h, fair_a = M.devig(be_h, be_a)
        p_h = (1 - blend) * raw + blend * fair_h
        for side, p, be, fair, price, label in (
            ("home", p_h, be_h, fair_h, float(o["ml_home"]), f'{g["home"]["abbr"]} ML'),
            ("away", 1 - p_h, be_a, fair_a, float(o["ml_away"]), f'{g["away"]["abbr"]} ML'),
        ):
            out.append({**base, "market": "ML", "side": side, "pick": label, "line": None,
                        "price": price, "model_prob": p, "raw_model_prob": raw if side == "home" else 1 - raw,
                        "market_fair_prob": fair, "breakeven": be, "push_prob": 0.0,
                        "edge": p - be, "ev": M.expected_value(p, price)})

    # ---- Spread ----------------------------------------------------------- #
    if (cfg["markets"]["spread"] and o.get("spread_home") is not None
            and o.get("spread_price_home") is not None and o.get("spread_price_away") is not None):
        sp = float(o["spread_home"])
        pw, pp, pl = M.cover_probability(mu, sd_m, sp, keys)
        # Re-normalise onto the non-push space, which is what the price pays on.
        denom = pw + pl
        raw_h = pw / denom if denom else 0.5
        ph_price = float(o["spread_price_home"])
        pa_price = float(o["spread_price_away"])
        be_h, be_a = M.american_to_prob(ph_price), M.american_to_prob(pa_price)
        fair_h, fair_a = M.devig(be_h, be_a)
        p_h = (1 - blend) * raw_h + blend * fair_h
        fmt = lambda x: f"{x:+g}"
        for side, p, be, fair, price, label in (
            ("home", p_h, be_h, fair_h, ph_price, f'{g["home"]["abbr"]} {fmt(sp)}'),
            ("away", 1 - p_h, be_a, fair_a, pa_price, f'{g["away"]["abbr"]} {fmt(-sp)}'),
        ):
            out.append({**base, "market": "ATS", "side": side, "pick": label, "line": sp,
                        "price": price, "model_prob": p,
                        "raw_model_prob": raw_h if side == "home" else 1 - raw_h,
                        "market_fair_prob": fair, "breakeven": be, "push_prob": round(pp, 4),
                        "projection_gap": abs(mu + sp),
                        "edge": p - be, "ev": M.expected_value(p * (1 - pp), price, pp)})

    # ---- Total ------------------------------------------------------------ #
    if (cfg["markets"]["total"] and o.get("total") is not None
            and o.get("over_price") is not None and o.get("under_price") is not None):
        tot = float(o["total"])
        po, pp, pu = M.over_probability(proj["proj_total"], tot, sd_t)
        denom = po + pu
        raw_o = po / denom if denom else 0.5
        op = float(o["over_price"])
        up = float(o["under_price"])
        be_o, be_u = M.american_to_prob(op), M.american_to_prob(up)
        fair_o, fair_u = M.devig(be_o, be_u)
        p_o = (1 - blend) * raw_o + blend * fair_o
        for side, p, be, fair, price, label in (
            ("over", p_o, be_o, fair_o, op, f"Over {tot:g}"),
            ("under", 1 - p_o, be_u, fair_u, up, f"Under {tot:g}"),
        ):
            out.append({**base, "market": "TOTAL", "side": side, "pick": label, "line": tot,
                        "price": price, "model_prob": p,
                        "raw_model_prob": raw_o if side == "over" else 1 - raw_o,
                        "market_fair_prob": fair, "breakeven": be, "push_prob": round(pp, 4),
                        "projection_gap": abs(proj["proj_total"] - tot),
                        "edge": p - be, "ev": M.expected_value(p * (1 - pp), price, pp)})

    for c in out:
        c["odds_verified"] = True
        c["raw_market_gap"] = abs(c["raw_model_prob"] - c["market_fair_prob"])
        c["tier"] = M.tier_for(c["edge"], cfg, conf)
    return out


def raw_gap_ceiling(cfg: dict, conf: float, thin: bool) -> float | None:
    """
    The raw model/market disagreement ceiling actually applied to a candidate.

    The configured ceiling is a floor-price on caution, not the last word. Edge
    rises monotonically with the raw gap, so a confidence-scaled tier threshold
    is itself a *minimum* raw gap in disguise. If the configured ceiling lands
    below that minimum, the two guards overlap to zero and nothing can qualify
    at any price -- the board goes quiet and looks exactly like a model with no
    opinion, which is the most dangerous failure mode a betting tool has.

    So the ceiling is widened, when it must be, to sit `guard_headroom` above
    whatever a LEAN currently requires. That keeps the guard doing its real job
    -- rejecting the absurd, like a rating model that has never seen an FCS team
    calling a 54-point favourite a coin flip -- without letting it silently
    gag the model it is meant to be protecting.
    """
    f = cfg["filters"]
    ceiling = f.get("max_raw_market_prob_gap")
    if thin and f.get("max_thin_data_raw_market_prob_gap") is not None:
        t = float(f["max_thin_data_raw_market_prob_gap"])
        ceiling = min(float(ceiling), t) if ceiling is not None else t
    if ceiling is None:
        return None
    headroom = float(f.get("guard_headroom", 1.0))
    if headroom > 1.0:
        needed = M.raw_gap_for_edge(M.edge_floor(cfg, conf, "lean"), cfg) * headroom
        ceiling = max(float(ceiling), needed)
    return float(ceiling)


def spread_gap_ceiling(cfg: dict, conf: float, thin: bool, market: str) -> float | None:
    """
    The projection-gap ceiling actually applied, in points.

    Same reasoning as raw_gap_ceiling(), in the other unit. A points ceiling and
    a probability threshold are the same constraint wearing different clothes,
    so this one gets the same headroom guarantee -- otherwise a 7-point rail
    quietly sits under a floor that needs 7.1 and the board never fills.
    """
    f = cfg["filters"]
    ceiling = (f.get("max_spread_projection_gap") if market == "ATS"
               else f.get("max_total_projection_gap") if market == "TOTAL" else None)
    if thin and market == "ATS" and f.get("max_thin_data_spread_gap") is not None:
        t = float(f["max_thin_data_spread_gap"])
        ceiling = min(float(ceiling), t) if ceiling is not None else t
    if ceiling is None:
        return None
    headroom = float(f.get("guard_headroom", 1.0))
    if headroom > 1.0 and market == "ATS":
        needed = M.spread_gap_for_edge(M.edge_floor(cfg, conf, "lean"), cfg) * headroom
        if needed != float("inf"):
            ceiling = max(float(ceiling), needed)
    return float(ceiling)


def apply_filters(cands: list[dict], cfg: dict, feed_healthy: bool = True) -> list[dict]:
    f = cfg["filters"]
    ok = []
    for c in cands:
        if not feed_healthy:
            c["tier"] = "PASS"
            c["filtered"] = "odds feed integrity warning"
        thin = c.get("confidence", 0.0) < float(f.get("thin_data_confidence_threshold", 0.5))
        gap_limit = spread_gap_ceiling(cfg, c.get("confidence", 0.0), thin, c["market"])
        if gap_limit is not None and c.get("projection_gap", 0.0) > float(gap_limit):
            c["tier"] = "PASS"
            c["filtered"] = "model and market are too far apart for a reliable early-season play"
        prob_limit = raw_gap_ceiling(cfg, c.get("confidence", 0.0), thin)
        if prob_limit is not None and c.get("raw_market_gap", 0.0) > float(prob_limit):
            c["tier"] = "PASS"
            c["filtered"] = "raw model/market disagreement exceeds the safety limit"
        if not (float(f["min_price"]) <= c["price"] <= float(f["max_price"])):
            c["tier"] = "PASS"
            c["filtered"] = "price outside allowed range"
        if c["ev"] <= 0 and c["tier"] != "PASS":
            c["tier"] = "PASS"
            c["filtered"] = "negative expected value"
        ok.append(c)
    return ok


def diagnose_board(board: list[dict], cfg: dict) -> dict:
    """
    Why the board looks the way it does -- especially when it is empty.

    An empty Best Bets tab is ambiguous in the worst possible way. It can mean
    the model looked and honestly disagreed with nothing, which is the normal
    and correct state most weeks. It can also mean the model was structurally
    prevented from having an opinion, which is a bug. Those two look identical
    from the outside, so this records which one actually happened: the live
    thresholds, the rails in force, what each rejection was for, and how close
    the nearest miss came.
    """
    qualified = [c for c in board if c["tier"] != "PASS"]
    reasons: dict[str, int] = {}
    for c in board:
        if c["tier"] != "PASS":
            continue
        why = c.get("filtered") or "edge below the tier threshold"
        reasons[why] = reasons.get(why, 0) + 1

    # Feasibility is only a meaningful claim when something was actually priced.
    # An empty board has no confidence to read, and defaulting it to zero makes
    # every quiet week look like a configuration fault -- a false alarm is worse
    # than no alarm, because it trains you to ignore the real one.
    confs = [c.get("confidence", 0.0) for c in board if c.get("confidence") is not None]
    conf = max(confs) if confs else M.confidence_score(0, 0, True, cfg)
    thin = conf < float(cfg["filters"].get("thin_data_confidence_threshold", 0.5))
    window = M.threshold_window(cfg, conf, thin,
                                raw_ceiling=raw_gap_ceiling(cfg, conf, thin),
                                spread_ceiling=spread_gap_ceiling(cfg, conf, thin, "ATS"))
    window["evaluated_on_priced_lines"] = bool(board)

    near = sorted((c for c in board if c["tier"] == "PASS" and not c.get("filtered")),
                  key=lambda c: -c["edge"])[:5]
    if qualified:
        headline = f"{len(qualified)} play(s) cleared the bar"
    elif not board:
        headline = "no games with a posted, two-sided price in the current window"
    elif not window["feasible"]:
        headline = ("the tier threshold and the safety rails leave no window -- "
                    "nothing could qualify at any price")
    else:
        headline = "the model did not disagree with the market by enough to bet"
    return {
        "priced_lines": len(board),
        "qualified": len(qualified),
        "headline": headline,
        "reasons": reasons,
        "window": window,
        "near_misses": [{
            "matchup": c["matchup"], "market": c["market"], "pick": c["pick"],
            "edge": round(c["edge"], 4), "needed": window["lean_edge_floor"],
            "short_by": round(window["lean_edge_floor"] - c["edge"], 4),
        } for c in near],
    }


def market_picks(cands: list[dict]) -> list[dict]:
    """
    Reduce a game's priced candidates to the model's actual pick per market.

    price_game() returns BOTH sides of every market (home and away, over and
    under) because the Full Board is meant to show the whole picture. But that
    makes both sides unsuitable for measuring accuracy in aggregate: home and
    away are complementary outcomes on the same game, so summing wins across
    both sides is a tautology -- exactly one of them wins every non-push game,
    so the aggregate win rate is forced toward 50% regardless of whether the
    model is any good. This keeps only the side with the larger edge per
    market -- the side the model would actually take -- which is what "was the
    model's pick right?" has to mean for the answer to carry information.
    """
    by_market: dict[str, dict] = {}
    for c in cands:
        cur = by_market.get(c["market"])
        if cur is None or c["edge"] > cur["edge"]:
            by_market[c["market"]] = c
    return list(by_market.values())


def correlation_guard(cands: list[dict], cfg: dict) -> list[dict]:
    """
    One angle per game.

    Taking a team's moneyline and its spread is close to taking the same bet
    twice: the outcomes are ~85% correlated, so the pair carries roughly double
    the variance the Kelly sizing assumed. Keep the best edge, demote the rest to
    a note rather than a wager.
    """
    limit = int(cfg["filters"].get("max_bets_per_game") or 1)
    by_game: dict[str, list[dict]] = {}
    for c in cands:
        by_game.setdefault(c["game_id"], []).append(c)
    keep = []
    for gid, rows in by_game.items():
        playable = [r for r in rows if r["tier"] != "PASS"]
        playable.sort(key=lambda r: (M.TIER_RANK[r["tier"]], -r["edge"]))
        for i, r in enumerate(playable):
            if i >= limit:
                r["tier"] = "PASS"
                r["filtered"] = "correlated with a stronger play on the same game"
        keep.extend(rows)
    return keep


def weekly_cap(cands: list[dict], cfg: dict) -> list[dict]:
    """
    Cap how many bets a single week can produce, best edges first.

    A 3% threshold against a full Saturday slate will qualify dozens of games.
    Betting all of them is not more edge, it is more variance and more exposure
    to the one thing every bet shares -- the model being wrong in the same
    direction all day. The cap is the difference between a staking plan and a
    spray.
    """
    limit = int(cfg["filters"].get("max_plays_per_week") or 0)
    if limit <= 0:
        return cands
    by_week: dict[str, list[dict]] = {}
    for c in cands:
        if c["tier"] == "PASS":
            continue
        by_week.setdefault(str(c.get("week") or (c.get("game_date") or "")[:10]), []).append(c)
    for wk, rows in by_week.items():
        rows.sort(key=lambda r: (M.TIER_RANK[r["tier"]], -r["edge"]))
        for r in rows[limit:]:
            r["tier"] = "PASS"
            r["filtered"] = f"outside the top {limit} plays for this week"
    return cands


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full-season backfill")
    ap.add_argument("--no-bet", action="store_true", help="price only, do not log bets")
    args = ap.parse_args()

    cfg = load_cfg()
    ovr = load_overrides()
    season = int(cfg["season"])
    prior_season = int(cfg["prior_season"])
    group = int(cfg["data"]["espn_group"])
    prio = cfg["data"]["odds_provider_priority"]

    today = dt.date.today()
    print(f"== ncaaf-edge build {store.now_iso()} (season {season}) ==")

    # 1. Prior season -- fetched once, then cached forever.
    prior_games = store.load(f"history_{prior_season}.json", [])
    if not prior_games:
        print(f"-- backfilling {prior_season} season (one time, a few minutes)")
        prior_games = espn.fetch_season(prior_season, group, prio)
        store.save(f"history_{prior_season}.json", prior_games)
    print(f"   prior season games: {len(prior_games)}")

    # 2. Current season.
    cache = store.load(f"games_{season}.json", [])
    if args.full or not cache:
        print("-- full season fetch")
        fresh = espn.fetch_range(dt.date(season, 8, 1),
                                 max(today + dt.timedelta(days=int(cfg["data"]["lookahead_days"])),
                                     dt.date(season + 1, 1, 31)),
                                 group, prio)
    else:
        lo = today - dt.timedelta(days=int(cfg["data"]["lookback_days"]))
        hi = today + dt.timedelta(days=int(cfg["data"]["lookahead_days"]))
        print(f"-- rolling fetch {lo} .. {hi}")
        fresh = espn.fetch_range(lo, hi, group, prio)
    games = merge_games(cache, fresh)
    store.save(f"games_{season}.json", games)
    print(f"   current season games: {len(games)} ({sum(1 for g in games if g.get('completed'))} final)")

    # 3. Odds snapshots (this is what makes CLV possible).
    raw_lines = store.load("lines.json", {})
    lines, removed_lines = store.clean_unverified_lines(raw_lines)
    lines = store.record_lines(lines, games)
    store.save("lines.json", lines)

    # Recover closing lines for finals the scoreboard has already stripped.
    ledg, removed_bets = store.clean_unverified_pending(store.load("ledger.json", {}))
    for bet in ledg.values():
        if bet.get("result") == "Pending" and not store.closer(lines, bet["game_id"]):
            rec = espn.odds_from_summary(bet["game_id"], prio)
            if rec:
                lines.setdefault(bet["game_id"], []).append({"ts": store.now_iso(), **rec})
    store.save("lines.json", lines)

    # 4. Ratings, solved from results.
    prior_rat, _ = R.solve_margin_ratings(prior_games, cfg)
    preseason = R.regress_to_prior(prior_rat, float(cfg["ratings"]["prior_regression"]))
    rat, hfa = R.solve_margin_ratings(games, cfg, prior=preseason)
    if not any(g.get("completed") for g in games):
        rat, hfa = preseason, float(cfg["model"]["home_field_fallback"])
    score_rat, league, home_bump = R.solve_scoring_ratings(games + prior_games, cfg)
    played = R.games_played(games)
    form = R.ats_form(games)
    rests = rest_days(games)
    fbs = fbs_teams(games)
    print(f"   ratings: {len(rat)} teams | home field {hfa:.2f} pts | league avg {league:.1f} pts")
    print(f"   FBS home participants this season: {len(fbs)} teams")

    # 5. Price the board.
    # Postponed/canceled games are excluded from pricing -- there is no market
    # forming around a game that isn't going to be played as scheduled, and
    # pricing one would just be noise on the board.
    board: list[dict] = []
    lookahead = int(cfg["data"]["lookahead_days"])
    upcoming = [g for g in games if is_priceable(g, today, lookahead)]
    odds_health = espn.odds_health(upcoming)
    for g in upcoming:
        has_odds = espn.has_priced_market(g.get("odds"))
        conf = M.confidence_score(played.get(g["home"]["abbr"], 0),
                                  played.get(g["away"]["abbr"], 0), has_odds, cfg)
        # Weakest link, not a product. Sample size and line-observation count are
        # two unrelated reasons to be unsure; multiplying them compounds a penalty
        # neither one justifies alone, and in the opening week that product pushed
        # confidence under the 0.35 floor in tier_for(), pinning the thresholds at
        # their harshest setting for reasons that had nothing to do with the model.
        conf = min(conf, snapshot_confidence(store.line_move(lines, g["game_id"]).get("snapshots", 0)))
        proj = project(g, rat, hfa, score_rat, league, home_bump, rests, ovr, cfg)
        if not proj["ratings_known"]:
            conf = min(conf, 0.4)
        cands = apply_filters(price_game(g, proj, cfg, conf), cfg, odds_health["healthy"])
        cands = fcs_guard(cands, g["home"]["abbr"], g["away"]["abbr"], fbs, cfg)
        for c in cands:
            c["projection"] = proj
        board.extend(cands)
    board = weekly_cap(correlation_guard(board, cfg), cfg)
    board.sort(key=lambda c: (M.TIER_RANK[c["tier"]], -c["edge"]))
    print(f"   priced {len(upcoming)} upcoming games -> {len(board)} market lines")

    board_diagnosis = diagnose_board(board, cfg)
    if board_diagnosis["qualified"] == 0:
        print(f"   NO QUALIFIED PLAYS -- {board_diagnosis['headline']}")
        for reason, n in board_diagnosis["reasons"].items():
            print(f"      {n:>4} x {reason}")
        if not board_diagnosis["window"]["feasible"]:
            print("      WARNING: thresholds and safety rails overlap to zero -- "
                  "nothing could have qualified at any price. Raise guard_headroom.")

    # 6. Log qualified bets, then grade finals.
    starting = float(cfg["bankroll"]["starting"])
    opened = 0
    if not args.no_bet:
        for c in board:
            if c["tier"] == "PASS":
                continue
            bankroll = (starting if cfg["bankroll"]["size_off"] == "starting"
                        else ledger.bankroll_from(ledg, starting))
            if ledger.open_bet(ledg, c, bankroll, cfg):
                opened += 1
    graded = ledger.grade_all(ledg, {g["game_id"]: g for g in games}, lines)
    store.save("ledger.json", ledg)
    print(f"   ledger: +{opened} new, {graded} graded, {len(ledg)} total")

    # 6b. Log EVERY priced market (bet or not) to the full prediction record.
    # This is what makes calibration and accuracy trustworthy: the ledger alone
    # is a biased sample (you only bet where the model disagrees most with the
    # market, which is where it's most likely to be wrong), while this records
    # what the model said about every game it ever priced.
    # Reduce to the model's actual pick per market before logging -- board
    # can carry both complementary sides of a market (home ATS + away ATS,
    # over + under) when both happen to pass the filters, and logging both
    # would make aggregate win-rate mathematically forced toward 50% (one
    # side always wins, the other always loses, however good the model is).
    preds, removed_predictions = store.clean_unverified_pending(store.load("predictions.json", {}))
    logged = 0
    by_game: dict[str, list[dict]] = {}
    for c in board:
        by_game.setdefault(c["game_id"], []).append(c)
    for cands in by_game.values():
        for c in market_picks(cands):
            if P.log_prediction(preds, c):
                logged += 1
    pred_graded = P.grade_all(preds, {g["game_id"]: g for g in games})
    store.save("predictions.json", preds)
    print(f"   predictions: +{logged} new, {pred_graded} graded, {len(preds)} total")

    # 7. Emit the site payload.
    os.makedirs(SITE_DATA, exist_ok=True)
    summary = ledger.summarise(ledg, starting)
    meta = {
        "generated_at": store.now_iso(),
        "season": season,
        "home_field_advantage": round(hfa, 2),
        "home_scoring_bump": round(home_bump, 2),
        "league_avg_points": round(league, 1),
        "games_final": sum(1 for g in games if g.get("completed")),
        "games_upcoming": len(upcoming),
        "settings": cfg,
        "brier": ledger.brier(ledg),
        "odds_health": odds_health,
        "board_diagnosis": board_diagnosis,
        "data_repair": {
            "unverified_line_snapshots_removed": removed_lines,
            "unverified_pending_bets_removed": removed_bets,
            "unverified_pending_predictions_removed": removed_predictions,
        },
    }

    def write(name: str, payload) -> None:
        with open(os.path.join(SITE_DATA, name), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"), default=str)

    write("meta.json", meta)
    write("board.json", [{**c, "line_move": store.line_move(lines, c["game_id"])} for c in board])
    write("ledger.json", sorted(ledg.values(), key=lambda b: (b.get("game_date") or ""), reverse=True))
    write("summary.json", {**summary, "calibration": ledger.calibration(ledg)})
    write("model_history.json", P.summarise(preds))
    write("ratings.json", sorted(
        [{"team": t,
          "rating": round(rat[t], 2),
          "off": round((score_rat.get(t) or {}).get("off", 0.0), 2),
          "def": round((score_rat.get(t) or {}).get("def", 0.0), 2),
          "games": played.get(t, 0),
          "ats": form.get(t)}
         for t in rat],
        key=lambda r: -r["rating"]))
    game_rows = [{
        "game_id": g["game_id"], "date": g.get("date_utc"), "week": g.get("week"),
        "away": g["away"]["abbr"], "home": g["home"]["abbr"],
        "away_name": g["away"]["name"], "home_name": g["home"]["name"],
        "away_score": g.get("away_score"), "home_score": g.get("home_score"),
        "completed": g.get("completed"), "neutral": g.get("neutral"),
        "postponed": g.get("postponed"), "canceled": g.get("canceled"),
        "status": g.get("status_detail"), "odds": g.get("odds") or None,
    } for g in games]
    write("games.json", game_rows)
    write("schedule.json", build_schedule(game_rows, fbs))

    print(f"   wrote {SITE_DATA}")
    roi_txt = "n/a" if summary["roi"] is None else f"{summary['roi'] * 100:.1f}%"
    print(f"== bankroll {cfg['currency_symbol']}{summary['current_bankroll']} "
          f"| {summary['settled']} settled | ROI {roi_txt} ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
