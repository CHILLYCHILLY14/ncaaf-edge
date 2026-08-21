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

from . import espn, ledger, model as M, ratings as R, store

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
    by_id = {g["game_id"]: g for g in cache}
    for g in fresh:
        old = by_id.get(g["game_id"])
        if old:
            if not (g.get("odds") or {}).get("spread_home") and (old.get("odds") or {}).get("spread_home"):
                g["odds"] = old["odds"]
            if g.get("home_score") is None and old.get("home_score") is not None:
                g["home_score"] = old["home_score"]
                g["away_score"] = old["away_score"]
                g["completed"] = old.get("completed", g.get("completed"))
        by_id[g["game_id"]] = g
    return sorted(by_id.values(), key=lambda x: (x.get("date_utc") or "", x["game_id"]))


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
    schools), so this infers it from behaviour instead: a genuine FBS program
    hosts several games a year, while an FCS opponent brought in for a payout
    game is, essentially without exception, always the AWAY team. A team that
    never once appears as the home side across the whole season's schedule is
    not being treated as an FBS peer by the schedule itself.
    """
    return {g["home"]["abbr"] for g in games if g.get("home", {}).get("abbr")}


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
    if cfg["markets"]["spread"] and o.get("spread_home") is not None:
        sp = float(o["spread_home"])
        pw, pp, pl = M.cover_probability(mu, sd_m, sp, keys)
        # Re-normalise onto the non-push space, which is what the price pays on.
        denom = pw + pl
        raw_h = pw / denom if denom else 0.5
        ph_price = float(o.get("spread_price_home") or -110)
        pa_price = float(o.get("spread_price_away") or -110)
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
                        "edge": p - be, "ev": M.expected_value(p * (1 - pp), price, pp)})

    # ---- Total ------------------------------------------------------------ #
    if cfg["markets"]["total"] and o.get("total") is not None:
        tot = float(o["total"])
        po, pp, pu = M.over_probability(proj["proj_total"], tot, sd_t)
        denom = po + pu
        raw_o = po / denom if denom else 0.5
        op = float(o.get("over_price") or -110)
        up = float(o.get("under_price") or -110)
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
                        "edge": p - be, "ev": M.expected_value(p * (1 - pp), price, pp)})

    for c in out:
        c["tier"] = M.tier_for(c["edge"], cfg, conf)
    return out


def apply_filters(cands: list[dict], cfg: dict) -> list[dict]:
    f = cfg["filters"]
    ok = []
    for c in cands:
        if not (float(f["min_price"]) <= c["price"] <= float(f["max_price"])):
            c["tier"] = "PASS"
            c["filtered"] = "price outside allowed range"
        if c["ev"] <= 0 and c["tier"] != "PASS":
            c["tier"] = "PASS"
            c["filtered"] = "negative expected value"
        ok.append(c)
    return ok


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
    lines = store.record_lines(store.load("lines.json", {}), games)
    store.save("lines.json", lines)

    # Recover closing lines for finals the scoreboard has already stripped.
    ledg = store.load("ledger.json", {})
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
    board: list[dict] = []
    upcoming = [g for g in games
                if not g.get("completed")
                and g.get("date_utc")
                and g["date_utc"][:10] >= (today - dt.timedelta(days=1)).isoformat()]
    for g in upcoming:
        conf = M.confidence_score(played.get(g["home"]["abbr"], 0),
                                  played.get(g["away"]["abbr"], 0),
                                  bool(g.get("odds", {}).get("spread_home") is not None
                                       or g.get("odds", {}).get("ml_home") is not None),
                                  cfg)
        proj = project(g, rat, hfa, score_rat, league, home_bump, rests, ovr, cfg)
        if not proj["ratings_known"]:
            conf = min(conf, 0.4)
        cands = apply_filters(price_game(g, proj, cfg, conf), cfg)
        cands = fcs_guard(cands, g["home"]["abbr"], g["away"]["abbr"], fbs, cfg)
        for c in cands:
            c["projection"] = proj
        board.extend(cands)
    board = weekly_cap(correlation_guard(board, cfg), cfg)
    board.sort(key=lambda c: (M.TIER_RANK[c["tier"]], -c["edge"]))
    print(f"   priced {len(upcoming)} upcoming games -> {len(board)} market lines")

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

    # 7. Emit the site payload.
    os.makedirs(SITE_DATA, exist_ok=True)
    summary = ledger.summarise(ledg, starting)
    meta = {
        "generated_at": store.now_iso(),
        "season": season,
        "home_field_advantage": round(hfa, 2),
        "league_avg_points": round(league, 1),
        "games_final": sum(1 for g in games if g.get("completed")),
        "games_upcoming": len(upcoming),
        "settings": cfg,
        "brier": ledger.brier(ledg),
    }

    def write(name: str, payload) -> None:
        with open(os.path.join(SITE_DATA, name), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"), default=str)

    write("meta.json", meta)
    write("board.json", [{**c, "line_move": store.line_move(lines, c["game_id"])} for c in board])
    write("ledger.json", sorted(ledg.values(), key=lambda b: (b.get("game_date") or ""), reverse=True))
    write("summary.json", {**summary, "calibration": ledger.calibration(ledg)})
    write("ratings.json", sorted(
        [{"team": t,
          "rating": round(rat[t], 2),
          "off": round((score_rat.get(t) or {}).get("off", 0.0), 2),
          "def": round((score_rat.get(t) or {}).get("def", 0.0), 2),
          "games": played.get(t, 0),
          "ats": form.get(t)}
         for t in rat],
        key=lambda r: -r["rating"]))
    write("games.json", [{
        "game_id": g["game_id"], "date": g.get("date_utc"), "week": g.get("week"),
        "away": g["away"]["abbr"], "home": g["home"]["abbr"],
        "away_name": g["away"]["name"], "home_name": g["home"]["name"],
        "away_score": g.get("away_score"), "home_score": g.get("home_score"),
        "completed": g.get("completed"), "neutral": g.get("neutral"),
        "status": g.get("status_detail"), "odds": g.get("odds"),
    } for g in games])

    print(f"   wrote {SITE_DATA}")
    roi_txt = "n/a" if summary["roi"] is None else f"{summary['roi'] * 100:.1f}%"
    print(f"== bankroll {cfg['currency_symbol']}{summary['current_bankroll']} "
          f"| {summary['settled']} settled | ROI {roi_txt} ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
