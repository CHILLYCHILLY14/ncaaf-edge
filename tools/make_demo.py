"""
Generate a realistic demo dataset by running the REAL model over a simulated
season. No network. Used to preview the site and to sanity-check the whole
pipeline end to end without waiting for Saturday.

    python -m tools.make_demo            # writes site/data/*.json
    python -m tools.make_demo --embed    # also writes site/preview.html
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import build as B, ledger as L, model as M, ratings as R, store  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAMS = [("ALA","Alabama"),("UGA","Georgia"),("OSU","Ohio State"),("MICH","Michigan"),
         ("TEX","Texas"),("ORE","Oregon"),("PSU","Penn State"),("ND","Notre Dame"),
         ("LSU","LSU"),("OU","Oklahoma"),("TENN","Tennessee"),("USC","USC"),
         ("CLEM","Clemson"),("FSU","Florida State"),("MISS","Ole Miss"),("UTAH","Utah"),
         ("IOWA","Iowa"),("WIS","Wisconsin"),("AUB","Auburn"),("TAMU","Texas A&M"),
         ("KSU","Kansas State"),("MIA","Miami"),("LOU","Louisville"),("ARIZ","Arizona"),
         ("UNC","North Carolina"),("TCU","TCU"),("BYU","BYU"),("MIZZ","Missouri"),
         ("WASH","Washington"),("NEB","Nebraska"),("SMU","SMU"),("IND","Indiana")]


def simulate(seed: int = 11):
    rng = random.Random(seed)
    strength = {a: rng.gauss(0, 9.5) for a, _ in TEAMS}
    off = {a: rng.gauss(0, 4.5) for a, _ in TEAMS}
    games, gid = [], 401_800_000
    season_start = dt.date(2026, 9, 5)
    today = dt.date(2026, 11, 7)   # mid-November: enough history to be interesting
    for wk in range(1, 15):
        d = season_start + dt.timedelta(days=7*(wk-1))
        pool = [a for a, _ in TEAMS]; rng.shuffle(pool)
        for i in range(0, len(pool)-1, 2):
            away, home = pool[i], pool[i+1]
            gid += 1
            neutral = rng.random() < 0.05
            mu = strength[home] - strength[away] + (0 if neutral else 2.5)
            # The book's number: the truth plus a little noise, rounded to a half.
            book_spread = -round((mu + rng.gauss(0, 1.6)) * 2) / 2
            book_total = round((54 + off[home] + off[away] + rng.gauss(0, 4)) * 2) / 2
            fav_p = 1/(1+2.71828**(-mu/7.5))
            ml_h = M.prob_to_american(min(0.95, max(0.05, fav_p*1.035)))
            ml_a = M.prob_to_american(min(0.95, max(0.05, (1-fav_p)*1.035)))
            done = d < today
            hs = as_ = None
            if done:
                margin = round(rng.gauss(mu, 13.2))
                base = (book_total + rng.gauss(0, 11)) / 2
                hs = max(0, round(base + margin/2)); as_ = max(0, hs - margin)
            games.append({
                "game_id": str(gid), "date_utc": f"{d.isoformat()}T{rng.choice(['16:00','19:30','23:00'])}Z",
                "season": 2026, "week": wk, "neutral": neutral, "indoor": False,
                "completed": done, "status_detail": "Final" if done else "Sat",
                "home": {"abbr": home, "name": dict(TEAMS)[home]},
                "away": {"abbr": away, "name": dict(TEAMS)[away]},
                "home_score": hs, "away_score": as_,
                "odds": {"book": "DraftKings", "spread_home": book_spread,
                         "spread_price_home": -110, "spread_price_away": -110,
                         "total": book_total, "over_price": -110, "under_price": -110,
                         "ml_home": round(ml_h), "ml_away": round(ml_a)},
            })
    return games, today


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--embed", action="store_true")
    args = ap.parse_args()
    cfg = B.load_cfg()
    games, today = simulate()

    lines = {}
    rng = random.Random(5)
    for g in games:                      # fabricate an opener + a close for CLV
        o = g["odds"]
        lines[g["game_id"]] = [
            {"ts": "2026-01-01T00:00:00+00:00", **{**o,
             "spread_home": o["spread_home"] + rng.choice([-1.5,-1,-.5,0,.5,1,1.5]),
             "total": o["total"] + rng.choice([-2,-1,0,1,2])}},
            {"ts": "2026-01-02T00:00:00+00:00", **o},
        ]

    prior_rat, _ = R.solve_margin_ratings(games[:len(games)//2], cfg)
    rat, hfa = R.solve_margin_ratings(games, cfg,
                                      prior=R.regress_to_prior(prior_rat, 0.62))
    score_rat, league, bump = R.solve_scoring_ratings(games, cfg)
    played, form, rests = R.games_played(games), R.ats_form(games), B.rest_days(games)

    # Mirror the real pipeline exactly: price everything, guard correlation,
    # cap the week, and only then log bets -- otherwise the preview shows a
    # model that bets far more often than the one that actually ships.
    priced, ledg = [], {}
    starting = float(cfg["bankroll"]["starting"])
    for g in sorted(games, key=lambda x: x["date_utc"]):
        conf = M.confidence_score(played.get(g["home"]["abbr"],0), played.get(g["away"]["abbr"],0), True, cfg)
        proj = B.project(g, rat, hfa, score_rat, league, bump, rests, {}, cfg)
        for c in B.apply_filters(B.price_game(g, proj, cfg, conf), cfg):
            c["projection"] = proj
            c["_completed"] = g["completed"]
            priced.append(c)
    priced = B.weekly_cap(B.correlation_guard(priced, cfg), cfg)

    for c in sorted(priced, key=lambda x: x["game_date"]):
        if c["tier"] != "PASS":
            L.open_bet(ledg, c, L.bankroll_from(ledg, starting), cfg)
    board = [c for c in priced if not c.pop("_completed")]

    L.grade_all(ledg, {g["game_id"]: g for g in games}, lines)
    board.sort(key=lambda c: (M.TIER_RANK[c["tier"]], -c["edge"]))

    summary = L.summarise(ledg, starting)
    payload = {
        "meta": {"generated_at": store.now_iso(), "season": 2026,
                 "home_field_advantage": round(hfa,2), "league_avg_points": round(league,1),
                 "games_final": sum(1 for g in games if g["completed"]),
                 "games_upcoming": sum(1 for g in games if not g["completed"]),
                 "settings": cfg, "brier": L.brier(ledg), "demo": True},
        "board": [{**c, "line_move": store.line_move(lines, c["game_id"])} for c in board],
        "ledger": sorted(ledg.values(), key=lambda b: b.get("game_date") or "", reverse=True),
        "summary": {**summary, "calibration": L.calibration(ledg)},
        "ratings": sorted([{"team": t, "rating": round(rat[t],2),
                            "off": round((score_rat.get(t) or {}).get("off",0),2),
                            "def": round((score_rat.get(t) or {}).get("def",0),2),
                            "games": played.get(t,0), "ats": form.get(t)} for t in rat],
                          key=lambda r: -r["rating"]),
    }

    out = os.path.join(ROOT, "site", "data")
    os.makedirs(out, exist_ok=True)
    for k, v in payload.items():
        with open(os.path.join(out, k + ".json"), "w", encoding="utf-8") as fh:
            json.dump(v, fh, separators=(",", ":"), default=str)

    if args.embed:
        src = open(os.path.join(ROOT, "site", "index.html"), encoding="utf-8").read()
        blob = json.dumps(payload, separators=(",", ":"), default=str)
        src = src.replace("<script>\n\"use strict\";",
                          "<script>window.__NCAAF_DATA__=" + blob + ";</script>\n<script>\n\"use strict\";", 1)
        with open(os.path.join(ROOT, "site", "preview.html"), "w", encoding="utf-8") as fh:
            fh.write(src)
        print("wrote site/preview.html")

    print(f"demo: {len(board)} board lines, {len(ledg)} bets, "
          f"{summary['settled']} settled, bankroll {summary['current_bankroll']}, "
          f"ROI {summary['roi']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
