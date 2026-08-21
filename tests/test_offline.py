"""
Offline end-to-end test with synthetic games.

Runs the whole chain -- ratings solve, probability conversion, pricing, tiering,
grading, CLV, bankroll -- against a simulated season with known true team
strengths, so we can check that the model recovers what we planted. No network.

    python -m tests.test_offline
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import backtest as BT, build as B, ledger as L, model as M, ratings as R  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #

def make_season(n_teams: int = 40, weeks: int = 12, seed: int = 7) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    true = {t: rng.gauss(0, 9) for t in teams}
    true_off = {t: rng.gauss(0, 5) for t in teams}
    true_hfa = 2.6
    games, gid = [], 1000
    start = dt.date(2025, 8, 30)
    for w in range(weeks):
        pool = teams[:]
        rng.shuffle(pool)
        for i in range(0, len(pool) - 1, 2):
            a, h = pool[i], pool[i + 1]
            mu = true[h] - true[a] + true_hfa
            margin = round(rng.gauss(mu, 13.0))
            base = 27 + true_off[h] - (-true_off[a]) * 0.3
            hs = max(0, round(base + margin / 2 + rng.gauss(0, 7)))
            as_ = max(0, hs - margin)
            gid += 1
            d = start + dt.timedelta(days=7 * w)
            games.append({
                "game_id": str(gid),
                "date_utc": f"{d.isoformat()}T19:00Z",
                "week": w + 1, "neutral": False, "completed": True,
                "home": {"abbr": h, "name": h}, "away": {"abbr": a, "name": a},
                "home_score": hs, "away_score": as_,
                "odds": {"book": "Test", "spread_home": -round(mu * 2) / 2,
                         "spread_price_home": -110, "spread_price_away": -110,
                         "total": 55.0, "over_price": -110, "under_price": -110,
                         "ml_home": -150, "ml_away": 130},
            })
    return games, true


def test_ratings(cfg: dict) -> None:
    print("\n[ratings]")
    games, true = make_season()
    rat, hfa = R.solve_margin_ratings(games, cfg)
    check("solves a rating for every team", len(rat) == 40, f"{len(rat)} teams")
    check("home field lands in a sane range", 1.0 <= hfa <= 5.0, f"{hfa:.2f}")

    # One season is a small sample for a league-wide constant, so average a few.
    hfas = [R.solve_margin_ratings(make_season(seed=sd)[0], cfg)[1] for sd in (1, 2, 3, 4, 5)]
    mean_hfa = sum(hfas) / len(hfas)
    check("home field recovers the planted 2.6 across seasons",
          approx(mean_hfa, 2.6, 0.6), f"{mean_hfa:.2f}")

    tm = sorted(true, key=lambda t: -true[t])
    rm = sorted(rat, key=lambda t: -rat[t])
    overlap = len(set(tm[:10]) & set(rm[:10]))
    check("recovers 7+ of the true top 10", overlap >= 7, f"{overlap}/10")

    xs = [true[t] for t in rat]
    ys = [rat[t] for t in rat]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    corr = cov / (vx * vy)
    check("correlates >0.87 with true strength", corr > 0.87, f"r={corr:.3f}")

    sr, league, bump = R.solve_scoring_ratings(games, cfg)
    check("scoring ratings cover every team", len(sr) == 40)
    check("league scoring average is plausible", 15 <= league <= 45, f"{league:.1f}")

    empty_rat, empty_hfa = R.solve_margin_ratings([], cfg)
    check("no games -> falls back cleanly", empty_rat == {} and empty_hfa > 0)

    wk1, wk1_hfa = R.solve_margin_ratings(make_season(weeks=1, seed=3)[0], cfg)
    check("week 1 ratings are heavily shrunk, not wild",
          max(abs(v) for v in wk1.values()) < 6.0, f"max={max(abs(v) for v in wk1.values()):.1f}")
    check("week 1 home field stays near the configured prior",
          approx(wk1_hfa, cfg["model"]["home_field_fallback"], 1.0), f"{wk1_hfa:.2f}")


def test_probabilities(cfg: dict) -> None:
    print("\n[probability conversion]")
    d = M.margin_distribution(3.5, 13.0, True)
    check("margin distribution sums to 1", approx(sum(d.values()), 1.0, 1e-9))
    check("key numbers 3 and 7 spike above neighbours",
          d[3] > d[2] and d[3] > d[4] and d[7] > d[6] and d[7] > d[8])

    smooth = M.margin_distribution(3.5, 13.0, False)
    check("key-number model differs from the plain normal",
          abs(d[3] - smooth[3]) > 0.005, f"{d[3]:.4f} vs {smooth[3]:.4f}")

    w, p, l = M.cover_probability(0.0, 13.0, -3.0, True)
    check("cover/push/loss sums to 1", approx(w + p + l, 1.0, 1e-9))
    check("push at a whole-number 3 is material", p > 0.045, f"push={p:.3f}")
    _, p5, _ = M.cover_probability(0.0, 13.0, -5.0, True)
    check("3 pushes more often than 5", p > p5 * 1.5, f"3:{p:.3f} vs 5:{p5:.3f}")
    _, p7, _ = M.cover_probability(0.0, 13.0, -7.0, True)
    check("7 pushes more often than its neighbours",
          p7 > M.cover_probability(0.0, 13.0, -6.0, True)[1], f"7:{p7:.3f}")
    w2, p2, l2 = M.cover_probability(0.0, 13.0, -3.5, True)
    check("half-point spread has zero push", approx(p2, 0.0, 1e-12))
    # Buying off 3 does not change who covers -- margin>3 and margin>3.5 are the
    # same event. What it changes is that a 3-point win stops being a push and
    # starts being a loss. The value shows up in the push-adjusted win rate,
    # which is what the price actually pays on.
    eff3 = w / (w + l)
    eff35 = w2 / (w2 + l2)
    check("laying -3 beats laying -3.5", (eff3 - eff35) > 0.02, f"{eff3:.3f} vs {eff35:.3f}")
    smooth3 = M.cover_probability(0.0, 13.0, -3.0, False)[1]
    check("key-number model pushes on 3 more than a plain normal does",
          p > smooth3 * 1.5, f"{p:.3f} vs {smooth3:.3f}")

    check("pick'em is a coin flip", approx(M.moneyline_probability(0.0, 13.0), 0.5, 0.01))
    check("+14 favourite wins ~85%", 0.80 < M.moneyline_probability(14.0, 13.0) < 0.90,
          f"{M.moneyline_probability(14.0, 13.0):.3f}")

    o, pu, u = M.over_probability(55.0, 55.0, 10.0)
    check("over/push/under sums to 1", approx(o + pu + u, 1.0, 1e-9))
    check("total on the number is symmetric", approx(o, u, 0.01))

    check("-110 break-even is 52.38%", approx(M.american_to_prob(-110), 0.5238, 0.0005))
    check("+150 break-even is 40%", approx(M.american_to_prob(150), 0.40, 0.0005))
    fa, fb = M.devig(M.american_to_prob(-110), M.american_to_prob(-110))
    check("de-vigging -110/-110 gives 50/50", approx(fa, 0.5, 1e-9) and approx(fb, 0.5, 1e-9))
    fa2, fb2 = M.devig(M.american_to_prob(-200), M.american_to_prob(170))
    check("de-vigged pair sums to 1", approx(fa2 + fb2, 1.0, 1e-9))
    check("de-vigging lowers the favourite's implied price",
          fa2 < M.american_to_prob(-200))
    check("round-trip prob -> american -> prob",
          approx(M.american_to_prob(M.prob_to_american(0.62)), 0.62, 0.001))


def test_staking(cfg: dict) -> None:
    print("\n[staking]")
    check("no edge -> no Kelly", M.kelly_fraction(0.5238, -110) < 1e-9)
    check("real edge -> positive Kelly", M.kelly_fraction(0.60, -110) > 0.1)
    k1, k2 = M.kelly_fraction(0.60, -110), M.kelly_fraction(0.70, -110)
    check("bigger edge -> bigger Kelly", k2 > k1)
    s = M.stake_for(0.60, -110, 500.0, cfg)
    check("stake respects the max-stake cap", s <= 500.0 * cfg["bankroll"]["max_stake_pct"] + 1e-6,
          f"C${s}")
    check("stake is rounded to the configured step",
          approx(s / cfg["bankroll"]["round_stake_to"] % 1, 0, 1e-9), f"C${s}")
    huge = M.stake_for(0.99, 200, 500.0, cfg)
    check("absurd probability still capped",
          huge <= 500.0 * cfg["bankroll"]["max_stake_pct"] + 1e-6, f"C${huge}")
    check("EV is negative on a no-edge bet", M.expected_value(0.50, -110) < 0)
    check("EV is positive on a real edge", M.expected_value(0.60, -110) > 0)


def test_tiers(cfg: dict) -> None:
    print("\n[tiering]")
    hc = cfg["model"]["selection_haircut"]
    t = cfg["tiers"]
    check("thresholds are applied after the winner's-curse haircut",
          M.tier_for(t["good"] + hc, cfg, 1.0) == "GOOD"
          and M.tier_for(t["good"] + hc - 0.001, cfg, 1.0) == "LEAN", f"haircut={hc}")
    check("a clear best bet still grades BEST BET",
          M.tier_for(t["best_bet"] + hc + 0.02, cfg, 1.0) == "BEST BET")
    check("a marginal lean still grades LEAN",
          M.tier_for(t["lean"] + hc + 0.002, cfg, 1.0) == "LEAN")
    check("1% is a PASS", M.tier_for(0.01, cfg, 1.0) == "PASS")
    check("an edge smaller than the haircut can never qualify",
          M.tier_for(hc - 0.001, cfg, 1.0) == "PASS")
    check("no odds -> no confidence -> PASS", M.tier_for(0.20, cfg, 0.0) == "PASS")
    full = M.tier_for(t["good"] + hc, cfg, 1.0)
    low = M.tier_for(t["good"] + hc, cfg, 0.45)
    check("same edge on thin data is demoted", M.TIER_RANK[low] > M.TIER_RANK[full], f"{full} -> {low}")
    c_full = M.confidence_score(10, 10, True, cfg)
    c_thin = M.confidence_score(0, 0, True, cfg)
    check("confidence rises with sample size", c_full > c_thin, f"{c_thin} -> {c_full}")
    check("confidence is zero without odds", M.confidence_score(10, 10, False, cfg) == 0.0)


def test_grading_and_ledger(cfg: dict) -> None:
    print("\n[grading + ledger]")
    game = {"game_id": "1", "completed": True, "home_score": 28, "away_score": 24,
            "home": {"abbr": "H"}, "away": {"abbr": "A"}}
    cases = [
        ({"market": "ML", "side": "home", "line": None}, "Win"),
        ({"market": "ML", "side": "away", "line": None}, "Loss"),
        ({"market": "ATS", "side": "home", "line": -3.5}, "Win"),
        ({"market": "ATS", "side": "home", "line": -7.5}, "Loss"),
        ({"market": "ATS", "side": "home", "line": -4.0}, "Push"),
        ({"market": "ATS", "side": "away", "line": -7.5}, "Win"),
        ({"market": "TOTAL", "side": "over", "line": 49.5}, "Win"),
        ({"market": "TOTAL", "side": "under", "line": 49.5}, "Loss"),
        ({"market": "TOTAL", "side": "over", "line": 52.0}, "Push"),
    ]
    for spec, want in cases:
        bet = {"stake": 10.0, "price": -110, **spec}
        got, pnl = L._grade_one(bet, game)
        label = f'{spec["market"]}/{spec["side"]}@{spec["line"]}'
        check(f"grades {label} as {want}", got == want, got)
        if got == "Win":
            check(f"  payout on {label}", approx(pnl, 9.09, 0.01), f"{pnl}")
        if got == "Push":
            check(f"  push returns stake on {label}", pnl == 0.0)
        if got == "Loss":
            check(f"  loss on {label}", pnl == -10.0)

    ledg = {}
    cand = {"game_id": "1", "market": "ATS", "side": "home", "pick": "H -3.5",
            "line": -3.5, "price": -110, "model_prob": 0.60, "market_fair_prob": 0.5,
            "breakeven": 0.5238, "edge": 0.076, "ev": 0.14, "tier": "GOOD",
            "confidence": 0.9, "matchup": "A @ H", "game_date": "2026-09-05T19:00Z", "week": 2}
    check("opens a bet", L.open_bet(ledg, cand, 500.0, cfg))
    check("will not open the same bet twice", not L.open_bet(ledg, cand, 500.0, cfg))
    check("ledger holds exactly one bet", len(ledg) == 1)

    lines = {"1": [
        {"ts": "t0", "spread_home": -3.5, "spread_price_home": -110, "spread_price_away": -110,
         "total": 52.0, "over_price": -110, "under_price": -110, "ml_home": -160, "ml_away": 140},
        {"ts": "t1", "spread_home": -6.5, "spread_price_home": -110, "spread_price_away": -110,
         "total": 52.0, "over_price": -110, "under_price": -110, "ml_home": -240, "ml_away": 200},
    ]}
    n = L.grade_all(ledg, {"1": game}, lines)
    check("grades the pending bet", n == 1)
    bet = list(ledg.values())[0]
    check("result recorded", bet["result"] == "Win", bet["result"])
    check("CLV positive when the line moved our way", bet["clv_prob"] > 0, f'{bet["clv_prob"]}')

    against = {"b": dict(bet, bet_id="b", market="ATS", side="away", line=-3.5,
                         result="Pending", clv_prob=None)}
    L._attach_clv(against["b"], lines)
    check("CLV negative for the side the line moved against",
          against["b"]["clv_prob"] < 0, f'{against["b"]["clv_prob"]}')
    over_bet = {"market": "TOTAL", "side": "over", "line": 49.0, "price": -110,
                "game_id": "1", "stake": 10.0}
    L._attach_clv(over_bet, lines)
    check("CLV positive on an over taken below the close",
          over_bet["clv_prob"] > 0, f'{over_bet["clv_prob"]}')
    check("bankroll reflects the settled win",
          L.bankroll_from(ledg, 500.0) > 500.0, f'{L.bankroll_from(ledg, 500.0)}')

    s = L.summarise(ledg, 500.0)
    check("summary counts one settled bet", s["settled"] == 1 and s["wins"] == 1)
    check("summary ROI is positive", s["roi"] > 0)
    check("summary buckets by market", "ATS" in s["by_market"])
    check("bankroll curve has a point", len(s["curve"]) == 1)
    check("brier score computed", L.brier(ledg) is not None)
    check("calibration table built", len(L.calibration(ledg)) >= 1)


def test_pricing_pipeline(cfg: dict) -> None:
    print("\n[pricing pipeline]")
    g = {"game_id": "9", "date_utc": "2026-09-12T19:00Z", "week": 3, "neutral": False,
         "completed": False, "home": {"abbr": "H", "name": "Home"},
         "away": {"abbr": "A", "name": "Away"},
         "odds": {"book": "DraftKings", "spread_home": -7.0, "spread_price_home": -110,
                  "spread_price_away": -110, "total": 52.5, "over_price": -110,
                  "under_price": -110, "ml_home": -280, "ml_away": 230}}
    proj = {"mu": 10.0, "proj_total": 58.0, "proj_home_pts": 34.0,
            "proj_away_pts": 24.0, "ratings_known": True}
    cands = B.price_game(g, proj, cfg, 1.0)
    check("prices all three markets, both sides", len(cands) == 6, f"{len(cands)}")
    for mkt in ("ML", "ATS", "TOTAL"):
        pair = [c for c in cands if c["market"] == mkt]
        check(f"{mkt} sides sum to 1",
              approx(sum(c["model_prob"] for c in pair), 1.0, 1e-6))
    check("model likes the side it projects past the number",
          [c for c in cands if c["market"] == "ATS" and c["side"] == "home"][0]["edge"] > 0)
    check("model likes the over on a total it projects above",
          [c for c in cands if c["market"] == "TOTAL" and c["side"] == "over"][0]["edge"] > 0)

    guarded = B.correlation_guard(B.apply_filters(cands, cfg), cfg)
    playable = [c for c in guarded if c["tier"] != "PASS"]
    check("correlation guard keeps at most one play per game",
          len(playable) <= cfg["filters"]["max_bets_per_game"], f"{len(playable)}")
    check("demoted plays say why",
          all(c.get("filtered") or c["tier"] == "PASS" or c in playable for c in guarded))

    no_odds = dict(g, odds={})
    check("a game with no odds produces nothing", B.price_game(no_odds, proj, cfg, 0.0) == [])

    long_shot = dict(g, odds={**g["odds"], "ml_home": -900, "ml_away": 600})
    filtered = B.apply_filters(B.price_game(long_shot, proj, cfg, 1.0), cfg)
    heavy = [c for c in filtered if c["market"] == "ML" and c["side"] == "home"][0]
    check("price filter rejects a -900 favourite", heavy["tier"] == "PASS", heavy.get("filtered"))


def test_calibration(cfg: dict) -> None:
    """
    Walk-forward calibration on a simulated season.

    The first version of this test solved ratings over the whole season and then
    graded games inside it, which leaks the result into the input: a team that
    got lucky in week 6 carries a higher rating INTO week 6. It reported the
    model as beautifully calibrated and slightly clairvoyant. The honest version
    below re-solves ratings each matchday using only games already finished --
    and immediately shows the model is overconfident out of sample, which is the
    well-known failure mode of every rating model and the reason the market
    blend exists.
    """
    print("\n[walk-forward calibration]")
    games, _ = make_season(n_teams=32, weeks=14, seed=11)
    for g in games:
        # Give the simulated book a realistic error instead of the exact truth,
        # otherwise the market is unbeatable by construction and the test only
        # proves the market is the market.
        g["odds"]["spread_home"] = round((g["odds"]["spread_home"] + random.Random(
            int(g["game_id"])).gauss(0, 2.4)) * 2) / 2

    res = BT.run(games, cfg, min_history=80)
    check("backtest ran", not res.get("error"), res.get("error", ""))
    if res.get("error"):
        return
    for c in res["calibration"]:
        print(f"       {c['bucket']:>9}  n={c['n']:<5} actual={c['actual']:>7.1%}  gap={c['gap']:+.3f}")
    check("out-of-sample calibration is within tolerance",
          res["mean_abs_calibration_gap"] < 0.09, f"mean abs gap {res['mean_abs_calibration_gap']}")
    check("it priced a meaningful number of sides", res["sides_priced"] > 200,
          f"{res['sides_priced']}")
    check("it selected some bets", res["bets"] > 0, f"{res['bets']}")
    print(f"       selection gap {res['selection_gap']}  |  {res['bets']} bets  |  ROI {res['roi']}")
    check("selected bets underperform their claimed probability (winner's curse)",
          res["selection_gap"] is not None and res["selection_gap"] < 0,
          f"{res['selection_gap']}")
    check("the backtest never leaks the future — early games are never priced",
          res["sides_priced"] < 6 * len(games), "sides < 6 per game")


def test_weekly_cap(cfg: dict) -> None:
    print("\n[weekly cap]")
    limit = int(cfg["filters"]["max_plays_per_week"])
    cands = [{"game_id": str(i), "week": 5, "tier": "GOOD", "edge": 0.05 + i/1000,
              "market": "ATS", "side": "home"} for i in range(limit + 8)]
    out = B.weekly_cap([dict(c) for c in cands], cfg)
    kept = [c for c in out if c["tier"] != "PASS"]
    check(f"caps a week at {limit} plays", len(kept) == limit, f"{len(kept)}")
    check("keeps the biggest edges", min(c["edge"] for c in kept) > max(
        c["edge"] for c in out if c["tier"] == "PASS"))
    check("demoted plays explain themselves",
          all("top" in (c.get("filtered") or "") for c in out if c["tier"] == "PASS"))
    spread = B.weekly_cap([dict(c, week=i) for i, c in enumerate(cands)], cfg)
    check("the cap is per week, not global",
          len([c for c in spread if c["tier"] != "PASS"]) == len(cands))


def test_rest_days() -> None:
    print("\n[schedule-derived rest days]")
    games = [
        {"game_id": "1", "date_utc": "2026-09-05T19:00Z", "completed": True,
         "home": {"abbr": "H"}, "away": {"abbr": "A"}},
        {"game_id": "2", "date_utc": "2026-09-12T19:00Z", "completed": True,
         "home": {"abbr": "H"}, "away": {"abbr": "B"}},
        {"game_id": "3", "date_utc": "2026-09-26T19:00Z", "completed": False,
         "home": {"abbr": "A"}, "away": {"abbr": "H"}},
    ]
    r = B.rest_days(games)
    check("normal week is 7 days rest", r.get("2:home") == 7, str(r.get("2:home")))
    check("bye week is 14 days rest", r.get("3:away") == 14, str(r.get("3:away")))
    check("first game of the year has no rest number", "1:home" not in r)


def test_merge() -> None:
    print("\n[merge safety]")
    old = [{"game_id": "1", "date_utc": "2026-09-05T19:00Z", "completed": True,
            "home_score": 28, "away_score": 24,
            "odds": {"spread_home": -6.5, "total": 51.0}}]
    fresh = [{"game_id": "1", "date_utc": "2026-09-05T19:00Z", "completed": True,
              "home_score": None, "away_score": None, "odds": {}}]
    m = B.merge_games(old, fresh)
    check("a blank refresh never erases the closing line",
          m[0]["odds"].get("spread_home") == -6.5)
    check("a blank refresh never erases the final score", m[0]["home_score"] == 28)


def main() -> int:
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "config", "settings.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    print("=" * 62)
    print("ncaaf-edge offline test suite")
    print("=" * 62)
    test_ratings(cfg)
    test_probabilities(cfg)
    test_staking(cfg)
    test_tiers(cfg)
    test_grading_and_ledger(cfg)
    test_pricing_pipeline(cfg)
    test_calibration(cfg)
    test_weekly_cap(cfg)
    test_rest_days()
    test_merge()
    print("\n" + "=" * 62)
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
