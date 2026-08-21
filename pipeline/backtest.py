"""
Walk-forward backtest.

    python -m pipeline.backtest              # on cached real seasons
    python -m pipeline.backtest --season 2025

The only evaluation of a betting model that means anything. For each week, the
ratings are solved using *only* games that had already finished, then that
week's games are priced and graded. Nothing the model sees when it prices a game
comes from after that game.

This is easy to get wrong and the wrong version is very flattering. Solving
ratings over the whole season and then "predicting" games inside that season
leaks the outcome into the input: a team that got lucky in week 6 carries a
higher rating into week 6, and the model looks clairvoyant. Any backtest that
reports a big edge is much more likely to have this bug than to have found one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import defaultdict

from . import model as M, ratings as R, store


def _outcome(cand: dict, game: dict) -> bool | None:
    """True/False if the bet won/lost, None if it pushed or can't be graded."""
    hs, as_ = game.get("home_score"), game.get("away_score")
    if hs is None or as_ is None:
        return None
    margin, total = hs - as_, hs + as_
    if cand["market"] == "ATS":
        adj = margin + cand["line"]
        if abs(adj) < 1e-9:
            return None
        return (adj > 0) if cand["side"] == "home" else (adj < 0)
    if cand["market"] == "ML":
        if margin == 0:
            return None
        return (margin > 0) if cand["side"] == "home" else (margin < 0)
    if abs(total - cand["line"]) < 1e-9:
        return None
    return (total > cand["line"]) if cand["side"] == "over" else (total < cand["line"])


def run(games: list[dict], cfg: dict, min_history: int = 60,
        prior: dict[str, float] | None = None) -> dict:
    """
    Returns calibration over every priced side, plus staked results for the
    sides the model actually would have bet.
    """
    from . import build as B  # imported here to avoid a circular import at module load

    played = sorted([g for g in games if g.get("completed")
                     and g.get("home_score") is not None],
                    key=lambda g: g.get("date_utc") or "")
    if len(played) < min_history + 10:
        return {"error": f"need at least {min_history + 10} completed games, have {len(played)}"}

    by_day: dict[str, list[dict]] = defaultdict(list)
    for g in played:
        by_day[(g.get("date_utc") or "")[:10]].append(g)
    days = sorted(by_day)

    all_sides: list[tuple[float, bool]] = []
    bets: list[dict] = []
    history: list[dict] = []
    starting = float(cfg["bankroll"]["starting"])
    bankroll = starting

    for d in days:
        todays = by_day[d]
        if len(history) >= min_history:
            rat, hfa = R.solve_margin_ratings(history, cfg, prior=prior)
            sr, league, bump = R.solve_scoring_ratings(history, cfg)
            n_played = R.games_played(history)
            rests = B.rest_days(history + todays)
            fbs = B.fbs_teams(history + todays)

            day_cands: list[dict] = []
            for g in todays:
                conf = M.confidence_score(n_played.get(g["home"]["abbr"], 0),
                                          n_played.get(g["away"]["abbr"], 0), True, cfg)
                proj = B.project(g, rat, hfa, sr, league, bump, rests, {}, cfg)
                cands = B.apply_filters(B.price_game(g, proj, cfg, conf), cfg)
                cands = B.fcs_guard(cands, g["home"]["abbr"], g["away"]["abbr"], fbs, cfg)
                for c in cands:
                    won = _outcome(c, g)
                    if won is not None:
                        all_sides.append((c["model_prob"], won))
                        c["_won"] = won
                day_cands.extend(cands)

            for c in B.correlation_guard(day_cands, cfg):
                if c["tier"] == "PASS" or c.get("_won") is None:
                    continue
                stake = M.stake_for(c["model_prob"], c["price"], bankroll, cfg)
                if stake <= 0:
                    continue
                pnl = stake * (M.american_to_decimal(c["price"]) - 1.0) if c["_won"] else -stake
                bankroll += pnl
                bets.append({"date": d, "tier": c["tier"], "market": c["market"],
                             "prob": c["model_prob"], "edge": c["edge"], "price": c["price"],
                             "stake": stake, "won": c["_won"], "pnl": pnl,
                             "bankroll": round(bankroll, 2)})
        history.extend(todays)

    return _report(all_sides, bets, starting, bankroll)


def _report(all_sides, bets, starting, bankroll) -> dict:
    buckets: dict[int, list[int]] = {}
    for p, w in all_sides:
        k = int(p * 10)
        buckets.setdefault(k, [0, 0])
        buckets[k][0] += 1
        buckets[k][1] += int(w)
    cal = [{"bucket": f"{k*10}-{k*10+10}%", "n": n, "predicted": k / 10 + 0.05,
            "actual": round(w / n, 4), "gap": round(w / n - (k / 10 + 0.05), 4)}
           for k, (n, w) in sorted(buckets.items()) if n >= 25]
    mean_gap = round(sum(abs(c["gap"]) for c in cal) / len(cal), 4) if cal else None

    staked = sum(b["stake"] for b in bets)
    pnl = sum(b["pnl"] for b in bets)
    wins = sum(1 for b in bets if b["won"])
    sel_gap = None
    if bets:
        sel_gap = round(wins / len(bets) - sum(b["prob"] for b in bets) / len(bets), 4)

    by_tier: dict[str, dict] = {}
    for b in bets:
        r = by_tier.setdefault(b["tier"], {"n": 0, "w": 0, "staked": 0.0, "pnl": 0.0})
        r["n"] += 1
        r["w"] += int(b["won"])
        r["staked"] += b["stake"]
        r["pnl"] += b["pnl"]
    for r in by_tier.values():
        r["roi"] = round(r["pnl"] / r["staked"], 4) if r["staked"] else None
        r["staked"] = round(r["staked"], 2)
        r["pnl"] = round(r["pnl"], 2)

    return {
        "sides_priced": len(all_sides),
        "calibration": cal,
        "mean_abs_calibration_gap": mean_gap,
        "bets": len(bets),
        "wins": wins,
        "win_rate": round(wins / len(bets), 4) if bets else None,
        "selection_gap": sel_gap,
        "staked": round(staked, 2),
        "pnl": round(pnl, 2),
        "roi": round(pnl / staked, 4) if staked else None,
        "starting_bankroll": starting,
        "ending_bankroll": round(bankroll, 2),
        "by_tier": by_tier,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--min-history", type=int, default=60)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config", "settings.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    season = args.season or int(cfg["prior_season"])

    games = store.load(f"history_{season}.json", None) or store.load(f"games_{season}.json", None)
    if not games:
        print(f"No cached data for {season}. Run `python -m pipeline.build --full` first,")
        print("which backfills the prior season into state/.")
        return 1

    res = run(games, cfg, min_history=args.min_history)
    if res.get("error"):
        print(res["error"])
        return 1

    print(f"\nWalk-forward backtest — {season}")
    print("=" * 58)
    print(f"{res['sides_priced']} market sides priced out of sample\n")
    print("Calibration (all priced sides, not just bets)")
    for c in res["calibration"]:
        print(f"  {c['bucket']:>9}  n={c['n']:<6} actual={c['actual']:>7.1%}  gap={c['gap']:+.3f}")
    print(f"  mean absolute gap: {res['mean_abs_calibration_gap']}")
    wr = "n/a" if res["win_rate"] is None else f"{res['win_rate']:.1%}"
    roi = "n/a" if res["roi"] is None else f"{res['roi']:.2%}"
    print("\nBets the model would have placed")
    print(f"  {res['bets']} bets, {res['wins']} wins ({wr})")
    print(f"  selection gap: {res['selection_gap']}  "
          "(actual win rate minus claimed probability, on bets only)")
    print(f"  staked {res['staked']}, P/L {res['pnl']}, ROI {roi}")
    print(f"  bankroll {res['starting_bankroll']} -> {res['ending_bankroll']}")
    print("\n  by tier")
    for t in ("BEST BET", "GOOD", "LEAN"):
        r = res["by_tier"].get(t)
        if r:
            troi = "n/a" if r["roi"] is None else f"{r['roi']:.2%}"
            print(f"    {t:<9} n={r['n']:<4} {r['w']}W  P/L {r['pnl']:>8}  ROI {troi}")
    print("\nA mean absolute calibration gap under ~0.03 means the probability")
    print("engine is sound. A positive ROI here is NOT evidence you will win —")
    print("one season is a few hundred bets, which is nowhere near enough to")
    print("separate a 2% edge from noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
