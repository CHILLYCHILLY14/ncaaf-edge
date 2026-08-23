"""
The full prediction record — every market the model has ever priced, graded,
whether or not it was ever bet.

The Bet Ledger only tells you about the bets you actually placed, and that's
inherently a biased sample: it's exactly the games where the model disagreed
most with the market, which is where its own error is largest (the winner's
curse the tiering already corrects for). A model that only ever gets judged on
its bets can't be told apart from a model that got lucky on a small selected
slice.

This module keeps the other, much larger, much less biased dataset: what the
model said about EVERY game it saw a price for -- PASS included -- logged once
at first sight and graded once the result is in, with the same discipline as
the ledger (never re-priced after the fact). Over a season that's thousands of
graded predictions instead of a few dozen bets, which is what it actually takes
to tell a 3-point calibration gap from noise.
"""

from __future__ import annotations

from . import store


def pred_key(game_id: str, market: str, side: str) -> str:
    return f"{game_id}:{market}:{side}"


def log_prediction(preds: dict, cand: dict) -> bool:
    """Record one priced market side, once. Returns True if newly logged."""
    key = pred_key(cand["game_id"], cand["market"], cand["side"])
    if key in preds:
        return False
    preds[key] = {
        "pred_id": key,
        "game_id": cand["game_id"],
        "logged_at": store.now_iso(),
        "game_date": cand["game_date"],
        "week": cand.get("week"),
        "matchup": cand["matchup"],
        "market": cand["market"],
        "side": cand["side"],
        "pick": cand["pick"],
        "line": cand.get("line"),
        "price": cand["price"],
        "book": cand.get("book"),
        "model_prob": round(cand["model_prob"], 4),
        "raw_model_prob": round(cand.get("raw_model_prob", cand["model_prob"]), 4),
        "market_fair_prob": round(cand.get("market_fair_prob", 0.0), 4),
        "breakeven": round(cand["breakeven"], 4),
        "edge": round(cand["edge"], 4),
        "tier": cand["tier"],
        "confidence": cand.get("confidence"),
        "odds_verified": cand.get("odds_verified") is True,
        "result": "Pending",
        "correct": None,
        "graded_at": None,
    }
    return True


def _outcome(pred: dict, game: dict) -> str | None:
    hs, as_ = game.get("home_score"), game.get("away_score")
    if hs is None or as_ is None:
        return None
    margin, total = hs - as_, hs + as_
    market, side, line = pred["market"], pred["side"], pred.get("line")
    if market == "ML":
        if margin == 0:
            return "Push"
        won = (margin > 0) if side == "home" else (margin < 0)
        return "Win" if won else "Loss"
    if market == "ATS":
        if line is None:
            return None
        adj = margin + float(line)
        if abs(adj) < 1e-9:
            return "Push"
        won = (adj > 0) if side == "home" else (adj < 0)
        return "Win" if won else "Loss"
    if market == "TOTAL":
        if line is None:
            return None
        if abs(total - float(line)) < 1e-9:
            return "Push"
        won = (total > float(line)) if side == "over" else (total < float(line))
        return "Win" if won else "Loss"
    return None


def grade_all(preds: dict, games_by_id: dict) -> int:
    """Grade every pending prediction whose game finished. Returns count graded."""
    n = 0
    for pred in preds.values():
        if pred.get("result") not in (None, "Pending"):
            continue
        g = games_by_id.get(pred["game_id"])
        if not g or not g.get("completed"):
            continue
        result = _outcome(pred, g)
        if result is None:
            continue
        pred["result"] = result
        pred["correct"] = result == "Win"
        pred["graded_at"] = store.now_iso()
        pred["final_score"] = f'{g["away"]["abbr"]} {g["away_score"]} - {g["home"]["abbr"]} {g["home_score"]}'
        n += 1
    return n


def _calibration(rows: list[dict]) -> list[dict]:
    buckets: dict[int, list[int]] = {}
    for p in rows:
        k = int(float(p["model_prob"]) * 10)
        buckets.setdefault(k, [0, 0])
        buckets[k][0] += 1
        buckets[k][1] += int(p["result"] == "Win")
    out = []
    for k in sorted(buckets):
        n, w = buckets[k]
        if n < 5:
            continue
        out.append({"bucket": f"{k*10}-{k*10+10}%", "n": n,
                    "predicted": round(k / 10 + 0.05, 4), "actual": round(w / n, 4),
                    "gap": round(w / n - (k / 10 + 0.05), 4)})
    return out


def _brier(rows: list[dict]) -> float | None:
    if not rows:
        return None
    s = sum((float(p["model_prob"]) - (1.0 if p["result"] == "Win" else 0.0)) ** 2 for p in rows)
    return round(s / len(rows), 4)


def summarise(preds: dict) -> dict:
    """
    The model's full track record, independent of what was ever staked.

    This is the number to trust over the ledger's own calibration table once
    enough of the season has played out -- it's a much larger, much less
    selected sample of the same underlying question: when this model says X%,
    does X% actually happen?
    """
    all_rows = list(preds.values())
    settled = [p for p in all_rows if p.get("result") in ("Win", "Loss")]
    pending = [p for p in all_rows if p.get("result") == "Pending"]

    by_tier: dict[str, dict] = {}
    for p in settled:
        t = p.get("tier") or "PASS"
        row = by_tier.setdefault(t, {"n": 0, "wins": 0})
        row["n"] += 1
        row["wins"] += int(p["result"] == "Win")
    for row in by_tier.values():
        row["win_rate"] = round(row["wins"] / row["n"], 4) if row["n"] else None

    by_market: dict[str, dict] = {}
    for p in settled:
        m = p.get("market") or "?"
        row = by_market.setdefault(m, {"n": 0, "wins": 0})
        row["n"] += 1
        row["wins"] += int(p["result"] == "Win")
        row.setdefault("brier_rows", []).append(p)
    for row in by_market.values():
        row["win_rate"] = round(row["wins"] / row["n"], 4) if row["n"] else None
        row["brier"] = _brier(row.pop("brier_rows"))

    by_week: dict[str, dict] = {}
    for p in settled:
        wk = str(p.get("week") or "?")
        row = by_week.setdefault(wk, {"n": 0, "wins": 0})
        row["n"] += 1
        row["wins"] += int(p["result"] == "Win")
        row.setdefault("rows", []).append(p)
    week_trend = []
    for wk in sorted(by_week, key=lambda x: (len(x), x)):
        row = by_week[wk]
        week_trend.append({"week": wk, "n": row["n"],
                           "win_rate": round(row["wins"] / row["n"], 4) if row["n"] else None,
                           "brier": _brier(row["rows"])})
    for row in by_week.values():
        row.pop("rows", None)

    return {
        "total_logged": len(all_rows),
        "settled": len(settled),
        "pending": len(pending),
        "brier": _brier(settled),
        "calibration": _calibration(settled),
        "by_tier": by_tier,
        "by_market": {k: {kk: vv for kk, vv in v.items()} for k, v in by_market.items()},
        "week_trend": week_trend,
    }
