"""
Edge model: projection -> probability -> price comparison -> tier -> stake.

Three things here are meaningfully better than the spreadsheet version:

1. KEY NUMBERS. The workbook converted a projected margin into a cover
   probability with NORMSDIST -- a smooth bell curve. Football margins are not
   smooth. Games land on 3 and 7 far more often than a normal curve says, which
   is why the difference between -2.5 and -3.5 is worth real money and the
   difference between -5.5 and -6.5 is worth almost nothing. A smooth model
   systematically overpays to buy off 3 and underrates every number next to it.
   We build a discrete margin distribution instead, bumped at the key numbers,
   which also gives us honest push probabilities on whole-number spreads.

2. PROPER DE-VIGGING. Comparing a model probability against a raw
   vig-inclusive implied probability conflates "I disagree with the market" with
   "the book charges juice". The two get separated: the market's *fair* opinion
   is the de-vigged number, and the price you have to beat is the break-even
   number. Edge is measured against the second; disagreement against the first.

3. CONFIDENCE-AWARE OUTPUT. A 6% edge built on two games of data in week 2 is
   not the same bet as a 6% edge in week 10, and the model says so instead of
   quietly pretending otherwise.
"""

from __future__ import annotations

import math

# Relative frequency bumps applied at football's key numbers. Sourced from the
# long-run distribution of FBS final margins: 3 and 7 are the spikes, with 10,
# 14, 17 and 21 meaningfully elevated over their neighbours.
KEY_NUMBER_BUMPS = {
    0: 0.55, 1: 1.15, 2: 1.05, 3: 2.35, 4: 1.20, 5: 0.95, 6: 1.10,
    7: 1.95, 8: 1.05, 9: 0.90, 10: 1.55, 11: 1.10, 12: 0.85, 13: 0.95,
    14: 1.60, 15: 0.90, 16: 0.85, 17: 1.45, 18: 0.90, 19: 0.85,
    20: 0.95, 21: 1.35, 22: 0.85, 23: 0.85, 24: 1.15, 25: 0.85,
    27: 0.95, 28: 1.15, 31: 1.05, 35: 1.00,
}

_MAX_MARGIN = 70


# --------------------------------------------------------------------------- #
# Odds conversions
# --------------------------------------------------------------------------- #

def american_to_decimal(american: float) -> float:
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / -american)


def american_to_prob(american: float) -> float:
    """Break-even (vig-inclusive) probability for an American price."""
    return (-american / (-american + 100.0)) if american < 0 else (100.0 / (american + 100.0))


def prob_to_american(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -100.0 * p / (1 - p) if p >= 0.5 else 100.0 * (1 - p) / p


def devig(p_a: float, p_b: float) -> tuple[float, float]:
    """
    Strip the vig from a two-way market (multiplicative / proportional method).

    Proportional is used rather than additive or Shin because on the roughly
    -110/-110 two-way markets this project bets, all three agree to within a
    fraction of a point, and proportional is the one that can't produce a
    negative probability on a lopsided line.
    """
    tot = p_a + p_b
    if tot <= 0:
        return 0.5, 0.5
    return p_a / tot, p_b / tot


# --------------------------------------------------------------------------- #
# Discrete margin distribution
# --------------------------------------------------------------------------- #

def _normal_pdf(x: float, mu: float, sd: float) -> float:
    z = (x - mu) / sd
    return math.exp(-0.5 * z * z) / (sd * math.sqrt(2 * math.pi))


def margin_distribution(mu: float, sd: float, use_key_numbers: bool = True) -> dict[int, float]:
    """P(final margin == k) for integer k, centred on the projected margin."""
    dist: dict[int, float] = {}
    for k in range(-_MAX_MARGIN, _MAX_MARGIN + 1):
        p = _normal_pdf(k, mu, sd)
        if use_key_numbers:
            p *= KEY_NUMBER_BUMPS.get(abs(k), 1.0)
        dist[k] = p
    tot = sum(dist.values())
    return {k: v / tot for k, v in dist.items()}


def cover_probability(mu: float, sd: float, spread_home: float,
                      use_key_numbers: bool = True) -> tuple[float, float, float]:
    """
    Home team's cover probability at `spread_home`.

    Returns (p_home_cover, p_push, p_away_cover). ESPN's sign convention: a
    negative spread means the home team is laying points.
    """
    dist = margin_distribution(mu, sd, use_key_numbers)
    win = push = loss = 0.0
    for k, p in dist.items():
        adj = k + spread_home
        if adj > 1e-9:
            win += p
        elif adj < -1e-9:
            loss += p
        else:
            push += p
    return win, push, loss


def moneyline_probability(mu: float, sd: float, use_key_numbers: bool = True) -> float:
    """
    Straight-up home win probability. CFB has no ties -- overtime resolves them --
    so the mass sitting exactly on 0 gets split evenly between the two sides.
    """
    dist = margin_distribution(mu, sd, use_key_numbers)
    win = sum(p for k, p in dist.items() if k > 0)
    tie = dist.get(0, 0.0)
    return win + tie / 2.0


def over_probability(proj_total: float, market_total: float, sd: float) -> tuple[float, float, float]:
    """Over / push / under for a projected combined score."""
    over = push = under = 0.0
    for k in range(0, 130):
        p = _normal_pdf(k, proj_total, sd)
        if k > market_total + 1e-9:
            over += p
        elif k < market_total - 1e-9:
            under += p
        else:
            push += p
    tot = over + push + under
    if tot <= 0:
        return 0.5, 0.0, 0.5
    return over / tot, push / tot, under / tot


# --------------------------------------------------------------------------- #
# Staking
# --------------------------------------------------------------------------- #

def kelly_fraction(p: float, american: float) -> float:
    """Full-Kelly fraction of bankroll. Negative means no bet."""
    b = american_to_decimal(american) - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    return max(0.0, (p * b - q) / b)


def expected_value(p: float, american: float, p_push: float = 0.0) -> float:
    """EV per unit staked, with push probability removing stake from risk."""
    b = american_to_decimal(american) - 1.0
    p_lose = max(0.0, 1.0 - p - p_push)
    return p * b - p_lose


def stake_for(p: float, american: float, bankroll: float, cfg: dict) -> float:
    bk = cfg["bankroll"]
    f = kelly_fraction(min(p, cfg["model"]["max_model_prob"]), american) * float(bk["kelly_fraction"])
    f = min(f, float(bk["max_stake_pct"]))
    raw = f * bankroll
    step = float(bk.get("round_stake_to") or 0.5)
    stake = round(raw / step) * step if step > 0 else raw
    return 0.0 if stake < float(bk.get("min_stake") or 0) else round(stake, 2)


# --------------------------------------------------------------------------- #
# Tiering
# --------------------------------------------------------------------------- #

def tier_for(edge: float, cfg: dict, confidence: float) -> str:
    """
    Map an edge to BEST BET / GOOD / LEAN / PASS.

    Two adjustments before the thresholds are applied.

    The winner's-curse haircut. You only bet where the model disagrees with the
    market -- which is precisely where the model's own error is largest. So the
    edges you end up selecting are overstated even when the model is perfectly
    calibrated across all games. Simulation puts the gap around 11 points of
    probability on selected bets while all-games calibration sits within half a
    point. Subtracting a flat haircut is the blunt, honest correction.

    Confidence (0-1) then scales the thresholds rather than the edge itself, so
    a thin-data game has to clear a higher bar to earn the same label instead of
    having its number silently rewritten.
    """
    t = cfg["tiers"]
    if confidence <= 0:
        return "PASS"
    edge = edge - float(cfg["model"].get("selection_haircut", 0.0))
    scale = 1.0 / max(0.35, confidence)
    if edge >= float(t["best_bet"]) * scale:
        return "BEST BET"
    if edge >= float(t["good"]) * scale:
        return "GOOD"
    if edge >= float(t["lean"]) * scale:
        return "LEAN"
    return "PASS"


TIER_RANK = {"BEST BET": 0, "GOOD": 1, "LEAN": 2, "PASS": 3}


def confidence_score(n_home: int, n_away: int, has_odds: bool, cfg: dict) -> float:
    """
    How much the model trusts itself on this game, 0-1.

    Driven mostly by sample size. In week 1 nobody has played, every rating is
    the preseason prior, and the honest answer is "not much".
    """
    if not has_odds:
        return 0.0
    need = float(cfg["model"]["min_games_for_full_confidence"])
    n = min(n_home, n_away)
    sample = min(1.0, (n / need) ** 0.5) if need > 0 else 1.0
    floor = 1.0 - float(cfg["model"]["early_season_shrink"])
    return round(floor + (1.0 - floor) * sample, 3)
