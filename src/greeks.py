"""Implied volatility and gamma via QuantLib.

For 0DTE at the 15:30 snapshot (≤ 30 min to expiry) the American premium over
a European Black-Scholes price on a non-dividend-paying stock is immaterial
(there is no early-exercise benefit without dividends, and we explicitly skip
ex-dividend windows). We therefore price as European — closed-form, numerically
stable, and fast — and invert for IV using QuantLib's Brent solver.

Inputs per contract:
    spot      : underlying price at snapshot (float)
    strike    : option strike (float)
    T         : time to expiry in years (float, > 0)
    r         : risk-free rate, continuous, decimal
    mid       : observed option mid price
    is_call   : True for calls, False for puts
    q         : continuous dividend yield (default 0; we skip ex-div weeks)

Outputs:
    iv        : implied volatility, decimal (e.g. 0.45 = 45%)
    gamma     : ∂²V/∂S², per-share basis
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import QuantLib as ql

from config import IV_ACCURACY, IV_MAX, IV_MAX_ITERATIONS, IV_MIN

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GreeksResult:
    iv: float
    gamma: float


def _price_and_gamma(
    spot: float,
    strike: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
) -> tuple[float, float]:
    """European Black-Scholes price and gamma.

    Uses QuantLib.BlackCalculator, which takes T directly — so we avoid any
    eval-date/expiry-date rounding that could matter for 0DTE intraday T.
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, (spot - strike) if is_call else (strike - spot))
        return intrinsic, 0.0

    forward = spot * math.exp((r - q) * T)
    discount = math.exp(-r * T)
    stddev = sigma * math.sqrt(T)

    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if is_call else ql.Option.Put, strike
    )
    calc = ql.BlackCalculator(payoff, forward, stddev, discount)
    return calc.value(), calc.gamma(spot)


def _no_arb_bounds(
    spot: float, strike: float, T: float, r: float, q: float, is_call: bool
) -> tuple[float, float]:
    """Lower and upper no-arbitrage bounds for a European option premium."""
    discount = math.exp(-r * T)
    forward = spot * math.exp((r - q) * T)
    if is_call:
        lower = max(0.0, discount * (forward - strike))
        upper = spot
    else:
        lower = max(0.0, discount * (strike - forward))
        upper = strike * discount
    return lower, upper


def implied_vol_and_gamma(
    *,
    spot: float,
    strike: float,
    T: float,
    r: float,
    mid: float,
    is_call: bool,
    q: float = 0.0,
) -> GreeksResult | None:
    """Solve for σ such that BS(σ) == mid, then return (σ, Γ).

    Returns None and logs a warning when inputs are bad, the mid is outside
    no-arb bounds, or Brent fails to converge.
    """
    if T <= 0 or spot <= 0 or strike <= 0 or mid <= 0:
        logger.warning(
            "IV inputs invalid: S=%s K=%s T=%s mid=%s", spot, strike, T, mid
        )
        return None

    lower, upper = _no_arb_bounds(spot, strike, T, r, q, is_call)
    if not (lower - 1e-6 <= mid <= upper + 1e-6):
        # Common for deep ITM options near expiry (stale/wide quotes pull
        # the mid below intrinsic). Their gamma contribution is ≈0 anyway.
        logger.debug(
            "IV: mid %.4f outside no-arb [%.4f, %.4f] "
            "(S=%.2f K=%.2f T=%.6f is_call=%s)",
            mid, lower, upper, spot, strike, T, is_call,
        )
        return None

    solver = ql.Brent()
    solver.setMaxEvaluations(IV_MAX_ITERATIONS)

    def objective(sigma: float) -> float:
        price, _ = _price_and_gamma(spot, strike, T, r, q, sigma, is_call)
        return price - mid

    try:
        iv = solver.solve(objective, IV_ACCURACY, 0.5, IV_MIN, IV_MAX)
    except RuntimeError as e:
        # Common for wide-spread quotes on illiquid wings where the
        # quoted mid has more time value than σ ∈ [IV_MIN, IV_MAX] can
        # produce at this T. Contract is correctly skipped; gamma at
        # these strikes is small so the GEX impact is negligible.
        logger.debug(
            "IV solver failed: %s (S=%.2f K=%.2f T=%.6f mid=%.4f is_call=%s)",
            e, spot, strike, T, mid, is_call,
        )
        return None

    _, gamma = _price_and_gamma(spot, strike, T, r, q, iv, is_call)
    return GreeksResult(iv=iv, gamma=gamma)
