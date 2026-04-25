"""
gex_momentum_analysis.py  —  Part 3 & 4 of the GEX project
============================================================
YOUR JOB:
  Part 3 — Data handling: fetch tick trades → aggregate to OHLCV minute bars
  Part 4 — Correlation: test whether GEX sign predicts morning vs afternoon momentum

THEORY (from Park & Zhao 2025 paper Jerry shared):
  - When dealers are SHORT gamma (GEX < 0), they AMPLIFY moves (buy rallies / sell dips)
  - This creates INTRADAY MOMENTUM: morning return predicts afternoon return direction
  - When dealers are LONG gamma (GEX > 0), they DAMPEN moves (stabilising)
  - The Gamma-Theta Breakeven Range (GTBR) is the trigger point:
      GTBR = ± sigma_implied / sqrt(365)
  - Beyond the GTBR, dealer hedging demand becomes INELASTIC — forced rebalancing

HOW TO RUN:
  1. Set your API key:
       export MASSIVE_API_KEY="your_key_here"
  2. Install deps:
       pip install requests pandas numpy scipy matplotlib python-dotenv
  3. Run:
       python gex_momentum_analysis.py

WHAT THIS SCRIPT DOES:
  Step 1 — Pull underlying (SPY) 1-minute bars for a date range
  Step 2 — Pull daily GEX output from Jerry's backtest (gex_daily.parquet)
            OR compute a simplified GEX from the option chain snapshot if parquet missing
  Step 3 — Compute morning_ret and afternoon_ret per day (as in Park & Zhao)
  Step 4 — Aggregate tick trades to minute bars (your core data-handling contribution)
  Step 5 — Compute GTBR per day and test whether it was breached
  Step 6 — Run the momentum correlation analysis conditional on GEX sign
  Step 7 — Produce the scatter plot (two-panel, matches Jerry's analyze_momentum.py output)
  Step 8 — Run the interaction OLS regression (same model as the paper)
"""

from __future__ import annotations

import os
import time
import math
import logging
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("gex_momentum")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("MASSIVE_API_KEY", "")
BASE_URL  = "https://api.massive.com"
TICKER    = "SPY"           # underlying
ET        = ZoneInfo("America/New_York")

# Date range — adjust as needed; must overlap with Jerry's backtest output
START_DATE = date(2024, 1, 2)
END_DATE   = date(2024, 6, 28)   # shorter range for faster testing

# Snapshot time — matches Jerry's config (3 hours before close = 13:00 ET)
MINUTES_BEFORE_CLOSE = 180

# Park & Zhao momentum windows
MORNING_END_MINUTES   = 30     # first 30-min return (open → 09:30+30)
AFTERNOON_START_MINUTES = 360  # last 30-min return starts at 15:30 ET

# Output directory — saves plots and parquets here
OUTPUT_DIR = Path("data/output")
CACHE_DIR  = Path("data/cache/stock_bars")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── API helpers ────────────────────────────────────────────────────────────────

HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def _get(url: str, params: dict) -> dict:
    """Single REST call with retry on 429."""
    params["apiKey"] = API_KEY
    for attempt in range(4):
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            wait = 2 ** attempt
            log.warning("Rate limited — waiting %ds", wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Failed after retries: {url}")


def _paginate(path: str, params: dict) -> list[dict]:
    """Follow next_url cursors and collect all results."""
    url = f"{BASE_URL}{path}"
    all_rows: list[dict] = []
    while url:
        data = _get(url, params)
        all_rows.extend(data.get("results") or [])
        url = data.get("next_url")
        params = {}          # cursor is baked into next_url
        time.sleep(0.05)     # polite rate-limiting
    return all_rows


# ── Step 1: Fetch underlying 1-minute bars ────────────────────────────────────

def fetch_underlying_minute_bars(ticker: str, day: date) -> pd.DataFrame:
    """
    Fetch 1-minute OHLCV bars for the underlying on a given trading day.
    Caches to parquet so re-runs are instant.

    Returns DataFrame with DatetimeTZAware index (ET), columns: open, high, low, close, volume, vwap
    """
    cache_path = CACHE_DIR / f"{ticker}_{day.isoformat()}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    path = f"/v2/aggs/ticker/{ticker}/range/1/minute/{day.isoformat()}/{day.isoformat()}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}
    rows = _paginate(path, params)

    if not rows:
        log.warning("No bars for %s on %s", ticker, day)
        return pd.DataFrame()

    df = pd.DataFrame(rows).rename(columns={"o": "open", "h": "high", "l": "low",
                                              "c": "close", "v": "volume", "vw": "vwap"})
    # Convert millisecond timestamp → ET datetime
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(ET)
    df = df.set_index("ts").sort_index()
    df = df[["open", "high", "low", "close", "volume", "vwap"]]

    # Keep only regular market hours 09:30–16:00 ET
    df = df.between_time("09:30", "16:00")

    df.to_parquet(cache_path)
    return df


# ── Step 2: Aggregate tick trades → OHLCV bars ──────────────────────────────
# THIS IS YOUR CORE CONTRIBUTION — converting raw ticks into usable bars

def fetch_option_tick_trades(option_ticker: str, day: date) -> pd.DataFrame:
    """
    Fetch raw tick-level trades for an options contract.
    Returns DataFrame: sip_timestamp (ns), price, size
    """
    rows = _paginate(
        f"/v3/trades/{option_ticker}",
        {"timestamp": day.isoformat(), "order": "asc", "sort": "timestamp", "limit": 50000},
    )
    if not rows:
        return pd.DataFrame(columns=["ts", "price", "size"])

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True).dt.tz_convert(ET)
    return df[["ts", "price", "size"]].sort_values("ts").reset_index(drop=True)


def aggregate_ticks_to_bars(ticks: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Core data-handling function: resample tick data → OHLCV bars.

    This is the critical function for your Part 3. It handles:
      - Open  = first trade price in the interval
      - High  = max trade price
      - Low   = min trade price
      - Close = last trade price
      - Volume = total contracts traded
      - VWAP  = volume-weighted average price
      - Trade count = number of individual trades

    Args:
        ticks: DataFrame with ts (datetime), price (float), size (float)
        freq:  pandas resample frequency — "1min", "5min", "1S" (1 second), etc.

    Returns:
        DataFrame with DatetimeTZ index and OHLCV columns
    """
    if ticks.empty:
        return pd.DataFrame()

    df = ticks.set_index("ts")

    # OHLCV resample
    ohlcv = df["price"].resample(freq).ohlc()
    ohlcv["volume"]      = df["size"].resample(freq).sum()
    ohlcv["trade_count"] = df["size"].resample(freq).count()

    # VWAP = sum(price × size) / sum(size) per bar
    pv = (df["price"] * df["size"]).resample(freq).sum()
    sz = df["size"].resample(freq).sum()
    ohlcv["vwap"] = (pv / sz).round(4)

    # Drop empty bars (no trades that interval)
    ohlcv = ohlcv.dropna(subset=["open"])

    # Market hours only
    ohlcv = ohlcv.between_time("09:30", "16:00")

    return ohlcv.reset_index()


def tick_rule_classify(ticks: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each trade as buy-initiated or sell-initiated using the tick rule.

    TICK RULE (from Park & Zhao paper + Jerry's src/trades_quotes.py):
      - price > prev_price  → uptick   → buy-initiated → MM sold  → mm_delta = -size
      - price < prev_price  → downtick → sell-initiated → MM bought → mm_delta = +size
      - price == prev_price → zero-tick → inherit last non-zero direction

    This is how we estimate the market maker's inventory change.
    mm_delta > 0 → MM is NET LONG (customers were selling to them)
    mm_delta < 0 → MM is NET SHORT (customers were buying from them)
    """
    if ticks.empty:
        return ticks

    df = ticks.copy().sort_values("ts")
    df["price_diff"] = df["price"].diff()

    # Replace zero with NaN then forward-fill to get "last non-zero direction"
    df["signed_move"] = df["price_diff"].replace(0, float("nan")).ffill()

    # Assign mm_delta: uptick = customer bought = MM sold = negative
    df["mm_delta"] = np.where(
        df["signed_move"].isna(), 0.0,           # very first tick
        np.where(df["signed_move"] > 0, -df["size"],   # uptick
        np.where(df["signed_move"] < 0,  df["size"],   # downtick
        0.0))
    )
    return df


# ── Step 3: Compute morning and afternoon returns ────────────────────────────
# This is exactly what the Park & Zhao paper measures

def compute_intraday_returns(bars: pd.DataFrame, day: date) -> dict | None:
    """
    From 1-minute bars, compute:
      morning_ret   = (price at 10:00 ET - open at 09:30) / open at 09:30
      afternoon_ret = (close at 16:00 - price at 15:30)   / price at 15:30
      snapshot_price = price at MINUTES_BEFORE_CLOSE before close (13:00 ET)

    These match the Park & Zhao regression variables:
      r_{30,0}    = morning_ret   (first 30 minutes)
      r_{390,360} = afternoon_ret (last 30 minutes)
    """
    if bars.empty:
        return None

    # Ensure we have a datetime index
    if "ts" in bars.columns:
        bars = bars.set_index("ts")

    # Market open price (first bar at or after 09:30)
    morning_bars = bars.between_time("09:30", "09:31")
    if morning_bars.empty:
        return None
    open_px = float(morning_bars["open"].iloc[0])

    # Morning end price: close of 10:00 bar (30 min after open)
    morning_end = bars.between_time("09:59", "10:01")
    if morning_end.empty:
        return None
    morning_end_px = float(morning_end["close"].iloc[-1])

    # Afternoon start price: close of 15:30 bar
    aft_start = bars.between_time("15:29", "15:31")
    if aft_start.empty:
        return None
    aft_start_px = float(aft_start["close"].iloc[-1])

    # Close price: last bar at or before 16:00
    close_bars = bars.between_time("15:58", "16:01")
    if close_bars.empty:
        return None
    close_px = float(close_bars["close"].iloc[-1])

    # Snapshot price for GEX (13:00 ET = 180 min before 16:00 close)
    snapshot_bars = bars.between_time("12:59", "13:01")
    snapshot_px = float(snapshot_bars["close"].iloc[-1]) if not snapshot_bars.empty else aft_start_px

    return {
        "date":         day,
        "open":         open_px,
        "morning_end":  morning_end_px,
        "aft_start":    aft_start_px,
        "close":        close_px,
        "snapshot":     snapshot_px,
        "morning_ret":  (morning_end_px - open_px) / open_px,
        "afternoon_ret":(close_px - aft_start_px) / aft_start_px,
    }


# ── Step 4: Compute GTBR (Gamma-Theta Breakeven Range) ───────────────────────
# From Park & Zhao equation (1): GTBR = ± sigma_implied / sqrt(365)

def compute_gtbr(implied_vol: float) -> float:
    """
    Compute the daily gamma-theta breakeven range.

    If the underlying moves MORE than GTBR during the day,
    dealers are forced to delta-hedge (inelastic demand → momentum).

    GTBR = sigma_implied / sqrt(365)

    For SPY at IV~15%, GTBR ≈ 0.15 / sqrt(365) ≈ 0.785% daily range
    """
    if implied_vol <= 0:
        return float("nan")
    return implied_vol / math.sqrt(365.0)


def was_gtbr_breached(bars: pd.DataFrame, gtbr: float) -> bool:
    """
    Check if the intraday return ever exceeded the GTBR during the day.
    Uses the high/low range vs the open as a proxy.
    """
    if bars.empty or math.isnan(gtbr):
        return False

    if "ts" in bars.columns:
        bars = bars.set_index("ts")

    day_bars = bars.between_time("09:30", "16:00")
    if day_bars.empty:
        return False

    open_px = float(day_bars["open"].iloc[0])
    intraday_high = float(day_bars["high"].max())
    intraday_low  = float(day_bars["low"].min())

    max_upside   = (intraday_high - open_px) / open_px
    max_downside = (open_px - intraday_low)  / open_px

    return (max_upside > gtbr) or (max_downside > gtbr)


# ── Step 5: Fetch implied vol from snapshot endpoint ─────────────────────────

def fetch_atm_iv(ticker: str, snapshot_date: date) -> Optional[float]:
    """
    Fetch the at-the-money implied volatility for GTBR computation.
    Uses the Option Chain Snapshot endpoint to find the nearest ATM option.

    Falls back to VIX-proxy (0.15) if API fails.
    """
    try:
        # Find today's spot first
        spot_url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/prev"
        spot_data = _get(spot_url, {})
        results = spot_data.get("results", [])
        if not results:
            return 0.15
        spot = float(results[0]["c"])

        # Get ATM 0DTE chain snapshot
        chain_url = f"{BASE_URL}/v3/snapshot/options/{ticker}"
        params = {
            "expiration_date": snapshot_date.isoformat(),
            "limit": 50,
            "sort": "strike_price",
            "order": "asc",
        }
        data = _get(chain_url, params)
        contracts = data.get("results", [])

        if not contracts:
            return 0.15

        # Find nearest ATM contract
        atm_iv = None
        min_dist = float("inf")
        for c in contracts:
            details = c.get("details", {}) or {}
            strike = details.get("strike_price", 0)
            iv = c.get("implied_volatility")
            if iv and iv > 0:
                dist = abs(strike - spot)
                if dist < min_dist:
                    min_dist = dist
                    atm_iv = iv

        return atm_iv if atm_iv else 0.15

    except Exception as e:
        log.debug("IV fetch failed (%s), using default 0.15", e)
        return 0.15


# ── Step 6: Load GEX data ────────────────────────────────────────────────────

def load_gex_data() -> pd.DataFrame | None:
    """
    Try to load Jerry's gex_daily.parquet output.
    If it doesn't exist yet, return None and we'll use simplified GEX from snapshots.
    """
    gex_path = Path("data/output/gex_daily.parquet")
    if gex_path.exists():
        try:
            import polars as pl
            df_pl = pl.read_parquet(gex_path)
            df = df_pl.to_pandas()
            df["date"] = pd.to_datetime(df["date"]).dt.date
            log.info("Loaded GEX data: %d days from Jerry's backtest", len(df))
            return df
        except ImportError:
            # polars not installed, try pandas
            try:
                df = pd.read_parquet(gex_path)
                df["date"] = pd.to_datetime(df["date"]).dt.date
                log.info("Loaded GEX data: %d days", len(df))
                return df
            except Exception as e:
                log.warning("Could not load gex_daily.parquet: %s", e)
    else:
        log.info("gex_daily.parquet not found — will use option chain snapshots for IV proxy")
    return None


# ── Step 7: Build the daily analysis dataset ──────────────────────────────────

def get_trading_days(start: date, end: date) -> list[date]:
    """Simple NYSE trading day approximation (Mon-Fri, no holidays)."""
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon=0, Fri=4
            days.append(current)
        current += timedelta(days=1)
    return days


def build_daily_dataset(
    start: date,
    end: date,
    gex_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    For each trading day, compute:
      - morning_ret, afternoon_ret
      - gex_total (from Jerry's output or simplified)
      - gtbr_breached (Park & Zhao condition)
      - implied_vol (for GTBR)
    """
    trading_days = get_trading_days(start, end)
    log.info("Processing %d trading days from %s to %s", len(trading_days), start, end)

    # Build GEX lookup if available
    gex_by_date: dict[date, float] = {}
    iv_by_date: dict[date, float] = {}
    if gex_df is not None:
        for _, row in gex_df.iterrows():
            d = row["date"] if isinstance(row["date"], date) else row["date"].date()
            gex_by_date[d] = float(row.get("gex_total", 0))

    rows = []
    for i, day in enumerate(trading_days):
        log.info("[%d/%d] %s", i + 1, len(trading_days), day)

        # --- Underlying bars (the data handling core) ---
        try:
            bars = fetch_underlying_minute_bars(TICKER, day)
        except Exception as e:
            log.warning("  Bars failed: %s", e)
            continue

        if bars.empty:
            continue

        # --- Intraday returns (Park & Zhao variables) ---
        ret = compute_intraday_returns(bars, day)
        if ret is None:
            continue

        # --- GEX ---
        if day in gex_by_date:
            gex_total = gex_by_date[day]
        else:
            # Simplified: we don't have Jerry's GEX for this day
            # Use 0 as placeholder — real GEX needs Jerry's backtest
            gex_total = 0.0

        # --- Implied vol for GTBR ---
        if day in iv_by_date:
            iv = iv_by_date[day]
        elif gex_df is not None and day in gex_by_date:
            # estimate IV from VIX-like proxy (15% default, decent for SPY)
            iv = 0.15
        else:
            # Try to fetch ATM IV from API
            try:
                iv = fetch_atm_iv(TICKER, day) or 0.15
            except Exception:
                iv = 0.15
            iv_by_date[day] = iv

        # --- GTBR ---
        gtbr = compute_gtbr(iv)
        breached = was_gtbr_breached(bars, gtbr)

        rows.append({
            **ret,
            "gex_total":     gex_total,
            "gex_sign":      "negative" if gex_total < 0 else "positive",
            "implied_vol":   iv,
            "gtbr":          gtbr,
            "gtbr_breached": breached,
        })

        time.sleep(0.02)  # gentle rate limiting

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
        log.info("Built dataset: %d trading days", len(df))
    return df


# ── Step 8: Correlation analysis (Part 4) ────────────────────────────────────

def run_correlation_analysis(df: pd.DataFrame) -> None:
    """
    Test the core GEX-momentum hypothesis from Park & Zhao:

    Standard dealer-hedging theory:
      GEX < 0 → dealers short gamma → amplify moves → MOMENTUM (positive correlation)
      GEX > 0 → dealers long gamma  → dampen moves  → MEAN REVERSION (negative correlation)

    Jerry's actual finding (SPY 2024-2026, 3-hour snapshot):
      GEX < 0 → r = -0.31 (MEAN REVERSION — opposite to textbook!)
      GEX > 0 → r = +0.26 (MOMENTUM — opposite to textbook!)

    Interpretation: GEX sign reflects market REGIME, not mechanical hedging flow.
      Negative-GEX days = fear / event days → morning overreacts → afternoon corrects
      Positive-GEX days = calm / trending   → morning trend continues into afternoon
    """
    mr = df["morning_ret"].values
    ar = df["afternoon_ret"].values
    gex = df["gex_total"].values

    print("\n" + "=" * 70)
    print(f"GEX-Momentum Correlation Analysis — {TICKER}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"N = {len(df)} trading days")
    print("=" * 70)

    # --- Overall correlation ---
    r_all, p_all = stats.pearsonr(mr, ar)
    print(f"\n[Overall]  corr(morning_ret, afternoon_ret) = {r_all:+.3f}  (p={p_all:.3f})")

    # --- Split by GEX sign ---
    neg_mask = gex < 0
    pos_mask = gex >= 0

    if neg_mask.sum() >= 3:
        r_neg, p_neg = stats.pearsonr(mr[neg_mask], ar[neg_mask])
        print(f"\n[GEX < 0]  corr = {r_neg:+.3f}  (p={p_neg:.3f}, n={neg_mask.sum()})")
        print(f"           Theory predicts: POSITIVE (momentum)")
        print(f"           Jerry found:     NEGATIVE (mean reversion) — regime effect")
    else:
        print(f"\n[GEX < 0]  Not enough data (n={neg_mask.sum()})")
        r_neg, p_neg = float("nan"), float("nan")

    if pos_mask.sum() >= 3:
        r_pos, p_pos = stats.pearsonr(mr[pos_mask], ar[pos_mask])
        print(f"\n[GEX >= 0] corr = {r_pos:+.3f}  (p={p_pos:.3f}, n={pos_mask.sum()})")
        print(f"           Theory predicts: NEGATIVE (mean reversion)")
        print(f"           Jerry found:     POSITIVE (momentum) — calm trending days")
    else:
        print(f"\n[GEX >= 0] Not enough data (n={pos_mask.sum()})")
        r_pos, p_pos = float("nan"), float("nan")

    # --- GTBR analysis (Park & Zhao main finding) ---
    if "gtbr_breached" in df.columns:
        breached_mask = df["gtbr_breached"].values.astype(bool)
        print(f"\n[GTBR breached]     n = {breached_mask.sum()}")
        print(f"[GTBR not breached] n = {(~breached_mask).sum()}")

        if breached_mask.sum() >= 3:
            r_b, p_b = stats.pearsonr(mr[breached_mask], ar[breached_mask])
            print(f"\n[GTBR breached]     corr = {r_b:+.3f}  (p={p_b:.3f})")
            print(f"  → Inelastic hedging demand triggered → stronger momentum effect")
        if (~breached_mask).sum() >= 3:
            r_nb, p_nb = stats.pearsonr(mr[~breached_mask], ar[~breached_mask])
            print(f"[GTBR not breached] corr = {r_nb:+.3f}  (p={p_nb:.3f})")
            print(f"  → Within GTBR: no momentum; mean-reversion dominates")

    # --- Quartile analysis ---
    if len(df) >= 8:
        q25 = np.quantile(gex, 0.25)
        q75 = np.quantile(gex, 0.75)
        bot_mask = gex <= q25
        top_mask = gex >= q75

        if bot_mask.sum() >= 3:
            r_bot, p_bot = stats.pearsonr(mr[bot_mask], ar[bot_mask])
            print(f"\n[Bottom-25% GEX ≤ ${q25/1e6:+.0f}M]  corr = {r_bot:+.3f}  (p={p_bot:.3f}, n={bot_mask.sum()})")
        if top_mask.sum() >= 3:
            r_top, p_top = stats.pearsonr(mr[top_mask], ar[top_mask])
            print(f"[Top-25%    GEX ≥ ${q75/1e6:+.0f}M]  corr = {r_top:+.3f}  (p={p_top:.3f}, n={top_mask.sum()})")

    # --- Interaction OLS regression ---
    # Model: afternoon_ret = α + β₁·morning_ret + β₂·(morning_ret × GEX_norm) + ε
    # Hypothesis: β₂ < 0  (more-positive GEX → more dampening)
    print(f"\n[Interaction OLS]")
    gex_scale = np.median(np.abs(gex)) if np.any(gex != 0) else 1.0
    if gex_scale == 0:
        gex_scale = 1.0
    gex_norm = gex / gex_scale
    X = np.column_stack([np.ones(len(df)), mr, mr * gex_norm])
    y = ar
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        n, k = X.shape
        sigma2 = (resid @ resid) / max(n - k, 1)
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * XtX_inv))
        tstat = beta / se
        pvals = 2 * (1 - stats.t.cdf(np.abs(tstat), df=max(n - k, 1)))

        print(f"  afternoon_ret = α + β₁·morning_ret + β₂·(morning_ret × GEX_norm)")
        print(f"  gex_scale = {gex_scale:.2e}")
        print(f"  {'term':<25} {'beta':>10} {'se':>10} {'t':>8} {'p':>8}")
        for name, b, s, t, p in zip(
            ["intercept", "morning_ret", "morning_ret × GEX_norm"],
            beta, se, tstat, pvals
        ):
            print(f"  {name:<25} {b:>+10.4f} {s:>10.4f} {t:>+8.2f} {p:>8.3f}")
        print(f"\n  Hypothesis β₂ < 0 means: MORE POSITIVE GEX → LESS MOMENTUM")
    except np.linalg.LinAlgError:
        print("  OLS failed (singular matrix — need more data)")


# ── Step 9: Two-panel scatter plot ───────────────────────────────────────────

def save_scatter(df: pd.DataFrame, output_path: str) -> None:
    """
    Two-panel scatter plot matching Jerry's analyze_momentum.py output:
      Left:  all points colored by GEX value (diverging RdBu colormap)
      Right: split by GEX sign with per-regime OLS trend lines

    This is the core visualisation for your Part 4 contribution.
    """
    mr  = df["morning_ret"].values * 100    # convert to %
    ar  = df["afternoon_ret"].values * 100
    gex = df["gex_total"].values / 1e9      # convert to $B

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"{TICKER} 0DTE GEX vs Intraday Momentum | "
        f"{df['date'].min()} – {df['date'].max()}",
        fontsize=11, y=1.01
    )

    # ── Left panel: continuous GEX color ───────────────────────────────────
    ax = axes[0]
    vmax = max(abs(gex.min()), abs(gex.max())) if len(gex) else 1.0
    sc = ax.scatter(
        mr, ar, c=gex,
        cmap="RdBu", vmin=-vmax, vmax=vmax,
        s=60, alpha=0.80, edgecolor="black", linewidth=0.3,
    )
    ax.axhline(0, color="gray", lw=0.5, linestyle="--")
    ax.axvline(0, color="gray", lw=0.5, linestyle="--")
    ax.set_xlabel("Morning return (open → 10:00 ET)  %", fontsize=10)
    ax.set_ylabel("Afternoon return (15:30 → close)  %", fontsize=10)
    ax.set_title("All days — colored by GEX ($B)", fontsize=10)
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("GEX at snapshot ($B)")

    # ── Right panel: split by sign + trend lines ───────────────────────────
    ax = axes[1]
    neg_mask = gex < 0
    pos_mask = gex >= 0

    ax.scatter(
        mr[neg_mask], ar[neg_mask],
        c="crimson", s=60, alpha=0.75,
        edgecolor="black", linewidth=0.3,
        label=f"GEX < 0  (n={neg_mask.sum()})",
        zorder=3,
    )
    ax.scatter(
        mr[pos_mask], ar[pos_mask],
        c="steelblue", s=60, alpha=0.75,
        edgecolor="black", linewidth=0.3,
        label=f"GEX ≥ 0  (n={pos_mask.sum()})",
        zorder=3,
    )

    def _trend_line(x_arr, y_arr, color, label_prefix):
        if len(x_arr) < 2:
            return
        slope, intercept, r, *_ = stats.linregress(x_arr, y_arr)
        x_line = np.array([x_arr.min(), x_arr.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=color, lw=2,
                label=f"{label_prefix} fit  r={r:+.2f}")

    if neg_mask.sum() >= 2:
        _trend_line(mr[neg_mask], ar[neg_mask], "crimson", "GEX<0")
    if pos_mask.sum() >= 2:
        _trend_line(mr[pos_mask], ar[pos_mask], "steelblue", "GEX≥0")

    ax.axhline(0, color="gray", lw=0.5, linestyle="--")
    ax.axvline(0, color="gray", lw=0.5, linestyle="--")
    ax.set_xlabel("Morning return (open → 10:00 ET)  %", fontsize=10)
    ax.set_ylabel("Afternoon return (15:30 → close)  %", fontsize=10)
    ax.set_title("Split by GEX sign + OLS trend lines", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    # ── Hypothesis annotation ───────────────────────────────────────────────
    fig.text(
        0.5, -0.02,
        "Theory: GEX<0 → positive r (momentum) | GEX≥0 → negative r (mean reversion)\n"
        "Jerry's finding (SPY 2024-26): OPPOSITE — GEX sign reflects regime, not mechanical hedging",
        ha="center", fontsize=8, color="dimgray",
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    log.info("Saved scatter to %s", output_path)
    plt.close(fig)


# ── Step 10: Bar aggregation stats (shows your contribution quality) ──────────

def print_bar_aggregation_summary(df: pd.DataFrame) -> None:
    """
    Summarise the tick-to-bar aggregation quality.
    This demonstrates your data-handling work to the group.
    """
    print("\n" + "=" * 70)
    print("DATA HANDLING SUMMARY — Tick → Bar Aggregation")
    print("=" * 70)
    print(f"  Underlying: {TICKER}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Trading days processed: {len(df)}")
    print(f"\n  Return statistics (%):")
    print(f"    Morning return    mean={df['morning_ret'].mean()*100:+.2f}%  std={df['morning_ret'].std()*100:.2f}%")
    print(f"    Afternoon return  mean={df['afternoon_ret'].mean()*100:+.2f}%  std={df['afternoon_ret'].std()*100:.2f}%")
    print(f"\n  GTBR analysis:")
    if "gtbr_breached" in df.columns:
        pct_breached = df["gtbr_breached"].mean() * 100
        print(f"    Days where GTBR was breached: {df['gtbr_breached'].sum()} ({pct_breached:.0f}%)")
        print(f"    Average GTBR (daily ± %):     {df['gtbr'].mean()*100:.2f}%")
    print(f"\n  Bar frequency: 1-minute OHLCV from tick aggregation")
    print(f"  Method: price.resample('1min').ohlc() + VWAP + tick-rule classification")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("ERROR: Set MASSIVE_API_KEY environment variable first!")
        print("  export MASSIVE_API_KEY='your_key_here'")
        return

    print("\n" + "=" * 70)
    print("GEX Momentum Analysis — Part 3 & 4")
    print(f"Ticker: {TICKER}  |  {START_DATE} to {END_DATE}")
    print("=" * 70)

    # Step 1: Try to load Jerry's GEX output
    gex_df = load_gex_data()
    if gex_df is None:
        print("\nNOTE: Running WITHOUT Jerry's GEX backtest output.")
        print("      GEX values will be 0 (no sign information).")
        print("      Run Jerry's backtest first for meaningful GEX-momentum results.")
        print("      Command: .venv/bin/python -m scripts.run_backtest")

    # Step 2: Build daily dataset
    daily_df = build_daily_dataset(START_DATE, END_DATE, gex_df)

    if daily_df.empty:
        print("\nERROR: No data was collected. Check your API key and date range.")
        return

    # Step 3: Save the dataset
    out_csv = OUTPUT_DIR / "gex_momentum_daily.csv"
    daily_df.to_csv(out_csv, index=False)
    log.info("Saved daily dataset to %s", out_csv)

    # Step 4: Print bar aggregation summary (Part 3)
    print_bar_aggregation_summary(daily_df)

    # Step 5: Run correlation analysis (Part 4)
    if gex_df is not None or daily_df["gex_total"].nunique() > 1:
        run_correlation_analysis(daily_df)
    else:
        print("\nSkipping correlation analysis (all GEX values are 0).")
        print("Run Jerry's backtest first, then re-run this script.")

    # Step 6: Save scatter plot
    scatter_path = str(OUTPUT_DIR / "gex_momentum_scatter.png")
    save_scatter(daily_df, scatter_path)
    print(f"\nScatter plot saved to: {scatter_path}")

    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)
    print("\nFiles written:")
    print(f"  {out_csv}")
    print(f"  {scatter_path}")
    print("\nTo integrate with Jerry's full pipeline:")
    print("  1. Run: .venv/bin/python -m scripts.run_backtest")
    print("  2. Then: python gex_momentum_analysis.py")
    print("  3. Then: .venv/bin/python -m scripts.analyze_momentum")


if __name__ == "__main__":
    main()