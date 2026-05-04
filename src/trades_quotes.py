from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import polars as pl

from config import CLASSIFICATION_MODE
from src import storage
from src.client import PolygonClient

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION CODES — non-bona-fide prints to strip before classification
# ─────────────────────────────────────────────────────────────────────────────

_EXCLUDE_CONDITIONS: set[str] = {
    "MLEG",    # multi-leg spread
    "MLEGA",   # multi-leg auction
    "AVG",     # average price / bunched
    "OPD",     # auction open
    "MOO",     # market on open
    "MOC",     # market on close
    "LOO",     # limit on open
    "LOC",     # limit on close
    "CANC",    # cancelled
    "OOO",     # out-of-sequence
    "DP",      # derivatively priced
    "SO",      # stock-option combo
    "SOLEG",   # stock-option combo leg
}

# ─────────────────────────────────────────────────────────────────────────────
# TWAP / VWAP ALGO DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_ALGO_WINDOW      = 10    # sliding window size (number of trades)
_ALGO_SIZE_CV_MAX = 0.20  # max coefficient of variation on trade sizes
_ALGO_TIME_CV_MAX = 0.35  # max CV on inter-trade time gaps
_ALGO_MIN_TRADES  = 5     # min trades needed to check


def flag_algo_trades(trades: pl.DataFrame) -> pl.DataFrame:
    """Flag trades that look like TWAP/VWAP algorithmic execution.

    A large player using TWAP/VWAP breaks one big order into many small
    trades at regular time intervals with similar sizes. These are a large
    directional bet disguised as retail flow — NOT MM hedging demand.
    We detect them by sliding a window and flagging runs where:
      1. Trade sizes are suspiciously uniform  (CV < _ALGO_SIZE_CV_MAX)
      2. Inter-trade gaps are suspiciously regular (CV < _ALGO_TIME_CV_MAX)
    Flagged trades are excluded from sum_mm_position() alongside block trades.
    """
    if trades.is_empty() or len(trades) < _ALGO_MIN_TRADES:
        return trades.with_columns(pl.lit(False).alias("is_algo_trade"))

    df = trades.sort("sip_timestamp")
    sizes = df["size"].to_list()
    ts    = df["sip_timestamp"].cast(pl.Int64).to_list()
    n     = len(sizes)
    algo_flags = [False] * n
    window = min(_ALGO_WINDOW, n)

    for start in range(n - window + 1):
        end       = start + window
        win_sizes = sizes[start:end]
        win_ts    = ts[start:end]

        # size uniformity check
        mean_s = sum(win_sizes) / window
        if mean_s == 0:
            continue
        cv_s = (sum((s - mean_s) ** 2 for s in win_sizes) / window) ** 0.5 / mean_s
        if cv_s > _ALGO_SIZE_CV_MAX:
            continue

        # timing regularity check
        gaps = [win_ts[i+1] - win_ts[i] for i in range(window - 1)]
        if not gaps or min(gaps) <= 0:
            continue
        mean_g = sum(gaps) / len(gaps)
        if mean_g == 0:
            continue
        cv_g = (sum((g - mean_g) ** 2 for g in gaps) / len(gaps)) ** 0.5 / mean_g
        if cv_g > _ALGO_TIME_CV_MAX:
            continue

        for i in range(start, end):
            algo_flags[i] = True

    return df.with_columns(
        pl.Series("is_algo_trade", algo_flags, dtype=pl.Boolean)
    )


# ─────────────────────────────────────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────────────────────────────────────

def filter_trades(trades: pl.DataFrame) -> pl.DataFrame:
    """Remove non-bona-fide prints, flag block trades and TWAP/VWAP algos."""
    if trades.is_empty():
        return trades

    n_before = len(trades)

    # ── condition-code filter ─────────────────────────────────────────────
    if "conditions" in trades.columns:
        exclude_list = list(_EXCLUDE_CONDITIONS)

        def _is_clean(conds) -> bool:
            if conds is None or len(conds) == 0:
                return True
            return not bool(set(conds) & _EXCLUDE_CONDITIONS)

        try:
            has_bad = (
                trades["conditions"]
                .list.eval(pl.element().is_in(exclude_list))
                .list.any()
            )
            keep = has_bad.is_null() | has_bad.not_()
            filtered = trades.filter(keep)
        except Exception:
            mask = trades["conditions"].map_elements(
                _is_clean, return_dtype=pl.Boolean
            )
            filtered = trades.filter(mask)
    else:
        filtered = trades

    # ── size filter ───────────────────────────────────────────────────────
    filtered = filtered.filter(pl.col("size") > 0)

    n_removed = n_before - len(filtered)
    if n_removed > 0:
        pct = n_removed / n_before * 100
        log.debug(
            "filter_trades: removed %d/%d non-bona-fide prints (%.1f%%)",
            n_removed, n_before, pct,
        )

    if filtered.is_empty():
        return filtered.with_columns(
            pl.lit(False).alias("is_block_trade"),
            pl.lit(False).alias("is_algo_trade"),
        )

    # ── block-trade flag (top 5%, floor 50 contracts) ────────────────────
    p95 = filtered["size"].quantile(0.95) or 0.0
    threshold = max(float(p95), 50.0)
    filtered = filtered.with_columns(
        (pl.col("size") >= threshold).alias("is_block_trade")
    )
    n_blocks = int(filtered["is_block_trade"].sum())
    if n_blocks:
        log.debug(
            "filter_trades: flagged %d block trades (size >= %.0f)",
            n_blocks, threshold,
        )

    # ── TWAP/VWAP algo detection ──────────────────────────────────────────
    filtered = flag_algo_trades(filtered)
    n_algo = int(filtered["is_algo_trade"].sum())
    if n_algo:
        log.debug(
            "filter_trades: flagged %d TWAP/VWAP algo prints", n_algo,
        )

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# TICK RULE
# ─────────────────────────────────────────────────────────────────────────────

def _tick_rule_vectorized(prices: list[float]) -> list[int]:
    directions: list[int] = []
    last_nonzero: int = 1

    for i, price in enumerate(prices):
        if i == 0:
            directions.append(last_nonzero)
            continue
        diff = price - prices[i - 1]
        if diff > 0:
            d = 1
            last_nonzero = 1
        elif diff < 0:
            d = -1
            last_nonzero = -1
        else:
            d = last_nonzero
        directions.append(d)

    return directions


# ─────────────────────────────────────────────────────────────────────────────
# LEE-READY CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_trades_lee_ready(
    trades: pl.DataFrame,
    quotes: pl.DataFrame,
) -> pl.DataFrame:
    """Lee-Ready (1991): compare trade price to bid-ask midpoint.
    Falls back to tick rule when no quote is available."""
    if trades.is_empty():
        return trades.with_columns(pl.lit(0, dtype=pl.Int8).alias("direction"))

    trades_s = trades.sort("sip_timestamp")
    prices   = trades_s["price"].to_list()
    tick_dirs = _tick_rule_vectorized(prices)
    trades_s = trades_s.with_columns(
        pl.Series("_tick_dir", tick_dirs, dtype=pl.Int8)
    )

    if quotes.is_empty() or "bid_price" not in quotes.columns:
        log.debug("classify_lee_ready: no quotes — pure tick rule")
        return trades_s.with_columns(
            pl.col("_tick_dir").alias("direction")
        ).drop("_tick_dir")

    quotes_s = (
        quotes.sort("sip_timestamp")
        .with_columns(
            ((pl.col("bid_price") + pl.col("ask_price")) / 2.0).alias("midpoint")
        )
        .select(["sip_timestamp", "bid_price", "ask_price", "midpoint"])
    )

    joined = trades_s.join_asof(
        quotes_s,
        on="sip_timestamp",
        strategy="backward",
    )

    result = joined.with_columns(
        pl.when(pl.col("midpoint").is_null())
            .then(pl.col("_tick_dir"))
        .when(pl.col("price") > pl.col("midpoint"))
            .then(pl.lit(1, dtype=pl.Int8))
        .when(pl.col("price") < pl.col("midpoint"))
            .then(pl.lit(-1, dtype=pl.Int8))
        .otherwise(pl.col("_tick_dir"))
        .cast(pl.Int8)
        .alias("direction")
    )

    n_quote = result.filter(pl.col("midpoint").is_not_null()).height
    n_tick  = result.height - n_quote
    log.debug(
        "classify_lee_ready: quote rule=%d  tick fallback=%d", n_quote, n_tick
    )

    return result.drop(["_tick_dir", "midpoint", "bid_price", "ask_price"])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASSIFY ENTRY POINT  (called by run_backtest.py — no changes needed there)
# ─────────────────────────────────────────────────────────────────────────────

def classify_trades(
    trades: pl.DataFrame,
    quotes: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Filter junk prints then classify trade direction.
    Uses Lee-Ready when CLASSIFICATION_MODE='quote' and quotes provided,
    otherwise tick rule."""
    trades = filter_trades(trades)

    if trades.is_empty():
        return trades

    if CLASSIFICATION_MODE == "quote" and quotes is not None:
        return classify_trades_lee_ready(trades, quotes)
    else:
        prices    = trades.sort("sip_timestamp")["price"].to_list()
        tick_dirs = _tick_rule_vectorized(prices)
        return trades.sort("sip_timestamp").with_columns(
            pl.Series("direction", tick_dirs, dtype=pl.Int8)
        )


def sum_mm_position(classified: pl.DataFrame) -> float:
    """Sum signed volume, excluding block trades and TWAP/VWAP algo prints."""
    if classified.is_empty() or "direction" not in classified.columns:
        return 0.0

    df = classified

    # exclude single large prints
    if "is_block_trade" in df.columns:
        df = df.filter(pl.col("is_block_trade").not_())

    # exclude TWAP/VWAP algo sequences
    if "is_algo_trade" in df.columns:
        df = df.filter(pl.col("is_algo_trade").not_())

    if df.is_empty():
        return 0.0

    return float(
        (df["direction"].cast(pl.Float64) * df["size"]).sum()
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_trades(
    client: PolygonClient,
    option_ticker: str,
    day: date,
    **kwargs,
) -> pl.DataFrame:
    """Fetch all option trades for a given ticker and day, with parquet cache."""
    cache_path = storage.trades_cache_path(option_ticker, day)
    cached     = storage.read_parquet(cache_path)
    if cached is not None:
        return cached

    rows: list[dict] = []
    params = {
        "timestamp": day.isoformat(),
        "order": "asc",
        "sort": "timestamp",
        "limit": 50000,
        **kwargs,
    }
    async for row in client.paginate(f"/v3/trades/{option_ticker}", params=params):
        rows.append(row)

    if not rows:
        df = pl.DataFrame(
            schema={
                "sip_timestamp": pl.Datetime,
                "price":         pl.Float64,
                "size":          pl.Float64,
                "conditions":    pl.List(pl.String),
                "exchange":      pl.Int64,
            }
        )
    else:
        df = pl.from_dicts(rows, infer_schema_length=len(rows))
        if "sip_timestamp" in df.columns:
            df = df.with_columns(
                pl.from_epoch(pl.col("sip_timestamp"), time_unit="ns")
            )

    storage.write_parquet(df, cache_path)
    return df


async def fetch_quotes(
    client: PolygonClient,
    option_ticker: str,
    day: date,
    timestamp_gte=None,
    timestamp_lte=None,
) -> pl.DataFrame:
    """Fetch NBBO quotes for a given ticker and day, with parquet cache."""
    cache_path = storage.quotes_cache_path(option_ticker, day)
    cached     = storage.read_parquet(cache_path)
    if cached is not None:
        return cached

    params: dict = {
        "order": "asc",
        "sort": "timestamp",
        "limit": 50000,
    }
    if timestamp_gte is not None:
        params["timestamp.gte"] = (
            timestamp_gte.isoformat()
            if hasattr(timestamp_gte, "isoformat")
            else str(timestamp_gte)
        )
    if timestamp_lte is not None:
        params["timestamp.lte"] = (
            timestamp_lte.isoformat()
            if hasattr(timestamp_lte, "isoformat")
            else str(timestamp_lte)
        )
    else:
        params["timestamp"] = day.isoformat()

    rows: list[dict] = []
    async for row in client.paginate(
        f"/v3/quotes/{option_ticker}", params=params
    ):
        rows.append(row)

    if not rows:
        df = pl.DataFrame(
            schema={
                "sip_timestamp": pl.Datetime,
                "bid_price":     pl.Float64,
                "ask_price":     pl.Float64,
                "bid_size":      pl.Float64,
                "ask_size":      pl.Float64,
            }
        )
    else:
        df = pl.from_dicts(rows, infer_schema_length=len(rows))
        if "sip_timestamp" in df.columns:
            df = df.with_columns(
                pl.from_epoch(pl.col("sip_timestamp"), time_unit="ns")
            )

    storage.write_parquet(df, cache_path)
    return df


async def fetch_snapshot_quote(
    client: PolygonClient,
    option_ticker: str,
    timestamp: datetime,
) -> pl.DataFrame:
    """Fetch the best bid/ask quote closest to a given snapshot timestamp.
    Used by run_backtest.py to get option mid-price for IV and gamma calculation."""
    if isinstance(timestamp, datetime):
        ts_gte = timestamp - timedelta(seconds=30)
        ts_lte = timestamp + timedelta(seconds=30)
        ts_gte_str = ts_gte.isoformat()
        ts_lte_str = ts_lte.isoformat()
    else:
        ts_gte_str = str(timestamp)
        ts_lte_str = str(timestamp)

    params: dict = {
        "order": "desc",
        "sort": "timestamp",
        "limit": 1,
        "timestamp.gte": ts_gte_str,
        "timestamp.lte": ts_lte_str,
    }

    rows: list[dict] = []
    async for row in client.paginate(
        f"/v3/quotes/{option_ticker}", params=params
    ):
        rows.append(row)
        break

    if not rows:
        log.debug(
            "fetch_snapshot_quote: no quote found for %s at %s",
            option_ticker, timestamp,
        )
        return pl.DataFrame(
            schema={
                "sip_timestamp": pl.Datetime,
                "bid_price":     pl.Float64,
                "ask_price":     pl.Float64,
                "bid_size":      pl.Float64,
                "ask_size":      pl.Float64,
            }
        )

    df = pl.from_dicts(rows, infer_schema_length=1)
    if "sip_timestamp" in df.columns:
        df = df.with_columns(
            pl.from_epoch(pl.col("sip_timestamp"), time_unit="ns")
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CONCURRENT FETCHING
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_CONCURRENCY = 8


async def batch_fetch_trades(
    client: PolygonClient,
    option_tickers: list[str],
    day: date,
    max_concurrent: int = _DEFAULT_CONCURRENCY,
) -> dict[str, pl.DataFrame]:
    """Fetch trades for multiple contracts concurrently."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(ticker: str) -> tuple[str, pl.DataFrame]:
        async with sem:
            try:
                return ticker, await fetch_trades(client, ticker, day)
            except Exception as exc:
                log.warning("batch_fetch_trades: %s on %s: %s", ticker, day, exc)
                return ticker, pl.DataFrame()

    results = await asyncio.gather(*[_one(t) for t in option_tickers])
    return dict(results)


async def batch_fetch_quotes(
    client: PolygonClient,
    option_tickers: list[str],
    day: date,
    trades_map: Optional[dict[str, pl.DataFrame]] = None,
    buffer_seconds: int = 5,
    max_concurrent: int = _DEFAULT_CONCURRENCY,
) -> dict[str, pl.DataFrame]:
    """Fetch quotes for multiple contracts concurrently.
    Narrows the time window to the actual trades timeframe to reduce download size."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(ticker: str) -> tuple[str, pl.DataFrame]:
        async with sem:
            try:
                kwargs: dict = {}
                if trades_map and ticker in trades_map:
                    t_df = trades_map[ticker]
                    if not t_df.is_empty() and "sip_timestamp" in t_df.columns:
                        t_min = t_df["sip_timestamp"].min()
                        t_max = t_df["sip_timestamp"].max()
                        kwargs["timestamp_gte"] = t_min - timedelta(seconds=buffer_seconds)
                        kwargs["timestamp_lte"] = t_max + timedelta(seconds=buffer_seconds)
                return ticker, await fetch_quotes(client, ticker, day, **kwargs)
            except Exception as exc:
                log.warning("batch_fetch_quotes: %s on %s: %s", ticker, day, exc)
                return ticker, pl.DataFrame()

    results = await asyncio.gather(*[_one(t) for t in option_tickers])
    return dict(results)