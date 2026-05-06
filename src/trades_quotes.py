"""Option trades & quotes: fetch per day, classify initiative direction.

sum_mm_position()
    Extended to exclude block trades and flagged algo prints before summing.
"""
from __future__ import annotations

import logging
from datetime import date

import polars as pl

from config import CLASSIFICATION_MODE, TICKER
from src import storage
from src.client import PolygonClient

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS  (unchanged from Jerry's baseline — must match flatfiles output)
# ─────────────────────────────────────────────────────────────────────────────

_TRADES_SCHEMA = {
    "sip_timestamp": pl.Int64,
    "price":         pl.Float64,
    "size":          pl.Float64,
    "exchange":      pl.Int64,
    "conditions":    pl.List(pl.Int64),   # integer codes per API spec
}

_QUOTES_SCHEMA = {
    "sip_timestamp":   pl.Int64,
    "bid_price":       pl.Float64,
    "ask_price":       pl.Float64,
    "bid_size":        pl.Float64,
    "ask_size":        pl.Float64,
    "sequence_number": pl.Int64,
}

_CLASSIFIED_SCHEMA = {
    "sip_timestamp": pl.Int64,
    "price":         pl.Float64,
    "size":          pl.Float64,
    "mm_delta":      pl.Float64,
}


def _select_or_null(rows: list[dict], field: str, dtype: pl.DataType) -> pl.Series:
    """Extract a column from list-of-dicts, filling missing keys with null."""
    return pl.Series(field, [r.get(field) for r in rows], dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# CONDITION CODES TO EXCLUDE  (integer — matches Polygon/Massive API response)
# ─────────────────────────────────────────────────────────────────────────────

_EXCLUDE_CONDITIONS: frozenset[int] = frozenset({
    2,   # Average Price / Bunched
    7,   # Bunched Sold
    15,  # Opening / Reopening
    16,  # Stopped Stock
    21,  # Cross Trade
    32,  # Inter-Market Sweep
    33,  # Derivatively Priced
    41,  # Contingent Trade
    52,  # Multi-Leg Auto-Electronic (MLEG)
    53,  # Multi-Leg Auction (MLEGA)
    54,  # Multi-Leg Cross
    55,  # Multi-Leg Floor Trade
})

# ─────────────────────────────────────────────────────────────────────────────
# TWAP / VWAP ALGO DETECTION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

_ALGO_WINDOW      = 6     # sliding window (trades). 10 is too wide for thin
                          # SPY 0DTE contracts with only ~15-50 trades/day.
_ALGO_SIZE_CV_MAX = 0.20  # max size coefficient of variation (uniform = TWAP)
_ALGO_TIME_CV_MAX = 0.35  # max gap coefficient of variation (regular = TWAP)
_ALGO_MIN_TRADES  = 5     # minimum trades before algo detection runs


def flag_algo_trades(trades: pl.DataFrame) -> pl.DataFrame:
    """Detect TWAP-style algorithmically split orders.

    Slides a window of _ALGO_WINDOW trades across the time-sorted sequence.
    Flags the entire window when:
        1. Trade sizes are suspiciously uniform: CV(sizes) < _ALGO_SIZE_CV_MAX
        2. Inter-trade gaps are suspiciously regular: CV(gaps) < _ALGO_TIME_CV_MAX

    This catches TWAP execution (split into equal-sized tranches at regular
    intervals). VWAP detection is not implemented here — see module docstring.

    Args:
        trades: DataFrame with sip_timestamp (Int64) and size (Float64) columns.

    Returns:
        Same DataFrame with boolean is_algo_trade column added.
    """
    if trades.is_empty() or len(trades) < _ALGO_MIN_TRADES:
        return trades.with_columns(pl.lit(False).alias("is_algo_trade"))

    df       = trades.sort("sip_timestamp")
    sizes    = df["size"].to_list()
    ts_vals  = df["sip_timestamp"].to_list()
    n        = len(sizes)
    flags    = [False] * n
    window   = min(_ALGO_WINDOW, n)

    for start in range(n - window + 1):
        end      = start + window
        w_sizes  = sizes[start:end]
        w_ts     = ts_vals[start:end]

        # ── size uniformity ──────────────────────────────────────────────────
        mean_s = sum(w_sizes) / window
        if mean_s <= 0:
            continue
        var_s = sum((s - mean_s) ** 2 for s in w_sizes) / window
        cv_s  = var_s ** 0.5 / mean_s
        if cv_s > _ALGO_SIZE_CV_MAX:
            continue

        # ── timing regularity ────────────────────────────────────────────────
        gaps   = [w_ts[i + 1] - w_ts[i] for i in range(window - 1)]
        if not gaps or min(gaps) <= 0:
            continue
        mean_g = sum(gaps) / len(gaps)
        if mean_g <= 0:
            continue
        var_g = sum((g - mean_g) ** 2 for g in gaps) / len(gaps)
        cv_g  = var_g ** 0.5 / mean_g
        if cv_g > _ALGO_TIME_CV_MAX:
            continue

        for i in range(start, end):
            flags[i] = True

    return df.with_columns(
        pl.Series("is_algo_trade", flags, dtype=pl.Boolean)
    )


# ─────────────────────────────────────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────────────────────────────────────

def filter_trades(trades: pl.DataFrame) -> pl.DataFrame:
    """Strip non-bona-fide prints; flag block trades and TWAP algo sequences.

    Called automatically at the start of classify_trades() — no changes
    needed anywhere in run_backtest.py.

    Filters applied in order:
        1. Condition-code filter: drop any trade whose conditions list
           contains any code in _EXCLUDE_CONDITIONS. Trades with null or
           empty conditions lists are kept (normal prints).
        2. Size filter: drop size <= 0.
        3. Block trade flag: is_block_trade = True when size >= max(p95, 50).
        4. Algo flag: is_algo_trade = True for TWAP-pattern windows.

    The is_block_trade and is_algo_trade columns are consumed by
    sum_mm_position() to exclude those prints from the MM inventory sum.

    Args:
        trades: Raw DataFrame from fetch_trades(). Schema: _TRADES_SCHEMA.

    Returns:
        Filtered DataFrame with two extra bool columns: is_block_trade,
        is_algo_trade. Schema otherwise identical to input.
    """
    if trades.is_empty():
        return trades.with_columns(
            pl.lit(False).alias("is_block_trade"),
            pl.lit(False).alias("is_algo_trade"),
        )

    n_before = len(trades)

    # ── 1. condition-code filter ─────────────────────────────────────────────
    if "conditions" in trades.columns:
        cond_dtype = trades["conditions"].dtype

        # If the column is all-null (dtype Null) or all rows are null lists,
        # every row is a normal print — keep all, skip filter.
        if cond_dtype == pl.Null:
            pass  # nothing to filter
        else:
            exclude_series = pl.Series(
                "excl", sorted(_EXCLUDE_CONDITIONS), dtype=pl.Int64
            )
            try:
                # Vectorised: does this row's conditions list overlap with
                # the exclude set?  Null list rows → has_bad is null → kept.
                has_bad = (
                    trades["conditions"]
                    .list.eval(pl.element().is_in(exclude_series))
                    .list.any()
                )
                trades = trades.filter(has_bad.is_null() | has_bad.not_())
            except Exception:
                # Last-resort fallback (unusual schema / Polars version gap)
                def _clean(conds) -> bool:
                    if conds is None or len(conds) == 0:
                        return True
                    return not bool(set(conds) & _EXCLUDE_CONDITIONS)

                mask = trades["conditions"].map_elements(
                    _clean, return_dtype=pl.Boolean
                )
                # map_elements on a null-heavy column can return nulls —
                # treat null as clean (keep row).
                trades = trades.filter(mask.fill_null(True))

    # ── 2. size filter ───────────────────────────────────────────────────────
    trades = trades.filter(pl.col("size") > 0)

    n_removed = n_before - len(trades)
    if n_removed > 0:
        logger.debug(
            "filter_trades: removed %d / %d non-bona-fide prints (%.1f%%)",
            n_removed, n_before, n_removed / n_before * 100,
        )

    if trades.is_empty():
        return trades.with_columns(
            pl.lit(False).alias("is_block_trade"),
            pl.lit(False).alias("is_algo_trade"),
        )

    # ── 3. block trade flag ──────────────────────────────────────────────────
    p95       = float(trades["size"].quantile(0.95) or 0.0)
    threshold = max(p95, 50.0)
    trades    = trades.with_columns(
        (pl.col("size") >= threshold).alias("is_block_trade")
    )
    n_blocks = int(trades["is_block_trade"].sum())
    if n_blocks:
        logger.debug(
            "filter_trades: flagged %d block trades (size >= %.0f)",
            n_blocks, threshold,
        )

    # ── 4. TWAP algo detection ───────────────────────────────────────────────
    trades   = flag_algo_trades(trades)
    n_algo   = int(trades["is_algo_trade"].sum())
    if n_algo:
        logger.debug(
            "filter_trades: flagged %d TWAP algo prints (window=%d)",
            n_algo, _ALGO_WINDOW,
        )

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# FETCH  (identical to Jerry's baseline — schema and cache logic unchanged)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_trades(
    client: PolygonClient, option_ticker: str, day: date
) -> pl.DataFrame:
    """All option trades on `day`. Cached per (contract, day).

    If the flat-file marker for this day is set, the cache is authoritative:
    missing parquet ⇒ this contract had no trades that day, return empty.
    REST is only used if flat-file preprocessing hasn't covered this day.
    """
    cache_path = storage.trades_cache_path(option_ticker, day)
    cached     = storage.read_parquet(cache_path)
    if cached is not None:
        return cached

    if storage.flatfile_marker_exists("trades", day, TICKER):
        return pl.DataFrame(schema=_TRADES_SCHEMA)

    rows: list[dict] = []
    async for row in client.paginate(
        f"/v3/trades/{option_ticker}",
        params={"timestamp": day.isoformat(), "limit": 50000, "order": "asc"},
    ):
        rows.append(row)

    if not rows:
        df = pl.DataFrame(schema=_TRADES_SCHEMA)
    else:
        df = pl.DataFrame(
            {
                "sip_timestamp": _select_or_null(rows, "sip_timestamp", pl.Int64),
                "price":         _select_or_null(rows, "price",         pl.Float64),
                "size":          _select_or_null(rows, "size",          pl.Float64),
                "exchange":      _select_or_null(rows, "exchange",      pl.Int64),
                "conditions":    _select_or_null(rows, "conditions",    pl.List(pl.Int64)),
            }
        ).sort("sip_timestamp")

    storage.write_parquet(df, cache_path)
    return df


async def fetch_snapshot_quote(
    client: PolygonClient,
    option_ticker: str,
    snapshot_day: date,
    snapshot_ns: int,
    n_probe: int = 50,
) -> float | None:
    """Get bid-ask midpoint at or before `snapshot_ns`.

    Signature matches run_backtest.py exactly:
        fetch_snapshot_quote(client, option_ticker, snapshot_day, snapshot_ns)

    Returns mid = (bid + ask) / 2 as a float, or None if no valid quote.
    Makes a single targeted REST call (no full-day pagination).
    Caches a one-row parquet per (contract, day) so reruns skip the network.
    """
    cache_path = storage.snapshot_quote_cache_path(option_ticker, snapshot_day)
    cached     = storage.read_parquet(cache_path)
    if cached is not None:
        if cached.is_empty():
            return None
        if "mid" in cached.columns:
            return float(cached["mid"][0])
        # Legacy cache had bid/ask columns — compute mid on the fly
        if "bid" in cached.columns and "ask" in cached.columns:
            return (float(cached["bid"][0]) + float(cached["ask"][0])) / 2.0

    payload = await client.get(
        f"/v3/quotes/{option_ticker}",
        params={
            "timestamp.lte": str(snapshot_ns),
            "order":         "desc",
            "sort":          "timestamp",
            "limit":         n_probe,
        },
    )
    rows = payload.get("results") or []
    mid: float | None = None
    for row in rows:
        bid = row.get("bid_price")
        ask = row.get("ask_price")
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            mid = (float(bid) + float(ask)) / 2.0
            break

    # Cache result
    if mid is None:
        storage.write_parquet(
            pl.DataFrame(schema={"mid": pl.Float64}),
            cache_path,
        )
    else:
        storage.write_parquet(
            pl.DataFrame({"mid": [mid]}),
            cache_path,
        )
    return mid


async def fetch_quotes(
    client: PolygonClient, option_ticker: str, day: date
) -> pl.DataFrame:
    """All option quotes on `day`. Cached per (contract, day).

    Used only in quote-rule mode. Under tick mode, only fetch_snapshot_quote
    is called (single targeted REST call at snapshot time).
    """
    cache_path = storage.quotes_cache_path(option_ticker, day)
    cached     = storage.read_parquet(cache_path)
    if cached is not None:
        return cached

    if storage.flatfile_marker_exists("quotes", day, TICKER):
        return pl.DataFrame(schema=_QUOTES_SCHEMA)

    rows: list[dict] = []
    async for row in client.paginate(
        f"/v3/quotes/{option_ticker}",
        params={"timestamp": day.isoformat(), "limit": 50000, "order": "asc"},
    ):
        rows.append(row)

    if not rows:
        df = pl.DataFrame(schema=_QUOTES_SCHEMA)
    else:
        df = pl.DataFrame(
            {
                "sip_timestamp":   _select_or_null(rows, "sip_timestamp",   pl.Int64),
                "bid_price":       _select_or_null(rows, "bid_price",       pl.Float64),
                "ask_price":       _select_or_null(rows, "ask_price",       pl.Float64),
                "bid_size":        _select_or_null(rows, "bid_size",        pl.Float64),
                "ask_size":        _select_or_null(rows, "ask_size",        pl.Float64),
                "sequence_number": _select_or_null(rows, "sequence_number", pl.Int64),
            }
        ).sort("sip_timestamp")

    storage.write_parquet(df, cache_path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_trades_quote_rule(
    trades: pl.DataFrame, quotes: pl.DataFrame
) -> pl.DataFrame:
    """Lee-Ready midpoint classification. Requires lifetime quotes.

    Kept for reference / switching back. Not used when CLASSIFICATION_MODE
    is "tick".
    """
    if trades.is_empty() or quotes.is_empty():
        return pl.DataFrame(schema=_CLASSIFIED_SCHEMA)

    valid_quotes = (
        quotes.filter((pl.col("bid_price") > 0) & (pl.col("ask_price") > 0))
        .with_columns(
            ((pl.col("bid_price") + pl.col("ask_price")) / 2.0).alias("mid")
        )
        .select(["sip_timestamp", "mid"])
        .sort("sip_timestamp")
    )
    if valid_quotes.is_empty():
        return pl.DataFrame(schema=_CLASSIFIED_SCHEMA)

    matched = trades.sort("sip_timestamp").join_asof(
        valid_quotes, on="sip_timestamp", strategy="backward",
    )

    classified = matched.with_columns(
        pl.when(pl.col("mid").is_null())
        .then(0.0)
        .when(pl.col("price") > pl.col("mid"))
        .then(-pl.col("size"))        # buy-initiated → MM sold
        .when(pl.col("price") < pl.col("mid"))
        .then(pl.col("size"))         # sell-initiated → MM bought
        .otherwise(0.0)               # at-mid → drop
        .alias("mm_delta")
    )
    return classified.select(["sip_timestamp", "price", "size", "mm_delta"])


def classify_trades_tick_rule(trades: pl.DataFrame) -> pl.DataFrame:
    """Tick rule — classify using only prior-trade prices. No quotes needed.

    For each trade (after sorting by sip_timestamp):
      * price  > prev      → uptick       → buy-initiated  → mm_delta = -size
      * price  < prev      → downtick     → sell-initiated → mm_delta = +size
      * price == prev      → zero-tick    → carry forward last non-zero direction
      * very first trade   → unclassified → mm_delta = 0 (dropped)
    """
    if trades.is_empty():
        return pl.DataFrame(schema=_CLASSIFIED_SCHEMA)

    t = trades.sort("sip_timestamp").with_columns(
        pl.col("price").diff().alias("_diff")
    )
    t = t.with_columns(
        pl.when(pl.col("_diff") == 0)
        .then(None)
        .otherwise(pl.col("_diff"))
        .forward_fill()
        .alias("_signed_move")
    )
    t = t.with_columns(
        pl.when(pl.col("_signed_move").is_null())
        .then(0.0)
        .when(pl.col("_signed_move") > 0)
        .then(-pl.col("size"))   # uptick → MM sold
        .when(pl.col("_signed_move") < 0)
        .then(pl.col("size"))    # downtick → MM bought
        .otherwise(0.0)
        .alias("mm_delta")
    )
    return t.select(["sip_timestamp", "price", "size", "mm_delta"])


def classify_trades(
    trades: pl.DataFrame, quotes: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Filter junk prints then classify trade direction.

    This is the function called by run_backtest.py — no changes needed there.
    filter_trades() runs automatically before any classification.

    Dispatches to:
        - quote rule (Lee-Ready) when CLASSIFICATION_MODE="quote" + quotes given
        - tick rule otherwise (default)
    """
    # ── filter first — always ────────────────────────────────────────────────
    trades = filter_trades(trades)

    if trades.is_empty():
        return pl.DataFrame(schema=_CLASSIFIED_SCHEMA)

    if CLASSIFICATION_MODE == "tick":
        return classify_trades_tick_rule(trades)
    if CLASSIFICATION_MODE == "quote":
        if quotes is None:
            raise ValueError("quote-rule classification requires quotes")
        return classify_trades_quote_rule(trades, quotes)
    raise ValueError(f"Unknown CLASSIFICATION_MODE: {CLASSIFICATION_MODE!r}")


# ─────────────────────────────────────────────────────────────────────────────
# MM POSITION AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def sum_mm_position(classified: pl.DataFrame) -> float:
    """Total MM net position from a classified trades frame.

    Excludes block trades and TWAP algo prints flagged by filter_trades().
    If those columns are absent (e.g. classify_trades_quote_rule path which
    doesn't carry them through), falls back to summing everything.
    """
    if classified.is_empty():
        return 0.0

    df = classified

    if "is_block_trade" in df.columns:
        df = df.filter(pl.col("is_block_trade").not_())

    if "is_algo_trade" in df.columns:
        df = df.filter(pl.col("is_algo_trade").not_())

    if df.is_empty():
        return 0.0

    total = df["mm_delta"].sum()
    return float(total) if total is not None else 0.0