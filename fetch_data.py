#!/usr/bin/env python3
"""
fetch_data.py — download REAL OHLCV history into a CSV that bot.py can backtest.

Run this on a machine that can reach the exchange (your laptop/VPS) — NOT inside
a locked-down sandbox. It paginates backwards so you can pull months/years of
15m candles, not just the last ~1500 bars.

    pip install ccxt pandas
    python3 fetch_data.py --symbol SOL/USDT --timeframe 15m --days 365
    # -> writes sol_usdt_15m.csv  (columns: time,open,high,low,close,volume)

Then, anywhere (including the sandbox):
    python3 bot.py backtest --config config.json --csv sol_usdt_15m.csv
"""

from __future__ import annotations
import argparse
import time

import pandas as pd

try:
    import ccxt
except ImportError:
    raise SystemExit("pip install ccxt pandas")


def fetch(exchange: str, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    ms_per_bar = ex.parse_timeframe(timeframe) * 1000
    since = ex.milliseconds() - days * 86_400_000
    all_rows, seen = [], set()

    while since < ex.milliseconds():
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1500)
        if not batch:
            break
        # drop dupes the exchange may resend at the boundary
        batch = [r for r in batch if r[0] not in seen]
        if not batch:
            since += ms_per_bar * 1500
            continue
        all_rows += batch
        seen.update(r[0] for r in batch)
        since = batch[-1][0] + ms_per_bar
        print(f"\r  fetched {len(all_rows)} bars ... "
              f"{pd.to_datetime(batch[-1][0], unit='ms')}", end="", flush=True)
        time.sleep(ex.rateLimit / 1000)
    print()

    df = pd.DataFrame(all_rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exchange", default="binanceusdm", help="ccxt id (e.g. binance, binanceusdm, bybit, okx)")
    p.add_argument("--symbol", default="SOL/USDT")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="fetch a whole basket at once (overrides --symbol)")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--days", type=int, default=365, help="how far back to pull")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    symbols = args.symbols or [args.symbol]
    for sym in symbols:
        df = fetch(args.exchange, sym, args.timeframe, args.days)
        out = (args.out if (args.out and len(symbols) == 1)
               else f"{sym.replace('/', '_').lower()}_{args.timeframe}.csv")
        df.to_csv(out, index=False)
        span = f"{df['time'].iloc[0]} -> {df['time'].iloc[-1]}" if len(df) else "no data"
        print(f"wrote {out}: {len(df)} bars  [{span}]")


if __name__ == "__main__":
    main()
