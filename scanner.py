#!/usr/bin/env python3
"""
scanner.py — a signals bot. No orders, no auto-trading. It scans a basket of
coins, runs the strategy, and tells you where a setup is forming with a defined
entry, stop, and target. You decide whether to take it.

Usage:
    python3 scanner.py                       # scan top-10 once
    python3 scanner.py --basket solana       # SOL-ecosystem majors
    python3 scanner.py --timeframe 1h
    python3 scanner.py --loop 300            # rescan every 300s
    python3 scanner.py --offline             # demo on synthetic data
"""

from __future__ import annotations
import argparse
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bot import Config, add_indicators, fetch_ohlcv, synthetic

BASKETS = {
    "top10": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
              "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "TRX/USDT"],
    "solana": ["SOL/USDT", "JUP/USDT", "JTO/USDT", "WIF/USDT", "BONK/USDT",
               "RAY/USDT", "PYTH/USDT", "ORCA/USDT"],
}


def evaluate(df: pd.DataFrame, cfg: Config) -> dict:
    """Look at the most recent CLOSED bar and describe the setup, if any."""
    df = add_indicators(df, cfg)
    i = len(df) - 1
    row = df.iloc[i]
    price = float(row["close"])
    fast, slow = df["ema_fast"], df["ema_slow"]
    r = float(row["rsi"])
    atr = float(row["atr"])

    trend = "up" if fast.iloc[i] > slow.iloc[i] else "down"
    crossed_up = fast.iloc[i - 1] <= slow.iloc[i - 1] and fast.iloc[i] > slow.iloc[i]
    crossed_dn = fast.iloc[i - 1] >= slow.iloc[i - 1] and fast.iloc[i] < slow.iloc[i]

    sig, entry, stop, tp, note = 0, None, None, None, "no setup"
    if crossed_up and r < cfg.rsi_long_max:
        sig, note = 1, "fresh long cross"
    elif crossed_dn and r > cfg.rsi_short_min:
        sig, note = -1, "fresh short cross"
    elif trend == "up" and r < 45:
        note = "uptrend pullback — watch for long"
    elif trend == "down" and r > 55:
        note = "downtrend bounce — watch for short"

    if sig != 0:
        entry = price
        stop = entry - sig * cfg.atr_stop_mult * atr
        tp = entry + sig * cfg.atr_stop_mult * atr * cfg.rr

    # crude conviction score from how clean the setup is
    dist = abs(fast.iloc[i] - slow.iloc[i]) / price * 100
    conviction = "—"
    if sig != 0:
        conviction = "high" if dist < 0.3 else "med"  # fresh cross = bars close

    return {
        "signal": {1: "LONG", -1: "SHORT", 0: "—"}[sig],
        "price": round(price, 4),
        "trend": trend,
        "rsi": round(r, 1),
        "entry": round(entry, 4) if entry else "",
        "stop": round(stop, 4) if stop else "",
        "target": round(tp, 4) if tp else "",
        "conviction": conviction,
        "note": note,
    }


def scan(basket, cfg: Config, offline: bool):
    rows = []
    for symbol in basket:
        try:
            if offline:
                df = synthetic(seed=hash(symbol) % 1000)
            else:
                c = Config(**{**cfg.__dict__, "symbol": symbol})
                df = fetch_ohlcv(c, limit=cfg.ema_slow + 60)
            r = evaluate(df, cfg)
            r["symbol"] = symbol
            rows.append(r)
        except Exception as e:
            rows.append({"symbol": symbol, "signal": "ERR", "note": str(e)[:40]})
    return rows


def show(rows):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n=== scan @ {ts} ===")
    hdr = ["symbol", "signal", "price", "trend", "rsi", "entry", "stop",
           "target", "conviction", "note"]
    widths = [10, 6, 11, 5, 5, 11, 11, 11, 10, 34]
    print(" ".join(h.ljust(w) for h, w in zip(hdr, widths)))
    print("-" * (sum(widths) + len(widths)))
    # signals first, then watchlist, then nothing
    order = {"LONG": 0, "SHORT": 0, "—": 1, "ERR": 2}
    for r in sorted(rows, key=lambda x: order.get(x.get("signal", "—"), 1)):
        print(" ".join(str(r.get(h, "")).ljust(w) for h, w in zip(hdr, widths)))
    actionable = [r for r in rows if r.get("signal") in ("LONG", "SHORT")]
    print(f"\n{len(actionable)} actionable setup(s). The rest are watch/none.")
    print("Signals are suggestions, not orders. You place the trade, with the stop shown.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basket", choices=list(BASKETS), default="top10")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--loop", type=int, default=0, help="seconds between rescans; 0 = once")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()

    cfg = Config(timeframe=args.timeframe)
    basket = BASKETS[args.basket]

    while True:
        rows = scan(basket, cfg, args.offline)
        show(rows)
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
