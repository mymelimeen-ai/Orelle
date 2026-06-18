#!/usr/bin/env python3
"""
optimize.py — find the best-performing, still-survivable config for edgebot.

"Max profit" is the wrong target on its own: the highest raw return usually
comes with a drawdown that liquidates you before you get there. So this ranks
configs by a risk-ADJUSTED score (return divided by pain), and throws out
anything with a brutal drawdown or too few trades to trust.

Run on your VPS where it can reach the exchange:
    python3 optimize.py                      # real data via ccxt
    python3 optimize.py --offline            # synthetic, just to see it work
    python3 optimize.py --symbols BTC/USDT ETH/USDT SOL/USDT
"""

from __future__ import annotations
import argparse
import itertools
import json

import numpy as np

from bot import Config, backtest, add_indicators, fetch_ohlcv, synthetic


# the grid we search. add/remove values freely — more = slower but wider.
GRID = {
    "timeframe": ["15m", "1h", "4h"],
    "ema_fast":  [13, 21, 34],
    "ema_slow":  [55, 89],
    "rr":        [1.5, 1.8, 2.5],
    "atr_stop_mult": [1.5, 2.0, 3.0],
}

# survivability filters — a config must clear these to even be ranked
MIN_TRADES = 20          # too few trades = result is noise, not edge
MAX_DD_ALLOWED = -25.0   # reject anything that drew down worse than 25%


def score(res: dict) -> float:
    """Risk-adjusted: return per unit of max drawdown, nudged by profit factor.
    High return with a shallow drawdown wins; high return via a near-death
    drawdown does not."""
    if res.get("trades", 0) < MIN_TRADES:
        return -1e9
    if res.get("max_drawdown_pct", -100) < MAX_DD_ALLOWED:
        return -1e9
    ret = res.get("return_pct", 0.0)
    dd = abs(res.get("max_drawdown_pct", -1.0)) or 1.0
    pf = res.get("profit_factor", 0.0)
    pf = min(pf, 5.0) if pf != float("inf") else 5.0
    return (ret / dd) * (0.5 + 0.5 * pf / 5.0)


def run(symbols, offline=False, base=Config()):
    # cache data per (symbol, timeframe) so we fetch once, test many params
    cache = {}
    def data(symbol, tf):
        key = (symbol, tf)
        if key not in cache:
            if offline:
                cache[key] = synthetic()
            else:
                c = Config(**{**base.__dict__, "symbol": symbol, "timeframe": tf})
                cache[key] = fetch_ohlcv(c, limit=1500)
        return cache[key]

    results = []
    combos = list(itertools.product(*GRID.values()))
    keys = list(GRID.keys())
    total = len(symbols) * len(combos)
    n = 0

    for symbol in symbols:
        for combo in combos:
            n += 1
            params = dict(zip(keys, combo))
            if params["ema_fast"] >= params["ema_slow"]:
                continue
            cfg = Config(**{**base.__dict__, "symbol": symbol, **params})
            try:
                df = data(symbol, params["timeframe"])
                res = backtest(df, cfg)
            except Exception as e:
                continue
            res["score"] = round(score(res), 3)
            res["symbol"] = symbol
            res.update(params)
            results.append(res)
            print(f"\r[{n}/{total}] tested {symbol} {params['timeframe']} "
                  f"ema{params['ema_fast']}/{params['ema_slow']}", end="", flush=True)

    print()
    ranked = sorted([r for r in results if r.get("score", -1e9) > -1e8],
                    key=lambda r: r["score"], reverse=True)
    return ranked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    p.add_argument("--offline", action="store_true")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    ranked = run(args.symbols, offline=args.offline)
    if not ranked:
        print("No config cleared the survivability filters. Markets/params too weak.")
        return

    print(f"\nTop {args.top} configs by risk-adjusted score:\n")
    cols = ["score", "symbol", "timeframe", "ema_fast", "ema_slow", "rr",
            "atr_stop_mult", "return_pct", "max_drawdown_pct", "win_rate",
            "profit_factor", "trades"]
    print(" | ".join(c[:9].rjust(9) for c in cols))
    print("-" * 9 * len(cols))
    for r in ranked[:args.top]:
        print(" | ".join(str(r.get(c, "")).rjust(9)[:9] for c in cols))

    best = ranked[0]
    keep = {k: best[k] for k in ("symbol", "timeframe", "ema_fast", "ema_slow",
                                 "rr", "atr_stop_mult")}
    winner = Config(**{**Config().__dict__, **keep})
    winner.save("config.best.json")
    print("\nBest config written to config.best.json")
    print("Next: paper trade it for weeks before going live —")
    print("  python3 bot.py paper --config config.best.json")


if __name__ == "__main__":
    main()
