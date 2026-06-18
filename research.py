#!/usr/bin/env python3
"""
research.py — find a strategy/params with a REAL per-trade edge on your data,
without fooling yourself.

The trap with parameter search is overfitting: try enough combos and something
always looks great on history, then dies live. This harness avoids that with
**walk-forward validation** — it tunes parameters on an older "train" window,
then measures them on the *next, unseen* "test" window, rolls forward, and
repeats. The only number that counts is the aggregated **out-of-sample (OOS)**
result. If a strategy can't show positive OOS expectancy, we don't trade it.

Every strategy uses the same hard-stop risk engine from bot.py, so per-trade
loss is always bounded to your risk cap — profitable or not, you "hit the stop."

Usage (run in a session that has the data, i.e. locally):
    python3 research.py --csv sol_usdt_15m.csv
    python3 research.py --csv sol_usdt_15m.csv --strategies ema_cross_adx rsi_reversion
    python3 research.py --offline                      # synthetic smoke test
"""

from __future__ import annotations
import argparse
import json
from dataclasses import replace
from itertools import product

import numpy as np

from bot import (Config, add_indicators, _simulate, load_csv, synthetic,
                 _tf_seconds, _STRATEGIES)


# Parameter grids per strategy. Kept modest so a full run finishes in minutes.
GRIDS = {
    "ema_cross":         {"ema_fast": [13, 21], "ema_slow": [55, 89],
                          "atr_stop_mult": [1.5, 2.5], "rr": [1.5, 2.5]},
    "ema_cross_adx":     {"ema_slow": [55, 89], "adx_min": [18, 25],
                          "rr": [1.5, 2.5], "atr_stop_mult": [2.0]},
    "rsi_reversion":     {"rsi_len": [7, 14], "rsi_os": [20, 30], "rsi_ob": [70, 80],
                          "rr": [1.0, 1.5], "atr_stop_mult": [2.0]},
    "bb_reversion":      {"bb_len": [20, 34], "bb_std": [2.0, 2.5],
                          "rr": [1.0, 1.5], "atr_stop_mult": [2.0]},
    "donchian_breakout": {"donchian_len": [20, 55], "rr": [1.5, 2.5],
                          "atr_stop_mult": [2.0, 3.0]},
}


def _combos(grid):
    return [dict(zip(grid, vals)) for vals in product(*grid.values())]


def _aggregate(trades, test_days):
    if not trades:
        return None
    r = np.array([t["r"] for t in trades])
    pnl = np.array([t["pnl"] for t in trades])
    gw = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    mcl = cur = 0
    for p in pnl:
        cur = cur + 1 if p < 0 else 0
        mcl = max(mcl, cur)
    return {
        "trades": len(trades),
        "expectancy_R": round(float(r.mean()), 3),
        "win_rate": round(float((pnl > 0).mean()), 3),
        "profit_factor": round(float(gw / gl), 2) if gl > 0 else float("inf"),
        "net_usd": round(float(pnl.sum()), 2),
        "total_R": round(float(r.sum()), 2),
        "trades_per_day": round(len(trades) / test_days, 2) if test_days else 0.0,
        "max_consec_losses": int(mcl),
    }


def _train_score(trades, min_trades):
    """Pick params by in-train expectancy, but only if there are enough trades
    to mean anything."""
    if len(trades) < min_trades:
        return -1e18
    return float(np.mean([t["r"] for t in trades]))


def walk_forward(df, base, strategy, bpd, train_days, test_days, min_train_trades):
    combos = _combos(GRIDS[strategy])
    tb, sb = int(train_days * bpd), int(test_days * bpd)
    n = len(df)
    folds, s = [], 0
    while s + tb + sb <= n:
        folds.append((s, s + tb, s + tb + sb))
        s += sb
    if not folds:
        return None, None, 0

    oos_trades = []
    for a, b, c in folds:
        best, best_score = None, -1e18
        for combo in combos:
            cfg = replace(base, strategy=strategy, **combo)
            tr, _ = _simulate(add_indicators(df.iloc[:b], cfg).iloc[a:b], cfg, build_curve=False)
            sc = _train_score(tr, min_train_trades)
            if sc > best_score:
                best_score, best = sc, combo
        cfg = replace(base, strategy=strategy, **best)
        tr, _ = _simulate(add_indicators(df.iloc[:c], cfg).iloc[b:c], cfg, build_curve=False)
        oos_trades += tr
    return _aggregate(oos_trades, len(folds) * test_days), len(folds), combos


def in_sample_best(df, base, strategy, bpd, min_trades):
    """Best combo if you (naively) fit the whole dataset — shown only to expose
    the overfitting gap vs. the out-of-sample number."""
    best_combo, best_r, best_metrics = None, -1e18, None
    for combo in _combos(GRIDS[strategy]):
        cfg = replace(base, strategy=strategy, **combo)
        tr, _ = _simulate(add_indicators(df, cfg), cfg, build_curve=False)
        if len(tr) < min_trades:
            continue
        r = float(np.mean([t["r"] for t in tr]))
        if r > best_r:
            best_r, best_combo = r, combo
            best_metrics = _aggregate(tr, len(df) / bpd)
    return best_metrics, best_combo


def main():
    p = argparse.ArgumentParser(description="walk-forward strategy search")
    p.add_argument("--csv", default=None, help="OHLCV CSV (real data)")
    p.add_argument("--config", default=None, help="base config.json (risk settings)")
    p.add_argument("--offline", action="store_true", help="synthetic smoke test")
    p.add_argument("--strategies", nargs="+", default=list(GRIDS),
                   choices=list(_STRATEGIES))
    p.add_argument("--train-days", type=float, default=90)
    p.add_argument("--test-days", type=float, default=30)
    p.add_argument("--min-train-trades", type=int, default=8)
    args = p.parse_args()

    base = Config.load(args.config) if args.config else Config()
    if args.offline:
        df = synthetic(n=20000)
    elif args.csv:
        df = load_csv(args.csv)
    else:
        raise SystemExit("Pass --csv <file> (real data) or --offline (smoke test).")

    bpd = 86400 / _tf_seconds(base.timeframe)
    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    print(f"data: {len(df)} bars  {df.index[0]} -> {df.index[-1]}  (~{span_days:.0f} days)")
    print(f"walk-forward: train {args.train_days}d / test {args.test_days}d, "
          f"rolling. OOS = out-of-sample (the number that matters).\n")

    rows = []
    for strat in args.strategies:
        oos, nfolds, _ = walk_forward(df, base, strat, bpd,
                                      args.train_days, args.test_days, args.min_train_trades)
        ins, combo = in_sample_best(df, base, strat, bpd, args.min_train_trades)
        if not oos:
            print(f"  {strat:<18} not enough data for walk-forward")
            continue
        rows.append((strat, oos, ins, combo, nfolds))

    if not rows:
        print("No strategy could be evaluated on this data.")
        return

    rows.sort(key=lambda x: x[1]["expectancy_R"], reverse=True)
    hdr = f"{'strategy':<18}{'OOS expR':>10}{'OOS win%':>9}{'OOS PF':>8}" \
          f"{'OOS trades':>11}{'trd/day':>9}{'OOS net$':>10}{'IS expR':>9}"
    print(hdr)
    print("-" * len(hdr))
    for strat, oos, ins, combo, nfolds in rows:
        is_r = ins["expectancy_R"] if ins else float("nan")
        print(f"{strat:<18}{oos['expectancy_R']:>10}{oos['win_rate']*100:>8.1f}%"
              f"{oos['profit_factor']:>8}{oos['trades']:>11}{oos['trades_per_day']:>9}"
              f"{oos['net_usd']:>10}{is_r:>9}")

    print("\nLegend: OOS = unseen test windows (honest). IS = fit to all data "
          "(optimistic; the gap to OOS is the overfitting tax).")

    # verdict
    edge = [(s, o, i, c) for (s, o, i, c, _) in rows
            if o["expectancy_R"] > 0 and o["profit_factor"] > 1.0 and o["trades"] >= 30]
    print("\n" + "=" * 70)
    if not edge:
        print("VERDICT: no strategy showed a positive, trustworthy out-of-sample edge.")
        print("That is a real result — on this data, none of these is worth trading.")
        return
    best_strat, best_oos, _, best_combo = edge[0]
    print(f"VERDICT: '{best_strat}' is the only/best candidate with positive OOS edge:")
    print(f"  expectancy {best_oos['expectancy_R']}R/trade, PF {best_oos['profit_factor']}, "
          f"{best_oos['trades_per_day']} trades/day, {best_oos['trades']} trades OOS.")
    winner = replace(base, strategy=best_strat, **best_combo)
    winner.save("config.research.json")
    print("  Wrote config.research.json. Next: paper-trade it for weeks before any live use.")
    print("  (OOS edge is necessary, not sufficient — confirm it holds forward in paper.)")


if __name__ == "__main__":
    main()
