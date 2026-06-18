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

from bot import (Config, prepare_indicators, _simulate, load_csv, synthetic,
                 _STRATEGIES)


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
    # multi-timeframe: 1m entries (RSI pullback) filtered by the 15m trend.
    # Feed a 1m CSV; the 15m bias is derived internally.
    "mtf":               {"rsi_os": [25, 30], "rsi_ob": [70, 75],
                          "rr": [1.5, 2.5], "adx_min": [20]},
}


def _combos(grid):
    return [dict(zip(grid, vals)) for vals in product(*grid.values())]


def detect_bpd(df):
    """Bars per day inferred from the data itself, so any timeframe CSV works.
    Uses Timedelta.total_seconds() to stay agnostic to pandas' datetime unit."""
    if len(df) < 3:
        return 96.0
    sec = df.index.to_series().diff().dropna().median().total_seconds()
    return 86400.0 / sec if sec and sec > 0 else 96.0


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
            tr, _ = _simulate(prepare_indicators(df.iloc[:b], cfg).iloc[a:b], cfg, build_curve=False)
            sc = _train_score(tr, min_train_trades)
            if sc > best_score:
                best_score, best = sc, combo
        cfg = replace(base, strategy=strategy, **best)
        tr, _ = _simulate(prepare_indicators(df.iloc[:c], cfg).iloc[b:c], cfg, build_curve=False)
        oos_trades += tr
    return _aggregate(oos_trades, len(folds) * test_days), len(folds), combos


def in_sample_best(df, base, strategy, bpd, min_trades):
    """Best combo if you (naively) fit the whole dataset — shown only to expose
    the overfitting gap vs. the out-of-sample number."""
    best_combo, best_r, best_metrics = None, -1e18, None
    for combo in _combos(GRIDS[strategy]):
        cfg = replace(base, strategy=strategy, **combo)
        tr, _ = _simulate(prepare_indicators(df, cfg), cfg, build_curve=False)
        if len(tr) < min_trades:
            continue
        r = float(np.mean([t["r"] for t in tr]))
        if r > best_r:
            best_r, best_combo = r, combo
            best_metrics = _aggregate(tr, len(df) / bpd)
    return best_metrics, best_combo


def evaluate(df, base, strategies, args):
    """For one coin/file: walk-forward each strategy on the development data, and
    (if a holdout is set) score the per-strategy chosen params once on a final,
    never-touched holdout. Returns (bpd, holdout_days_used, rows)."""
    bpd = detect_bpd(df)
    n = len(df)
    hb = int(args.holdout_days * bpd) if args.holdout_days > 0 else 0
    if hb and hb >= n * 0.4:                  # keep enough development data
        hb = int(n * 0.2)
    dev_end = n - hb if hb else n
    dev = df.iloc[:dev_end]

    rows = []
    for strat in strategies:
        oos, _, _ = walk_forward(dev, base, strat, bpd,
                                 args.train_days, args.test_days, args.min_train_trades)
        ins, combo = in_sample_best(dev, base, strat, bpd, args.min_train_trades)
        hold = None
        if hb and combo is not None:
            cfg = replace(base, strategy=strat, **combo)
            tr, _ = _simulate(prepare_indicators(df, cfg).iloc[dev_end:], cfg, build_curve=False)
            hold = _aggregate(tr, hb / bpd)
        if oos:
            rows.append((strat, oos, ins, hold, combo))
    return bpd, (hb / bpd if hb else 0), rows


def report(label, bpd, holdout_days, rows, base):
    print(f"\n{'='*78}\n{label}   (~{bpd:.0f} bars/day)")
    if not rows:
        print("  no strategy could be evaluated (not enough data).")
        return None
    use_hold = holdout_days > 0 and any(r[3] for r in rows)
    key = (lambda r: (r[3] or {}).get("expectancy_R", -9) if use_hold
           else r[1]["expectancy_R"])
    rows = sorted(rows, key=key, reverse=True)
    hdr = f"{'strategy':<18}{'OOS expR':>10}{'OOS PF':>8}{'OOS trd':>9}"
    if use_hold:
        hdr += f"{'HOLD expR':>11}{'HOLD PF':>9}{'HOLD trd':>10}"
    hdr += f"{'IS expR':>9}"
    print(hdr); print("-" * len(hdr))
    for strat, oos, ins, hold, combo in rows:
        line = f"{strat:<18}{oos['expectancy_R']:>10}{oos['profit_factor']:>8}{oos['trades']:>9}"
        if use_hold:
            h = hold or {}
            line += (f"{h.get('expectancy_R', float('nan')):>11}"
                     f"{h.get('profit_factor', float('nan')):>9}{h.get('trades', 0):>10}")
        line += f"{(ins or {}).get('expectancy_R', float('nan')):>9}"
        print(line)

    # verdict — require positive OOS AND (if available) positive holdout
    def good(oos, hold):
        if oos["expectancy_R"] <= 0 or oos["profit_factor"] <= 1.0 or oos["trades"] < 30:
            return False
        if use_hold:
            return bool(hold) and hold["expectancy_R"] > 0 and hold["trades"] >= 10
        return True
    edge = [(s, o, h, c) for (s, o, i, h, c) in rows if good(o, h)]
    if not edge:
        print("VERDICT: no strategy survived"
              + (" out-of-sample AND the locked holdout." if use_hold else " out-of-sample."))
        return None
    s, o, h, c = edge[0]
    final = (h or o)
    print(f"VERDICT: '{s}' survived — expR {final['expectancy_R']}, PF {final['profit_factor']}, "
          f"{final['trades']} trades"
          + (" on the locked holdout." if use_hold else " out-of-sample."))
    return replace(base, strategy=s, **c)


def main():
    p = argparse.ArgumentParser(description="walk-forward strategy search (multi-coin)")
    p.add_argument("--csv", nargs="+", default=None, help="one or more OHLCV CSVs (real data)")
    p.add_argument("--config", default=None, help="base config.json (risk settings)")
    p.add_argument("--offline", action="store_true", help="synthetic smoke test")
    p.add_argument("--strategies", nargs="+", default=list(GRIDS),
                   choices=list(_STRATEGIES))
    p.add_argument("--train-days", type=float, default=90)
    p.add_argument("--test-days", type=float, default=30)
    p.add_argument("--holdout-days", type=float, default=60,
                   help="final locked holdout in days (0 to disable); guards against "
                        "data-dredging across coins/strategies")
    p.add_argument("--min-train-trades", type=int, default=8)
    args = p.parse_args()

    base = Config.load(args.config) if args.config else Config()
    if args.offline:
        sources = [("synthetic", synthetic(n=20000))]
    elif args.csv:
        sources = [(f, load_csv(f)) for f in args.csv]
    else:
        raise SystemExit("Pass --csv <file...> (real data) or --offline (smoke test).")

    print(f"walk-forward train {args.train_days}d / test {args.test_days}d, "
          f"locked holdout {args.holdout_days}d. OOS + HOLD are the honest numbers.")
    winners = []
    for label, df in sources:
        span = (df.index[-1] - df.index[0]).total_seconds() / 86400
        bpd, hold_days, rows = evaluate(df, base, args.strategies, args)
        w = report(f"{label}  [{len(df)} bars, ~{span:.0f}d]", bpd, hold_days, rows, base)
        if w:
            winners.append((label, w))

    print("\n" + "#" * 78)
    if not winners:
        print("FINAL: across every coin and strategy tested, nothing survived validation.")
        print("That is the honest result — no tradeable edge found. Do not risk money.")
        return
    label, w = winners[0]
    w.save("config.research.json")
    print(f"FINAL: best survivor was '{w.strategy}' on {label}. Wrote config.research.json.")
    print("Even so: paper-trade it forward for weeks before any live use. A holdout pass")
    print("is necessary, not proof — forward performance is the only thing that pays.")


if __name__ == "__main__":
    main()
