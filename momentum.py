#!/usr/bin/env python3
"""
momentum.py — cross-sectional momentum on a basket of crypto perps.

WHY THIS IS DIFFERENT (and has a better shot than single-asset indicator timing):
  Instead of timing one coin with chart indicators, we rank a *basket* by recent
  return and go LONG the strongest / SHORT the weakest, rebalancing periodically,
  roughly dollar-neutral. Momentum is the most robustly documented anomaly across
  every asset class (incl. crypto) because it has behavioral/flow reasons to
  persist — it's a risk premium, not a chart shape. Diversifying across many
  names means no single trade matters; you win slightly more than you lose, often.

  This is NOT a 1m/15m scalper. Momentum lives on daily / multi-day horizons.
  Feed DAILY (or 4h) candles for ~15-40 liquid coins.

HONESTY: this is better *odds*, not free money. Expect modest returns, real
drawdowns, momentum-crash risk, and decay. Funding costs are only approximated
here via a turnover fee — model them properly before trusting any live number.

Usage (run where the data is, i.e. locally):
    python3 fetch_data.py --symbols BTC/USDT ETH/USDT SOL/USDT ... --timeframe 1d --days 1000
    python3 momentum.py --csv *_1d.csv
    python3 momentum.py --offline           # synthetic basket smoke test
"""

from __future__ import annotations
import argparse
import glob
import json
import os
from itertools import product

import numpy as np
import pandas as pd

from bot import load_csv, synthetic


# --- search grid -------------------------------------------------------------
GRID = {
    "lookback":  [10, 20, 40, 60],      # bars of trailing return used to rank
    "rebal":     [1, 5, 10],            # rebalance every N bars
    "quantile":  [0.25, 0.34],          # fraction of basket in each leg
    "vol_adjust": [False, True],        # rank by return/vol and size inverse-vol
}


# --- data --------------------------------------------------------------------
def load_panel(files) -> pd.DataFrame:
    """Build an aligned close-price panel: index=time, columns=symbol."""
    cols = {}
    for f in files:
        sym = os.path.basename(f).split("_")[0].upper()
        cols[sym] = load_csv(f)["close"]
    panel = pd.DataFrame(cols).sort_index()
    return panel.dropna(how="all").ffill().dropna()


def synthetic_panel(k: int = 12, n: int = 900, freq: str = "1D") -> pd.DataFrame:
    """A basket of independent random walks — for mechanics only. Cross-sectional
    momentum on noise should show ~no edge; that's the honest expectation."""
    return pd.DataFrame({f"C{i}": synthetic(n=n, seed=i + 1, freq=freq)["close"].values
                         for i in range(k)},
                        index=synthetic(n=n, seed=1, freq=freq).index)


def bars_per_year(idx) -> float:
    sec = idx.to_series().diff().dropna().median().total_seconds()
    return (365 * 86400) / sec if sec and sec > 0 else 365.0


# --- portfolio backtest ------------------------------------------------------
def backtest_xs(prices, lookback, rebal, quantile, vol_adjust,
                fee=0.0005, gross=1.0, start=0, end=None):
    """Long top / short bottom by momentum, rebalanced every `rebal` bars,
    dollar-neutral. Signals use full history (no warmup loss); trading is active
    only on [start, end). No lookahead: a bar's return uses weights set on a
    PRIOR bar, and the rebalance at bar t uses only info through t."""
    end = len(prices) if end is None else end
    rets = prices.pct_change().fillna(0.0)
    mom = prices / prices.shift(lookback) - 1.0
    vol = rets.rolling(lookback).std()
    sig = (mom / vol.replace(0, np.nan)) if vol_adjust else mom

    cols = prices.columns
    W = pd.Series(0.0, index=cols)
    prev_W = W.copy()
    equity = 1.0
    port_rets = []
    curve = []

    for t in range(start, end):
        # apply this bar's return with weights held from the last rebalance
        if t > start:
            r = float((W * rets.iloc[t]).sum())
            equity *= (1.0 + r)
            port_rets.append(r)
            if equity <= 0:                      # ruin guard
                curve.append(0.0)
                break
        # rebalance at end of bar t (uses info through close t)
        if t >= lookback and (t % rebal == 0):
            s = sig.iloc[t].dropna()
            if len(s) >= 4:
                k = max(1, int(len(s) * quantile))
                ranked = s.sort_values()
                shorts, longs = ranked.index[:k], ranked.index[-k:]
                iv = (1.0 / vol.iloc[t]).replace([np.inf, -np.inf], np.nan) if vol_adjust \
                    else pd.Series(1.0, index=cols)
                w = pd.Series(0.0, index=cols)
                lw, sw = iv[longs].fillna(0), iv[shorts].fillna(0)
                if lw.sum() > 0 and sw.sum() > 0:
                    w[longs] = (lw / lw.sum()) * (gross / 2)
                    w[shorts] = -(sw / sw.sum()) * (gross / 2)
                    equity *= (1.0 - fee * float((w - prev_W).abs().sum()))
                    prev_W, W = w.copy(), w
        curve.append(equity)

    return np.array(port_rets), np.array(curve)


def metrics(port_rets, curve, bpy) -> dict | None:
    if len(port_rets) == 0 or len(curve) == 0:
        return None
    pr = np.asarray(port_rets)
    eq = np.asarray(curve)
    final = max(eq[-1], 1e-9)
    ann_ret = final ** (bpy / len(pr)) - 1.0
    ann_vol = pr.std() * np.sqrt(bpy)
    sharpe = (pr.mean() * bpy) / ann_vol if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = float((eq / peak - 1.0).min())
    return {
        "sharpe": round(float(sharpe), 2),
        "ann_return_pct": round(float(ann_ret) * 100, 1),
        "ann_vol_pct": round(float(ann_vol) * 100, 1),
        "max_drawdown_pct": round(dd * 100, 1),
        "total_return_pct": round((final - 1.0) * 100, 1),
        "pct_positive": round(float((pr > 0).mean()) * 100, 1),
        "periods": len(pr),
    }


# --- walk-forward + locked holdout ------------------------------------------
def _combos():
    return [dict(zip(GRID, v)) for v in product(*GRID.values())]


def evaluate(prices, train, test, holdout, fee, gross):
    bpy = bars_per_year(prices.index)
    n = len(prices)
    hb = holdout if 0 < holdout < n * 0.4 else (int(n * 0.2) if holdout else 0)
    dev_end = n - hb if hb else n

    # rolling walk-forward over the development span -> concatenated OOS returns
    folds, s = [], 0
    while s + train + test <= dev_end:
        folds.append((s, s + train, s + train + test))
        s += test
    oos = []
    for a, b, c in folds:
        best, bscore = None, -1e18
        for combo in _combos():
            pr, cv = backtest_xs(prices, **combo, fee=fee, gross=gross, start=a, end=b)
            m = metrics(pr, cv, bpy)
            sc = m["sharpe"] if m and m["periods"] >= test * 0.3 else -1e18
            if sc > bscore:
                bscore, best = sc, combo
        if best:
            pr, _ = backtest_xs(prices, **best, fee=fee, gross=gross, start=b, end=c)
            oos.extend(pr.tolist())
    oos_m = metrics(np.array(oos), np.cumprod(1 + np.array(oos)) if oos else [], bpy) if oos else None

    # one combo chosen on ALL dev data, scored ONCE on the locked holdout
    dev_best, dev_score = None, -1e18
    for combo in _combos():
        pr, cv = backtest_xs(prices, **combo, fee=fee, gross=gross, start=0, end=dev_end)
        m = metrics(pr, cv, bpy)
        if m and m["periods"] >= train * 0.3 and m["sharpe"] > dev_score:
            dev_score, dev_best = m["sharpe"], combo
    hold_m = None
    if hb and dev_best:
        pr, cv = backtest_xs(prices, **dev_best, fee=fee, gross=gross, start=dev_end, end=n)
        hold_m = metrics(pr, cv, bpy)
    return bpy, hb, oos_m, hold_m, dev_best


def main():
    p = argparse.ArgumentParser(description="cross-sectional momentum on a basket")
    p.add_argument("--csv", nargs="+", help="daily/4h OHLCV CSVs, one per coin")
    p.add_argument("--offline", action="store_true", help="synthetic basket smoke test")
    p.add_argument("--train", type=int, default=252, help="train window (bars)")
    p.add_argument("--test", type=int, default=63, help="test window (bars)")
    p.add_argument("--holdout", type=int, default=126, help="locked holdout (bars); 0=off")
    p.add_argument("--fee", type=float, default=0.0005, help="cost per unit turnover")
    p.add_argument("--gross", type=float, default=1.0, help="gross leverage (0.5 long+0.5 short=1)")
    args = p.parse_args()

    if args.offline:
        prices = synthetic_panel()
    elif args.csv:
        files = [f for pat in args.csv for f in glob.glob(pat)]
        if len(files) < 4:
            raise SystemExit("Need at least ~4 coins for a cross-section. Pass more CSVs.")
        prices = load_panel(files)
    else:
        raise SystemExit("Pass --csv <files...> or --offline.")

    print(f"basket: {len(prices.columns)} coins, {len(prices)} bars  "
          f"{prices.index[0]} -> {prices.index[-1]}")
    print(f"walk-forward train {args.train} / test {args.test} bars, "
          f"locked holdout {args.holdout} bars. Sharpe is the headline number.\n")

    bpy, hb, oos, hold, best = evaluate(prices, args.train, args.test,
                                        args.holdout, args.fee, args.gross)
    print(f"(~{bpy:.0f} bars/year)")
    print("OUT-OF-SAMPLE (rolling, honest):", json.dumps(oos) if oos else "n/a")
    if hb:
        print("LOCKED HOLDOUT (untouched, final):", json.dumps(hold) if hold else "n/a")
    print("params chosen on dev:", best)

    print("\n" + "=" * 70)
    ok_oos = oos and oos["sharpe"] > 0 and oos["ann_return_pct"] > 0
    ok_hold = (not hb) or (hold and hold["sharpe"] > 0.5 and hold["ann_return_pct"] > 0)
    if ok_oos and ok_hold:
        final = hold or oos
        print(f"VERDICT: survived. Sharpe {final['sharpe']}, "
              f"{final['ann_return_pct']}%/yr, max DD {final['max_drawdown_pct']}%.")
        print("Still: paper-trade forward for weeks and model funding/slippage "
              "before any live use. A holdout pass is necessary, not proof.")
    else:
        print("VERDICT: no robust cross-sectional momentum edge survived validation.")
        print("Honest result — not worth risking money as-is.")


if __name__ == "__main__":
    main()
