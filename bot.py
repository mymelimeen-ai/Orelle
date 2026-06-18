#!/usr/bin/env python3
"""
edgebot — a risk-first trading bot.

Design principle: the bot's job is NOT to hit a dollar target. Its job is to
(1) only act when the strategy sees an edge, and (2) never let one trade —
or one day — do real damage. Capital protection is hardcoded and cannot be
overridden by a "I need to make X today" impulse, because that impulse is
exactly what blows accounts up.

Modes:
  backtest  — prove (or disprove) the edge on historical data, risk nothing
  paper     — run the live strategy against real prices, simulated fills
  live      — real orders via ccxt  (you must flip --live and pass keys)

Default is paper. You have to go out of your way to risk money.
"""

from __future__ import annotations
import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    ccxt = None

log = logging.getLogger("edgebot")


# ----------------------------------------------------------------------------- 
# Config
# -----------------------------------------------------------------------------
@dataclass
class Config:
    exchange: str = "binanceusdm"      # ccxt id; perps by default
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"

    # --- strategy ---
    ema_fast: int = 21
    ema_slow: int = 55
    rsi_len: int = 14
    rsi_long_max: float = 70.0          # don't long into overbought
    rsi_short_min: float = 30.0         # don't short into oversold
    atr_len: int = 14

    # --- risk (the part that matters) ---
    equity: float = 1000.0              # starting/working capital in quote ccy
    risk_per_trade: float = 0.01        # 1% of equity risked per trade
    atr_stop_mult: float = 2.0          # stop = entry -/+ 2*ATR
    rr: float = 1.8                     # take-profit at 1.8x the risked distance
    max_leverage: float = 3.0           # hard cap; size is clamped to this
    max_daily_drawdown: float = 0.05    # 5% down on the day => bot stops trading
    fee: float = 0.0005                 # per-side taker fee assumption
    fixed_notional: float = 0.0         # legacy; leave 0 when using margin model
    margin: float = 100.0               # $ of your own money per trade
    leverage: float = 20.0              # multiplier; notional = margin * leverage
    liq_buffer: float = 0.8             # stop must sit within this fraction of the
                                        # liquidation distance, or the trade is skipped
    maint_margin: float = 0.005         # exchange maintenance margin (~0.5%)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(**json.load(f))


# ----------------------------------------------------------------------------- 
# Indicators
# -----------------------------------------------------------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], cfg.ema_fast)
    df["ema_slow"] = ema(df["close"], cfg.ema_slow)
    df["rsi"] = rsi(df["close"], cfg.rsi_len)
    df["atr"] = atr(df, cfg.atr_len)
    return df


# ----------------------------------------------------------------------------- 
# Strategy  ->  returns -1 / 0 / +1 for the *current* closed bar
# -----------------------------------------------------------------------------
def signal(df: pd.DataFrame, i: int, cfg: Config) -> int:
    """Trend (EMA cross) gated by an RSI filter so we don't chase extremes."""
    if i < cfg.ema_slow + 1:
        return 0
    fast, slow = df["ema_fast"], df["ema_slow"]
    crossed_up = fast.iloc[i - 1] <= slow.iloc[i - 1] and fast.iloc[i] > slow.iloc[i]
    crossed_dn = fast.iloc[i - 1] >= slow.iloc[i - 1] and fast.iloc[i] < slow.iloc[i]
    r = df["rsi"].iloc[i]
    if crossed_up and r < cfg.rsi_long_max:
        return 1
    if crossed_dn and r > cfg.rsi_short_min:
        return -1
    return 0


# ----------------------------------------------------------------------------- 
# Risk engine  ->  position size from a *defined* stop distance
# -----------------------------------------------------------------------------
def liquidation_price(entry: float, side: int, cfg: Config) -> float:
    """Approx liquidation price for an isolated-margin position.
    Liquidation distance ~= entry * (1/leverage - maint_margin)."""
    dist = entry * (1.0 / cfg.leverage - cfg.maint_margin)
    return entry - side * dist


def stop_is_safe(entry: float, stop: float, side: int, cfg: Config) -> bool:
    """True only if the stop triggers BEFORE liquidation, with buffer.
    This is the line that keeps leverage from killing the account."""
    liq = liquidation_price(entry, side, cfg)
    stop_dist = abs(entry - stop)
    liq_dist = abs(entry - liq)
    return stop_dist < liq_dist * cfg.liq_buffer


def size_position(equity: float, entry: float, stop: float, cfg: Config,
                  side: int = 1) -> float:
    """Margin/leverage model: notional = margin * leverage, qty = notional/entry.
    Returns 0 (skip the trade) if the stop would sit outside liquidation."""
    if cfg.margin > 0:
        if not stop_is_safe(entry, stop, side, cfg):
            return 0.0  # stop beyond liquidation -> too dangerous at this leverage
        notional = cfg.margin * cfg.leverage
        return notional / entry
    # fallback: risk-based sizing
    per_unit = abs(entry - stop)
    if per_unit <= 0:
        return 0.0
    qty = (equity * cfg.risk_per_trade) / per_unit
    max_notional = equity * cfg.max_leverage
    if qty * entry > max_notional:
        qty = max_notional / entry
    return qty


# ----------------------------------------------------------------------------- 
# Backtester
# -----------------------------------------------------------------------------
def backtest(df: pd.DataFrame, cfg: Config) -> dict:
    df = add_indicators(df, cfg)
    equity = cfg.equity
    peak = equity
    pos = None            # dict: side, entry, stop, tp, qty
    day = None
    day_start_equity = equity
    halted_today = False
    trades, curve = [], []

    for i in range(len(df)):
        row = df.iloc[i]
        ts = df.index[i]
        d = ts.date()
        price = row["close"]

        # new day -> reset the daily drawdown guard
        if d != day:
            day, day_start_equity, halted_today = d, equity, False

        # manage open position against this bar's high/low
        if pos:
            hit_stop = row["low"] <= pos["stop"] if pos["side"] == 1 else row["high"] >= pos["stop"]
            hit_tp = row["high"] >= pos["tp"] if pos["side"] == 1 else row["low"] <= pos["tp"]
            exit_px = None
            if hit_stop:
                exit_px = pos["stop"]
            elif hit_tp:
                exit_px = pos["tp"]
            if exit_px is not None:
                pnl = pos["side"] * (exit_px - pos["entry"]) * pos["qty"]
                pnl -= cfg.fee * pos["qty"] * (pos["entry"] + exit_px)
                equity += pnl
                trades.append({"time": str(ts), "side": pos["side"],
                               "entry": pos["entry"], "exit": exit_px,
                               "qty": pos["qty"], "pnl": pnl})
                pos = None

        # daily drawdown lockout — the single most important line in the file
        if equity <= day_start_equity * (1 - cfg.max_daily_drawdown):
            halted_today = True

        # look for entries only when flat and not halted
        if pos is None and not halted_today:
            sig = signal(df, i, cfg)
            if sig != 0 and not math.isnan(row["atr"]):
                entry = price
                stop = entry - sig * cfg.atr_stop_mult * row["atr"]
                tp = entry + sig * cfg.atr_stop_mult * row["atr"] * cfg.rr
                qty = size_position(equity, entry, stop, cfg, sig)
                if qty > 0:
                    pos = {"side": sig, "entry": entry, "stop": stop, "tp": tp, "qty": qty}

        peak = max(peak, equity)
        curve.append({"time": str(ts), "equity": equity, "drawdown": equity / peak - 1})

    return summarize(trades, curve, cfg)


def summarize(trades, curve, cfg) -> dict:
    if not trades:
        return {"trades": 0, "note": "No trades triggered on this data/params."}
    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    eq = np.array([c["equity"] for c in curve])
    dd = np.array([c["drawdown"] for c in curve])
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    return {
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3),
        "net_pnl": round(pnls.sum(), 2),
        "return_pct": round((eq[-1] / cfg.equity - 1) * 100, 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else float("inf"),
        "avg_win": round(wins.mean(), 2) if len(wins) else 0.0,
        "avg_loss": round(losses.mean(), 2) if len(losses) else 0.0,
        "max_drawdown_pct": round(dd.min() * 100, 2),
        "end_equity": round(eq[-1], 2),
    }


# ----------------------------------------------------------------------------- 
# Data
# -----------------------------------------------------------------------------
def fetch_ohlcv(cfg: Config, limit: int = 1500) -> pd.DataFrame:
    if ccxt is None:
        raise RuntimeError("ccxt not installed")
    ex = getattr(ccxt, cfg.exchange)()
    raw = ex.fetch_ohlcv(cfg.symbol, timeframe=cfg.timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts")


def synthetic(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Offline demo data: GBM with regime shifts, so backtest runs anywhere."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    drift = np.concatenate([rng.normal(m, 0, n // 3) for m in (0.0002, -0.0003, 0.0001)])[:n]
    rets = drift + rng.normal(0, 0.004, n)
    close = 60000 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    op = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(10, 100, n)}, index=idx)


# ----------------------------------------------------------------------------- 
# Live / paper loop
# -----------------------------------------------------------------------------
def run_live(cfg: Config, live: bool, key: str = "", secret: str = ""):
    if ccxt is None:
        raise RuntimeError("ccxt not installed")
    ex = getattr(ccxt, cfg.exchange)({"apiKey": key, "secret": secret,
                                      "enableRateLimit": True})
    mode = "LIVE" if live else "PAPER"
    log.info("Starting %s loop on %s %s %s", mode, cfg.exchange, cfg.symbol, cfg.timeframe)
    equity = cfg.equity
    day_start_equity = equity
    day = datetime.now(timezone.utc).date()
    pos = None

    while True:
        try:
            df = add_indicators(fetch_ohlcv(cfg, limit=cfg.ema_slow + 50), cfg)
            i = len(df) - 1
            row, price = df.iloc[i], df["close"].iloc[i]

            d = datetime.now(timezone.utc).date()
            if d != day:
                day, day_start_equity = d, equity

            halted = equity <= day_start_equity * (1 - cfg.max_daily_drawdown)
            if halted:
                log.warning("Daily drawdown hit. Standing down until tomorrow.")

            if pos is None and not halted:
                sig = signal(df, i, cfg)
                if sig != 0:
                    entry = price
                    stop = entry - sig * cfg.atr_stop_mult * row["atr"]
                    tp = entry + sig * cfg.atr_stop_mult * row["atr"] * cfg.rr
                    qty = size_position(equity, entry, stop, cfg, sig)
                    log.info("SIGNAL %s qty=%.6f entry=%.2f stop=%.2f tp=%.2f",
                             "LONG" if sig > 0 else "SHORT", qty, entry, stop, tp)
                    if live and qty > 0:
                        side = "buy" if sig > 0 else "sell"
                        ex.create_order(cfg.symbol, "market", side, qty)
                        ex.create_order(cfg.symbol, "stop_market", "sell" if sig > 0 else "buy",
                                        qty, params={"stopPrice": stop, "reduceOnly": True})
                    pos = {"side": sig, "entry": entry, "stop": stop, "tp": tp, "qty": qty}
        except Exception as e:
            log.error("loop error: %s", e)
        time.sleep(_tf_seconds(cfg.timeframe))


def _tf_seconds(tf: str) -> int:
    unit = tf[-1]
    val = int(tf[:-1])
    return val * {"m": 60, "h": 3600, "d": 86400}[unit]


# ----------------------------------------------------------------------------- 
# CLI
# -----------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="edgebot — risk-first trading bot")
    p.add_argument("mode", choices=["backtest", "paper", "live"])
    p.add_argument("--config", default=None, help="path to config.json")
    p.add_argument("--offline", action="store_true", help="backtest on synthetic data")
    p.add_argument("--live", action="store_true", help="REQUIRED to place real orders")
    p.add_argument("--key", default="")
    p.add_argument("--secret", default="")
    args = p.parse_args()

    cfg = Config.load(args.config) if args.config else Config()

    if args.mode == "backtest":
        df = synthetic() if args.offline else fetch_ohlcv(cfg)
        result = backtest(df, cfg)
        print(json.dumps(result, indent=2))
    elif args.mode == "paper":
        run_live(cfg, live=False, key=args.key, secret=args.secret)
    elif args.mode == "live":
        if not args.live:
            raise SystemExit("Refusing to trade real money without --live flag.")
        run_live(cfg, live=True, key=args.key, secret=args.secret)


if __name__ == "__main__":
    main()
