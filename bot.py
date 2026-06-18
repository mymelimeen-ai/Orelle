#!/usr/bin/env python3
"""
edgebot — a risk-first crypto futures trading bot.

WHAT THIS IS
  A bot that takes 3:1 setups. It risks a fixed, hard-capped dollar amount per
  trade (default $100) and targets 3x that ($300) when a valid signal appears.
  Every loss is bounded to the risk amount; there is NO daily profit quota,
  because forcing a fixed daily P&L is exactly how accounts die — it makes the
  bot trade when there is no edge. Some days it makes $300+, some days nothing,
  some days a capped loss. That is what honest looks like.

WHAT THIS IS NOT
  A machine that prints $300 every single day. That does not exist. This bot's
  only promise is: when it acts, the downside is bounded and the upside is ~3x
  the downside. Whether the whole thing is profitable depends on the strategy's
  edge — which you PROVE on real data (backtest) before risking a cent.

MODES
  backtest  — prove (or disprove) the edge on historical data, risk nothing
  paper     — live prices, simulated fills, no money at risk
  live      — real orders via ccxt (requires --live AND API keys)

Default is paper. You have to go out of your way to risk money.
"""

from __future__ import annotations
import argparse
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
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
    exchange: str = "binanceusdm"       # ccxt id; perps by default
    symbol: str = "SOL/USDT"
    timeframe: str = "15m"

    # --- strategy ---
    ema_fast: int = 21
    ema_slow: int = 55
    rsi_len: int = 14
    rsi_long_max: float = 70.0          # don't long into overbought
    rsi_short_min: float = 30.0         # don't short into oversold
    atr_len: int = 14
    atr_stop_mult: float = 2.0          # stop = entry -/+ 2*ATR
    rr: float = 3.0                     # take-profit at 3x risk -> $100 risk targets $300

    # --- account / risk (the part that actually matters) ---
    equity: float = 1000.0              # your working capital in quote ccy
    risk_model: str = "fixed_risk"      # "fixed_risk" (recommended) or "margin" (legacy)
    risk_dollars: float = 100.0         # HARD cap: max loss per trade in $
    risk_per_trade: float = 0.0         # if risk_dollars==0, risk this fraction of equity instead
    max_leverage: float = 10.0          # hard cap on notional / equity
    max_daily_drawdown: float = 0.06    # stop trading for the day after this % down
    max_consec_losses: int = 4          # cool off (skip) after N losses in a row
    fee: float = 0.0005                 # per-side taker fee assumption
    maint_margin: float = 0.005         # exchange maintenance margin (~0.5%)
    liq_buffer: float = 0.5             # stop must sit well inside the liquidation distance

    # --- legacy fixed-notional model (kept for reference; NOT the default) ---
    margin: float = 0.0                 # $ margin per trade; only used if risk_model=="margin"
    leverage: float = 0.0               # multiplier; notional = margin * leverage

    def risk_amount(self) -> float:
        """Dollars at risk per trade under the active model."""
        if self.risk_dollars > 0:
            return self.risk_dollars
        return self.equity * self.risk_per_trade

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            data = json.load(f)
        # tolerate older config files that carry retired keys
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in valid})


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
def stop_is_safe(entry: float, stop: float, qty: float, equity: float,
                 cfg: Config) -> bool:
    """True only if the stop would trigger BEFORE liquidation, with buffer.
    This is the line that keeps leverage from quietly killing the account."""
    notional = qty * entry
    if notional <= 0 or equity <= 0:
        return False
    lev = notional / equity
    if lev <= 1.0:
        return True                      # no meaningful liquidation risk
    liq_dist = entry * (1.0 / lev - cfg.maint_margin)
    if liq_dist <= 0:
        return False                     # leverage so high liquidation is ~immediate
    return abs(entry - stop) < liq_dist * cfg.liq_buffer


def size_position(equity: float, entry: float, stop: float, cfg: Config,
                  side: int = 1) -> float:
    """Size so that hitting the stop loses ~`risk_dollars` (or risk_per_trade).

    Returns qty (base units). Returns 0 -> skip the trade, when:
      - stop distance is zero, or
      - even after clamping to max_leverage the stop sits beyond liquidation.
    """
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return 0.0

    if cfg.risk_model == "margin" and cfg.margin > 0:
        # legacy fixed-notional model: notional = margin * leverage
        qty = (cfg.margin * cfg.leverage) / entry
    else:
        risk_amt = cfg.risk_amount()
        if risk_amt <= 0:
            return 0.0
        qty = risk_amt / stop_dist       # loss at stop = qty * stop_dist = risk_amt

    # hard leverage cap: never let notional exceed max_leverage * equity
    max_notional = equity * cfg.max_leverage
    if qty * entry > max_notional:
        qty = max_notional / entry

    # final safety: the stop must sit inside the liquidation distance
    if not stop_is_safe(entry, stop, qty, equity, cfg):
        return 0.0
    return qty


# -----------------------------------------------------------------------------
# Backtester
# -----------------------------------------------------------------------------
def backtest(df: pd.DataFrame, cfg: Config) -> dict:
    df = add_indicators(df, cfg)
    equity = cfg.equity
    peak = equity
    pos = None                 # dict: side, entry, stop, tp, qty, risk
    day = None
    day_start_equity = equity
    halted_today = False
    consec_losses = 0
    trades, curve = [], []

    for i in range(len(df)):
        row = df.iloc[i]
        ts = df.index[i]
        d = ts.date()
        price = row["close"]

        # new day -> reset the daily guards
        if d != day:
            day, day_start_equity, halted_today = d, equity, False

        # manage open position against this bar's high/low
        if pos:
            if pos["side"] == 1:
                hit_stop = row["low"] <= pos["stop"]
                hit_tp = row["high"] >= pos["tp"]
            else:
                hit_stop = row["high"] >= pos["stop"]
                hit_tp = row["low"] <= pos["tp"]
            # if a bar straddles both, assume the stop fills first (conservative)
            exit_px = pos["stop"] if hit_stop else (pos["tp"] if hit_tp else None)
            if exit_px is not None:
                pnl = pos["side"] * (exit_px - pos["entry"]) * pos["qty"]
                pnl -= cfg.fee * pos["qty"] * (pos["entry"] + exit_px)
                equity += pnl
                consec_losses = consec_losses + 1 if pnl < 0 else 0
                trades.append({
                    "time": str(ts), "side": pos["side"],
                    "entry": pos["entry"], "exit": exit_px, "qty": pos["qty"],
                    "pnl": pnl, "r": pnl / pos["risk"] if pos["risk"] else 0.0,
                })
                pos = None

        # daily drawdown lockout — the single most important guard in the file
        if equity <= day_start_equity * (1 - cfg.max_daily_drawdown):
            halted_today = True

        # look for entries only when flat, not halted, not on a losing streak
        if pos is None and not halted_today and consec_losses < cfg.max_consec_losses:
            sig = signal(df, i, cfg)
            if sig != 0 and not math.isnan(row["atr"]):
                entry = price
                stop = entry - sig * cfg.atr_stop_mult * row["atr"]
                tp = entry + sig * cfg.atr_stop_mult * row["atr"] * cfg.rr
                qty = size_position(equity, entry, stop, cfg, sig)
                if qty > 0:
                    risk = abs(entry - stop) * qty
                    pos = {"side": sig, "entry": entry, "stop": stop,
                           "tp": tp, "qty": qty, "risk": risk}

        peak = max(peak, equity)
        curve.append({"time": str(ts), "equity": equity, "drawdown": equity / peak - 1})

    return summarize(trades, curve, cfg)


def summarize(trades, curve, cfg) -> dict:
    if not trades:
        return {"trades": 0, "note": "No trades triggered on this data/params."}
    pnls = np.array([t["pnl"] for t in trades])
    rs = np.array([t["r"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    eq = np.array([c["equity"] for c in curve])
    dd = np.array([c["drawdown"] for c in curve])
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0

    # per-DAY P&L: this is what answers "how often did a day clear +$300?"
    daily = defaultdict(float)
    for t in trades:
        daily[t["time"][:10]] += t["pnl"]
    dvals = np.array(list(daily.values()))
    target = cfg.rr * cfg.risk_amount()      # e.g. 3 * $100 = $300

    # longest losing streak
    mcl = cur = 0
    for p in pnls:
        cur = cur + 1 if p < 0 else 0
        mcl = max(mcl, cur)

    return {
        # --- trade-level ---
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3),
        "expectancy_R": round(float(rs.mean()), 3),       # avg R per trade (>0 = edge)
        "expectancy_usd": round(float(pnls.mean()), 2),
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else float("inf"),
        "max_consec_losses": int(mcl),
        # --- account-level ---
        "net_pnl": round(float(pnls.sum()), 2),
        "return_pct": round((eq[-1] / cfg.equity - 1) * 100, 2),
        "max_drawdown_pct": round(float(dd.min()) * 100, 2),
        "end_equity": round(float(eq[-1]), 2),
        # --- the honest daily reality (your $300 question) ---
        "target_usd_per_day": round(target, 2),
        "trading_days": len(daily),
        "trades_per_day": round(len(trades) / max(len(daily), 1), 2),
        "avg_day_usd": round(float(dvals.mean()), 2),
        "best_day_usd": round(float(dvals.max()), 2),
        "worst_day_usd": round(float(dvals.min()), 2),
        "green_days": int((dvals > 0).sum()),
        "red_days": int((dvals < 0).sum()),
        "days_hit_target": int((dvals >= target).sum()),
        "pct_days_hit_target": round(100 * float((dvals >= target).mean()), 1),
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


def load_csv(path: str) -> pd.DataFrame:
    """Backtest real data WITHOUT touching the network. Download candles anywhere
    (exchange export, data vendor, your own dump) into a CSV with columns:
        time/ts/timestamp/date, open, high, low, close, volume
    Time may be epoch ms, epoch s, or an ISO timestamp."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in ("ts", "time", "timestamp", "date", "datetime")
                 if c in df.columns), None)
    if tcol is None:
        raise ValueError("CSV needs a time column: ts/time/timestamp/date/datetime")
    s = df[tcol]
    if pd.api.types.is_numeric_dtype(s):
        unit = "ms" if float(s.iloc[0]) > 1e11 else "s"
        df[tcol] = pd.to_datetime(s, unit=unit, utc=True)
    else:
        df[tcol] = pd.to_datetime(s, utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")
    return df.set_index(tcol)[["open", "high", "low", "close", "volume"]].sort_index()


def synthetic(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Offline demo data: GBM with regime shifts, so the engine runs anywhere.
    NOTE: this is random noise, NOT a real market. Use it to verify mechanics
    only — never to judge whether the strategy is profitable on a real symbol."""
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
    log.info("Starting %s loop on %s %s %s | risk $%.0f/trade, target $%.0f (%.1fR)",
             mode, cfg.exchange, cfg.symbol, cfg.timeframe,
             cfg.risk_amount(), cfg.rr * cfg.risk_amount(), cfg.rr)
    equity = cfg.equity
    day_start_equity = equity
    day = datetime.now(timezone.utc).date()
    consec_losses = 0
    pos = None

    while True:
        try:
            df = add_indicators(fetch_ohlcv(cfg, limit=cfg.ema_slow + 50), cfg)
            i = len(df) - 1
            row, price = df.iloc[i], df["close"].iloc[i]

            d = datetime.now(timezone.utc).date()
            if d != day:
                day, day_start_equity, consec_losses = d, equity, 0

            halted = (equity <= day_start_equity * (1 - cfg.max_daily_drawdown)
                      or consec_losses >= cfg.max_consec_losses)
            if halted:
                log.warning("Risk guard active (daily DD or loss streak). Standing down.")

            if pos is None and not halted:
                sig = signal(df, i, cfg)
                if sig != 0:
                    entry = price
                    stop = entry - sig * cfg.atr_stop_mult * row["atr"]
                    tp = entry + sig * cfg.atr_stop_mult * row["atr"] * cfg.rr
                    qty = size_position(equity, entry, stop, cfg, sig)
                    log.info("SIGNAL %s qty=%.6f entry=%.4f stop=%.4f tp=%.4f (lev~%.1fx)",
                             "LONG" if sig > 0 else "SHORT", qty, entry, stop, tp,
                             (qty * entry / equity) if equity else 0.0)
                    if qty <= 0:
                        log.info("Skipped: stop sits beyond safe liquidation distance.")
                    elif live:
                        side = "buy" if sig > 0 else "sell"
                        close_side = "sell" if sig > 0 else "buy"
                        ex.create_order(cfg.symbol, "market", side, qty)
                        ex.create_order(cfg.symbol, "stop_market", close_side, qty,
                                        params={"stopPrice": stop, "reduceOnly": True})
                        ex.create_order(cfg.symbol, "take_profit_market", close_side, qty,
                                        params={"stopPrice": tp, "reduceOnly": True})
                    if qty > 0:
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
    p = argparse.ArgumentParser(description="edgebot — risk-first crypto futures bot")
    p.add_argument("mode", choices=["backtest", "paper", "live"])
    p.add_argument("--config", default=None, help="path to config.json")
    p.add_argument("--offline", action="store_true", help="backtest on synthetic data (mechanics only)")
    p.add_argument("--csv", default=None, help="backtest on a local OHLCV CSV (real data, no network)")
    p.add_argument("--live", action="store_true", help="REQUIRED to place real orders")
    p.add_argument("--key", default="")
    p.add_argument("--secret", default="")
    args = p.parse_args()

    cfg = Config.load(args.config) if args.config else Config()

    if args.mode == "backtest":
        if args.csv:
            df = load_csv(args.csv)
        elif args.offline:
            df = synthetic()
        else:
            df = fetch_ohlcv(cfg)
        print(json.dumps(backtest(df, cfg), indent=2))
    elif args.mode == "paper":
        run_live(cfg, live=False, key=args.key, secret=args.secret)
    elif args.mode == "live":
        if not args.live:
            raise SystemExit("Refusing to trade real money without --live flag.")
        run_live(cfg, live=True, key=args.key, secret=args.secret)


if __name__ == "__main__":
    main()
