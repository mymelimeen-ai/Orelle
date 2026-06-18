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
from dataclasses import dataclass, asdict, replace
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

    # --- strategy selection & per-strategy params ---
    # ema_cross | ema_cross_adx | rsi_reversion | bb_reversion | donchian_breakout
    strategy: str = "ema_cross"
    adx_len: int = 14
    adx_min: float = 20.0               # ema_cross_adx: require trend strength >= this (skip chop)
    htf_len: int = 200                  # ema_cross_adx: long-trend EMA used as a regime filter
    rsi_os: float = 30.0                # rsi_reversion oversold / bb re-entry threshold
    rsi_ob: float = 70.0                # rsi_reversion overbought
    bb_len: int = 20                    # bollinger length
    bb_std: float = 2.0                 # bollinger band width (std devs)
    donchian_len: int = 20              # breakout lookback
    htf_rule: str = "15min"             # mtf: slow timeframe to resample to (input = 1m)

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

def _true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return _true_range(df).ewm(alpha=1 / n, adjust=False).mean()

def adx(df: pd.DataFrame, n: int) -> pd.Series:
    """Wilder's ADX — trend-strength gauge. High = trending, low (<~20) = chop."""
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr_w = _true_range(df).ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_w
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_w
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0.0)

def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], cfg.ema_fast)
    df["ema_slow"] = ema(df["close"], cfg.ema_slow)
    df["rsi"] = rsi(df["close"], cfg.rsi_len)
    df["atr"] = atr(df, cfg.atr_len)
    # extra indicators used by the alternative strategies (cheap to always compute)
    df["adx"] = adx(df, cfg.adx_len)
    df["htf_ema"] = ema(df["close"], cfg.htf_len)
    mid = df["close"].rolling(cfg.bb_len).mean()
    sd = df["close"].rolling(cfg.bb_len).std(ddof=0)
    df["bb_upper"] = mid + cfg.bb_std * sd
    df["bb_lower"] = mid - cfg.bb_std * sd
    df["dc_high"] = df["high"].rolling(cfg.donchian_len).max().shift(1)
    df["dc_low"] = df["low"].rolling(cfg.donchian_len).min().shift(1)
    return df


def add_mtf_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Multi-timeframe: the input is the FAST timeframe (e.g. 1m). We compute the
    entry-timeframe indicators on it, then resample to the SLOW timeframe
    (cfg.htf_rule, e.g. '15min') for the trend bias and align it back — using only
    *closed* slow bars (shift by one) so there is no lookahead."""
    df = df.copy()
    # fast-timeframe (entry) indicators
    df["rsi"] = rsi(df["close"], cfg.rsi_len)
    df["atr"] = atr(df, cfg.atr_len)
    df["ema_fast"] = ema(df["close"], cfg.ema_fast)
    df["ema_slow"] = ema(df["close"], cfg.ema_slow)

    # slow-timeframe (bias) via resample
    rs = df.resample(cfg.htf_rule, label="left", closed="left")
    htf = pd.DataFrame({
        "open": rs["open"].first(), "high": rs["high"].max(),
        "low": rs["low"].min(), "close": rs["close"].last(),
    }).dropna()
    ef, es = ema(htf["close"], cfg.ema_fast), ema(htf["close"], cfg.ema_slow)
    htf_trend = np.sign(ef - es)                     # +1 up / -1 down
    htf_adx = adx(htf, cfg.adx_len)
    # only known AFTER the slow bar closes -> shift one slow bar, then ffill onto fast
    df["htf_trend"] = htf_trend.shift(1).reindex(df.index, method="ffill")
    df["htf_adx"] = htf_adx.shift(1).reindex(df.index, method="ffill")
    return df


def prepare_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Route to the right indicator builder for the configured strategy."""
    return add_mtf_indicators(df, cfg) if cfg.strategy == "mtf" else add_indicators(df, cfg)


# -----------------------------------------------------------------------------
# Strategy  ->  returns -1 / 0 / +1 for the *current* closed bar
# -----------------------------------------------------------------------------
def _sig_ema_cross(df: pd.DataFrame, i: int, cfg: Config) -> int:
    """Trend: EMA fast/slow cross, gated by an RSI filter so we don't chase extremes."""
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


def _sig_ema_cross_adx(df: pd.DataFrame, i: int, cfg: Config) -> int:
    """EMA cross, but only in a trending regime (ADX) and aligned with the
    higher-timeframe trend. This is the 'don't trade chop' fix for the bleed we saw."""
    base = _sig_ema_cross(df, i, cfg)
    if base == 0:
        return 0
    if df["adx"].iloc[i] < cfg.adx_min:          # too choppy -> sit out
        return 0
    price, htf = df["close"].iloc[i], df["htf_ema"].iloc[i]
    if base == 1 and price < htf:                # long only with the long-term trend
        return 0
    if base == -1 and price > htf:               # short only against it
        return 0
    return base


def _sig_rsi_reversion(df: pd.DataFrame, i: int, cfg: Config) -> int:
    """Mean reversion: buy when RSI climbs back out of oversold, sell out of overbought."""
    if i < cfg.rsi_len + 1:
        return 0
    r0, r1 = df["rsi"].iloc[i - 1], df["rsi"].iloc[i]
    if r0 <= cfg.rsi_os and r1 > cfg.rsi_os:
        return 1
    if r0 >= cfg.rsi_ob and r1 < cfg.rsi_ob:
        return -1
    return 0


def _sig_bb_reversion(df: pd.DataFrame, i: int, cfg: Config) -> int:
    """Mean reversion: buy when price re-enters from below the lower Bollinger band."""
    if i < cfg.bb_len + 1 or math.isnan(df["bb_lower"].iloc[i]):
        return 0
    c0, c1 = df["close"].iloc[i - 1], df["close"].iloc[i]
    if c0 < df["bb_lower"].iloc[i - 1] and c1 >= df["bb_lower"].iloc[i]:
        return 1
    if c0 > df["bb_upper"].iloc[i - 1] and c1 <= df["bb_upper"].iloc[i]:
        return -1
    return 0


def _sig_donchian(df: pd.DataFrame, i: int, cfg: Config) -> int:
    """Breakout: long when price closes above the prior N-bar high, short below the low."""
    if i < cfg.donchian_len + 1 or math.isnan(df["dc_high"].iloc[i]):
        return 0
    c = df["close"].iloc[i]
    if c > df["dc_high"].iloc[i]:
        return 1
    if c < df["dc_low"].iloc[i]:
        return -1
    return 0


def _sig_mtf(df: pd.DataFrame, i: int, cfg: Config) -> int:
    """Multi-timeframe: take a 1m RSI pullback ONLY in the direction of the
    closed-15m trend (and only when that trend is strong enough, via 15m ADX)."""
    if i < cfg.rsi_len + 1:
        return 0
    trend, adx15 = df["htf_trend"].iloc[i], df["htf_adx"].iloc[i]
    if math.isnan(trend) or adx15 < cfg.adx_min:
        return 0
    r0, r1 = df["rsi"].iloc[i - 1], df["rsi"].iloc[i]
    if trend > 0 and r0 <= cfg.rsi_os and r1 > cfg.rsi_os:
        return 1
    if trend < 0 and r0 >= cfg.rsi_ob and r1 < cfg.rsi_ob:
        return -1
    return 0


_STRATEGIES = {
    "ema_cross": _sig_ema_cross,
    "ema_cross_adx": _sig_ema_cross_adx,
    "rsi_reversion": _sig_rsi_reversion,
    "bb_reversion": _sig_bb_reversion,
    "donchian_breakout": _sig_donchian,
    "mtf": _sig_mtf,
}


def signal(df: pd.DataFrame, i: int, cfg: Config) -> int:
    """Dispatch to the configured strategy. Returns -1 / 0 / +1 for closed bar i.
    Used by the live loop; the backtester uses the vectorized signals_array()."""
    try:
        return _STRATEGIES[cfg.strategy](df, i, cfg)
    except KeyError:
        raise ValueError(f"unknown strategy {cfg.strategy!r}; "
                         f"choices: {', '.join(_STRATEGIES)}")


def signals_array(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Vectorized equivalent of signal() for the whole frame — same semantics,
    far faster than calling signal() per bar in the backtest loop."""
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    strat = cfg.strategy
    if strat in ("ema_cross", "ema_cross_adx"):
        f = df["ema_fast"].to_numpy(); s = df["ema_slow"].to_numpy(); r = df["rsi"].to_numpy()
        up = np.zeros(n, bool); dn = np.zeros(n, bool)
        up[1:] = (f[:-1] <= s[:-1]) & (f[1:] > s[1:])
        dn[1:] = (f[:-1] >= s[:-1]) & (f[1:] < s[1:])
        sig[up & (r < cfg.rsi_long_max)] = 1
        sig[dn & (r > cfg.rsi_short_min)] = -1
        sig[:cfg.ema_slow + 1] = 0
        if strat == "ema_cross_adx":
            price = df["close"].to_numpy(); htf = df["htf_ema"].to_numpy()
            sig[df["adx"].to_numpy() < cfg.adx_min] = 0
            sig[(sig == 1) & (price < htf)] = 0
            sig[(sig == -1) & (price > htf)] = 0
    elif strat == "rsi_reversion":
        r = df["rsi"].to_numpy()
        up = np.zeros(n, bool); dn = np.zeros(n, bool)
        up[1:] = (r[:-1] <= cfg.rsi_os) & (r[1:] > cfg.rsi_os)
        dn[1:] = (r[:-1] >= cfg.rsi_ob) & (r[1:] < cfg.rsi_ob)
        sig[up] = 1; sig[dn] = -1
        sig[:cfg.rsi_len + 1] = 0
    elif strat == "bb_reversion":
        c = df["close"].to_numpy(); lo = df["bb_lower"].to_numpy(); up_ = df["bb_upper"].to_numpy()
        L = np.zeros(n, bool); S = np.zeros(n, bool)
        L[1:] = (c[:-1] < lo[:-1]) & (c[1:] >= lo[1:])
        S[1:] = (c[:-1] > up_[:-1]) & (c[1:] <= up_[1:])
        sig[L] = 1; sig[S] = -1
        sig[np.isnan(lo)] = 0
    elif strat == "donchian_breakout":
        c = df["close"].to_numpy(); hi = df["dc_high"].to_numpy(); lo = df["dc_low"].to_numpy()
        sig[c > hi] = 1; sig[c < lo] = -1
        sig[np.isnan(hi)] = 0
    elif strat == "mtf":
        r = df["rsi"].to_numpy()
        trend = df["htf_trend"].to_numpy(); adx15 = df["htf_adx"].to_numpy()
        up = np.zeros(n, bool); dn = np.zeros(n, bool)
        up[1:] = (r[:-1] <= cfg.rsi_os) & (r[1:] > cfg.rsi_os)
        dn[1:] = (r[:-1] >= cfg.rsi_ob) & (r[1:] < cfg.rsi_ob)
        ok = ~np.isnan(trend) & (adx15 >= cfg.adx_min)
        sig[up & ok & (trend > 0)] = 1
        sig[dn & ok & (trend < 0)] = -1
    else:
        raise ValueError(f"unknown strategy {strat!r}")
    return sig


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
def _simulate(df: pd.DataFrame, cfg: Config, build_curve: bool = True):
    """Run the engine over a df that ALREADY has indicators. Returns (trades, curve).
    Kept separate so the walk-forward research harness can simulate data slices.
    Uses numpy arrays + a precomputed signal vector for speed; set build_curve=False
    to skip the per-bar equity curve when only trades are needed (research)."""
    n = len(df)
    highs = df["high"].to_numpy(float); lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float); atrs = df["atr"].to_numpy(float)
    sig_arr = signals_array(df, cfg)
    idx = df.index
    day_ids = idx.normalize().asi8          # int per calendar day, for the daily reset
    msm, rr, fee = cfg.atr_stop_mult, cfg.rr, cfg.fee
    mdd, max_cl = cfg.max_daily_drawdown, cfg.max_consec_losses

    equity = peak = cfg.equity
    pos = None
    cur_day = None
    day_start_equity = equity
    halted_today = False
    consec_losses = 0
    trades, curve = [], []

    for i in range(n):
        price = closes[i]

        # new day -> reset the daily guards (matches the live loop)
        if day_ids[i] != cur_day:
            cur_day, day_start_equity, halted_today, consec_losses = day_ids[i], equity, False, 0

        # manage open position against this bar's high/low
        if pos is not None:
            if pos["side"] == 1:
                hit_stop, hit_tp = lows[i] <= pos["stop"], highs[i] >= pos["tp"]
            else:
                hit_stop, hit_tp = highs[i] >= pos["stop"], lows[i] <= pos["tp"]
            # if a bar straddles both, assume the stop fills first (conservative)
            exit_px = pos["stop"] if hit_stop else (pos["tp"] if hit_tp else None)
            if exit_px is not None:
                pnl = pos["side"] * (exit_px - pos["entry"]) * pos["qty"]
                pnl -= fee * pos["qty"] * (pos["entry"] + exit_px)
                equity += pnl
                consec_losses = consec_losses + 1 if pnl < 0 else 0
                trades.append({
                    "time": str(idx[i]), "side": pos["side"],
                    "entry": pos["entry"], "exit": exit_px, "qty": pos["qty"],
                    "pnl": pnl, "r": pnl / pos["risk"] if pos["risk"] else 0.0,
                })
                pos = None

        # daily drawdown lockout — the single most important guard in the file
        if equity <= day_start_equity * (1 - mdd):
            halted_today = True

        # look for entries only when flat, not halted, not on a losing streak
        if pos is None and not halted_today and consec_losses < max_cl:
            sig = int(sig_arr[i])
            a = atrs[i]
            if sig != 0 and not math.isnan(a):
                entry = price
                stop = entry - sig * msm * a
                tp = entry + sig * msm * a * rr
                qty = size_position(equity, entry, stop, cfg, sig)
                if qty > 0:
                    pos = {"side": sig, "entry": entry, "stop": stop, "tp": tp,
                           "qty": qty, "risk": abs(entry - stop) * qty}

        if build_curve:
            peak = max(peak, equity)
            curve.append({"time": str(idx[i]), "equity": equity, "drawdown": equity / peak - 1})

    return trades, curve


def backtest(df: pd.DataFrame, cfg: Config) -> dict:
    """Full backtest: add indicators, simulate, summarize."""
    trades, curve = _simulate(prepare_indicators(df, cfg), cfg)
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


def synthetic(n: int = 3000, seed: int = 7, freq: str = "15min") -> pd.DataFrame:
    """Offline demo data: GBM with regime shifts, so the engine runs anywhere.
    NOTE: this is random noise, NOT a real market. Use it to verify mechanics
    only — never to judge whether the strategy is profitable on a real symbol."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    drift = np.concatenate([rng.normal(m, 0, n // 3) for m in (0.0002, -0.0003, 0.0001)])
    if len(drift) < n:                       # n not divisible by 3 -> pad the tail
        drift = np.concatenate([drift, np.full(n - len(drift), 0.0001)])
    drift = drift[:n]
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

    # MTF needs enough fast bars to build the slow-timeframe trend
    htf_mult = (_tf_seconds(cfg.htf_rule.replace("min", "m")) // _tf_seconds(cfg.timeframe)
                if cfg.strategy == "mtf" else 1)
    fetch_limit = (cfg.ema_slow + 50) * max(htf_mult, 1)

    while True:
        try:
            df = prepare_indicators(fetch_ohlcv(cfg, limit=fetch_limit), cfg)
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
    p.add_argument("--strategy", default=None, choices=list(_STRATEGIES),
                   help="override the strategy in the config")
    p.add_argument("--live", action="store_true", help="REQUIRED to place real orders")
    p.add_argument("--key", default="")
    p.add_argument("--secret", default="")
    args = p.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.strategy:
        cfg = replace(cfg, strategy=args.strategy)

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
