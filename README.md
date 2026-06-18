# edgebot

A risk-first crypto **futures** bot. It only acts when the strategy sees an
edge, and it will not let one trade — or one day — do real damage. That second
part is the whole point.

## The deal: 3:1 setups, hard-capped risk

When a valid signal appears, the bot risks a **fixed, hard-capped dollar
amount** (default **$100**) and targets **3x that — $300** — by placing the
take-profit at 3R. Every loss is bounded to the risk amount. There is **no daily
profit quota**, on purpose: forcing a fixed daily P&L makes a bot trade when
there's no edge, and that's how accounts die.

So a realistic day looks like one of:
- a winning setup → **+$300-ish**,
- no clean setup → **$0, it sits out**,
- a loser → **a capped ~$100 loss**.

Whether that nets out positive over time depends entirely on the strategy's
**edge**, which you *prove on real data* before risking a cent. A bot that
reliably makes $300 *every* day does not exist; this one is built to give you
bounded downside and 3:1 upside, and to tell you the truth about how often it
actually pays.

## What it actually does
- **Strategy:** EMA(21/55) trend cross, gated by an RSI filter so it doesn't
  chase overbought/oversold extremes. ATR-based stops, take-profit at `rr`×risk
  (default 3R).
- **Risk engine (`risk_model: "fixed_risk"`):** every position is sized so
  hitting the stop loses ~`risk_dollars` ($100). Notional is hard-capped at
  `max_leverage`× equity, and any trade whose stop would sit *beyond* the
  liquidation price is **skipped**. If the account is down `max_daily_drawdown`
  (6%) on the day, or after `max_consec_losses` (4) losses in a row, the bot
  stands down. No revenge logic exists in the code.
- **Three modes:** `backtest` (risk nothing), `paper` (live prices, fake fills),
  `live` (real orders — you must pass `--live` plus API keys).

Default mode is paper. You have to go out of your way to risk money.

## Setup
```bash
pip install ccxt pandas numpy
```

## Run
```bash
# verify mechanics offline on synthetic data (NOT a real market — proves nothing
# about profitability, only that the engine runs)
python3 bot.py backtest --offline

# backtest on REAL recent data pulled from the exchange (needs network + reachable exchange)
python3 bot.py backtest --config config.json

# backtest on REAL data from a local CSV — no network needed.
# CSV columns: time/ts/timestamp/date, open, high, low, close, volume
python3 bot.py backtest --config config.json --csv sol_15m.csv

# paper trade live prices (no money at risk)
python3 bot.py paper --config config.json

# go live (only when paper has earned your trust)
python3 bot.py live --config config.json --live --key XXX --secret YYY
```

## Reading the backtest output
The summary answers your real question directly:
- `expectancy_R` / `expectancy_usd` — average result per trade. **Positive = the
  strategy has edge on this data; negative = it does not.** This is the number
  that matters most.
- `target_usd_per_day`, `days_hit_target`, `pct_days_hit_target` — how often a
  *day* actually cleared the $300 target. Look here before believing any
  daily-income fantasy.
- `avg_day_usd`, `best_day_usd`, `worst_day_usd`, `green_days`/`red_days` — the
  honest daily distribution.
- `max_drawdown_pct`, `max_consec_losses` — the pain you'd have had to sit
  through to collect the return.

## The honest part
The offline demo runs on random noise, not a real market, so its P&L means
nothing about SOL or any other symbol — it only proves the engine works. To know
whether this is profitable on **SOL 15m**, backtest **real** SOL candles
(`--config config.json` against the exchange, or `--csv` with a downloaded file)
across months, then paper trade for weeks. If paper isn't green, live won't be
either — and you'll have learned that for free instead of for $1,000.

## Tuning
Everything lives in `config.json`. Sane things to change first: `symbol`,
`timeframe`, `risk_dollars` (your hard per-trade loss cap), `rr` (reward:risk;
3.0 = target 3× the risk), `max_leverage` (leave modest), `max_daily_drawdown`.
Don't raise risk to chase a number — that's the failure mode the whole bot is
built to prevent.

### Legacy fixed-notional model
Setting `risk_model: "margin"` with `margin`/`leverage` restores the old
"notional = margin × leverage" sizing. It is **not** recommended: it ignores the
stop distance, so your real per-trade loss is whatever the market hands you, not
a number you chose. `fixed_risk` is the default for a reason.
