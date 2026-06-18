# edgebot

A risk-first trading bot. It only acts when the strategy sees an edge, and it
will not let one trade — or one day — do real damage. That second part is the
whole point.

## What it actually does
- **Strategy:** EMA(21/55) trend cross, gated by an RSI filter so it doesn't
  chase overbought/oversold extremes. ATR-based stops and a 1.8R take-profit.
- **Risk engine:** every position is sized so hitting the stop loses exactly
  1% of equity. Notional is hard-capped at 3x. If the account is down 5% on the
  day, the bot stops trading until tomorrow. No revenge logic exists in the code.
- **Three modes:** `backtest` (risk nothing), `paper` (live prices, fake fills),
  `live` (real orders — you must pass `--live` plus API keys).

Default mode is paper. You have to go out of your way to risk money.

## Setup
```bash
pip install ccxt pandas numpy
```

## Run
```bash
# prove the edge offline, no internet/keys needed
python3 bot.py backtest --offline

# backtest on real recent data from the exchange
python3 bot.py backtest --config config.json

# paper trade live prices (no money at risk)
python3 bot.py paper --config config.json

# go live (only when paper has earned your trust)
python3 bot.py live --config config.json --live --key XXX --secret YYY
```

## The honest part
The demo backtest returns roughly +7% with a sub-4% max drawdown. That is what a
*working* bot looks like — modest, survivable. It is not a $300/day machine,
because that machine does not exist. What this gives you is the only thing that
compounds: an edge you can measure, executed with discipline you don't have to
summon in the moment.

Before live: edit `config.json`, backtest your params on real data, then paper
trade for weeks. If paper isn't green, live won't be either — and you'll have
learned that for free instead of for $1,000.

## Tuning
Everything lives in `config.json`. Sane things to change first: `symbol`,
`timeframe`, `risk_per_trade` (leave at 0.01 while learning), `max_leverage`
(leave low), `max_daily_drawdown`. Don't raise risk to chase a number — that's
the failure mode the whole bot is built to prevent.
