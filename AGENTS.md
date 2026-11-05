# Crypto Paper & Live Trader Workspace Rules

## 1. Fee-Covered Net Profit Invariants
- Round-trip Spot taker fees on Binance without BNB discount equal `0.20%` (`0.10%` entry + `0.10%` exit).
- Never configure a profit-lock floor (`breakeven_floor`) or signal-reversal exit threshold (`min_signal_exit_profit_ratio`) below the round-trip fee threshold when locking positive gains. Any locked profit exit must guarantee positive net PnL in USDT after deducting both entry and exit commissions.
- Maintain a multi-tier protective trailing stop in `PaperTradingEngine.evaluate_protective_exits`:
  - Cut downside risk early once unrealized gain reaches `+0.14%`.
  - Lock guaranteed net profit (`>= +0.21%` gross floor with `0.02%` hair-trigger trail) once unrealized gain reaches `+0.22%`.
  - Execute partial take-profit (`TP1`) at `+0.32%` and trail the remaining runner tightly.

## 2. Anti-Overbought & Tick-Precision Entry Guardrails
- Always enforce `buy_mask = raw_buy_mask & (~sell_mask)` in `enrich_with_indicators` so a BUY signal can never fire on a candle where any SELL or overbought condition is active.
- Never allow trend or breakout entries when `RSI > 55.0` or when price is already extended at the upper Bollinger Band (`bb_upper`).
- Cap initial stop-loss distance in `build_market_signal_from_row` at a maximum of `0.65%` of `reference_price` so no single stop-out can erase multiple micro-scalping wins.
- Exclude ultra-low-priced tokens with coarse tick steps (`> 0.10%` per minimum price tick, such as `PEPEUSDT`) from `UniverseConfig.volatile_symbols`.

## 3. Binance WebSocket API v3 & Deployment Guardrails
- Always prioritize Binance WebSocket API v3 (`wss://ws-api.binance.com:443/ws-api/v3`) for signed account queries (`account.status` with empty additional parameters `{}`) and order execution to prevent REST IP `418` bans on shared cloud IPs.
- Before committing or deploying, verify zero `#` comments across all `.py` files using Python's `tokenize` module and pass `pytest` and `flake8 --max-line-length=88`.
- Always set `$env:GIT_TERMINAL_PROMPT="0"` before running `git push` on Windows PowerShell so interactive credential helpers never block background deployment tasks.
