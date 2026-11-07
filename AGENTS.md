# Crypto Paper & Live Trader Workspace Rules

## 1. Fee-Covered Net Profit & Zero-Below-Entry Spot Invariants
- Round-trip Spot taker fees on Binance without BNB discount equal `0.20%` (`0.10%` entry + `0.10%` exit).
- In production Spot execution (`RiskConfig.spot_no_loss_micro_profit()`, `no_loss_spot_mode=True`):
  - **Never sell below purchase price:** `stop_loss_price` (`Piso de seguridad`) must strictly satisfy `stop_loss_price >= entry_price` and cover round-trip commissions plus `min_target_net_profit_usdt` (`0.01 USDT`). Temporary 1x Spot drawdowns are held safely in Spot (zero leverage, zero liquidation risk) rather than realizing losses below `entry_price`.
  - **Immediate micro-profit capture (`>= +0.01 USDT` net):** Execute a full `100%` position exit as soon as price clears round-trip fees, `+0.01 USDT` net gain, and the live bid-ask spread cushion (`compute_no_loss_exit_levels`).
  - **Preserve BNB fee reserve:** Keep `BNBUSDT` out of `UniverseConfig` so the account's `BNB` balance remains dedicated to `0.075%` fee discounts without `0.001 BNB` lot-size truncation.
- When `no_loss_spot_mode=False` (legacy/backtest stop-loss mode), maintain the multi-tier protective trailing stop in `PaperTradingEngine.evaluate_protective_exits` so locked profit exits always exceed round-trip fees.

## 2. Anti-Overbought & Tick-Precision Entry Guardrails
- Always enforce `buy_mask = raw_buy_mask & (~sell_mask)` in `enrich_with_indicators` so a BUY signal can never fire on a candle where any SELL or overbought condition is active.
- Never allow trend or breakout entries when `RSI > 55.0` or when price is already extended at the upper Bollinger Band (`bb_upper`).
- Cap initial stop-loss distance in `build_market_signal_from_row` at a maximum of `0.65%` of `reference_price` so no single stop-out can erase multiple micro-scalping wins.
- Exclude ultra-low-priced tokens with coarse tick steps (`> 0.10%` per minimum price tick, such as `PEPEUSDT`) from `UniverseConfig.volatile_symbols`.

## 3. Binance WebSocket API v3 & Deployment Guardrails
- Always prioritize Binance WebSocket API v3 (`wss://ws-api.binance.com:443/ws-api/v3`) for signed account queries (`account.status` with empty additional parameters `{}`) and order execution to prevent REST IP `418` bans on shared cloud IPs.
- Before committing or deploying, verify zero `#` comments across all `.py` files using Python's `tokenize` module and pass `pytest` and `flake8 --max-line-length=88`.
- Always set `$env:GIT_TERMINAL_PROMPT="0"` before running `git push` on Windows PowerShell so interactive credential helpers never block background deployment tasks.
