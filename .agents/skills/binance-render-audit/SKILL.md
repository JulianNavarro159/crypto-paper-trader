---
name: binance-render-audit
description: >-
  Audit live Binance Spot account balances, trade executions, ntfy push alerts,
  and Render cloud logs for crypto-paper-trader without triggering HTTP 418 bans,
  and deploy verified fixes to GitHub and Render.
---

# Binance Spot & Render Live Audit Workflow

Use this skill whenever the user asks to inspect recent bot performance, check live Binance Spot assets, diagnose losing or missed trades, or deploy optimizations to Render.

## 1. Audit ntfy Mobile Alerts
Fetch the recent alert history from the configured `ntfy` topic:
```powershell
python -c "import urllib.request, json; req = urllib.request.urlopen('https://ntfy.sh/julian_binance_spot_a94f72c1/json?poll=1&since=24h', timeout=10); [print(l) for l in req.read().decode('utf-8').splitlines() if l.strip()]"
```

## 2. Audit Live Binance Spot Account via WebSocket API v3
Query real balances and `myTrades` over `wss://ws-api.binance.com:443/ws-api/v3` using `LiveBinanceSpotClient._request_via_ws_api("account.status", {})` and `myTrades` to avoid REST HTTP 418 rate-limit bans.

## 3. Verify Code Quality & Zero Inline Comments
Run the test suite, linter, and token check before committing:
```powershell
$env:PYTHONPATH="C:\Users\julia\.gemini\antigravity\scratch\crypto_paper_trader\src"
python -m pytest C:\Users\julia\.gemini\antigravity\scratch\crypto_paper_trader\tests -v
python -m flake8 C:\Users\julia\.gemini\antigravity\scratch\crypto_paper_trader\src C:\Users\julia\.gemini\antigravity\scratch\crypto_paper_trader\tests --max-line-length=88
python -c "import pathlib, tokenize; files = list(pathlib.Path(r'C:\Users\julia\.gemini\antigravity\scratch\crypto_paper_trader').rglob('*.py')); comments = [f'{p}:{t.start[0]}' for p in files for t in tokenize.generate_tokens(open(p, encoding='utf-8').readline) if t.type == tokenize.COMMENT]; assert not comments, comments"
```

## 4. Non-Interactive Git Push & Render Verification
Push with terminal prompts disabled so PowerShell never blocks:
```powershell
$env:GIT_TERMINAL_PROMPT="0"
git push origin main
```
Then verify deployment status on Render (`https://dashboard.render.com/web/srv-dbengcdg1s2s7385gg2g/logs`).
