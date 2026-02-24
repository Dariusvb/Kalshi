# Kalshi Risk-Managed Bot (MVP)

A Render-deployable Python bot skeleton for Kalshi that:
- Authenticates using Kalshi API key + RSA private key
- Scans markets
- Scores opportunities with a transparent rule-based signal engine
- Applies risk controls
- Simulates orders with `DRY_RUN=true`
- Sends Discord notifications
- Stores decisions in SQLite

## Important
- No profitability guarantees.
- Start in `demo` + `DRY_RUN=true`.
- Do not hardcode secrets in code or GitHub.

## Env Vars (exact names)
- `DISCORD_WEBHOOK_URL`
- `DRY_RUN`
- `KALSHI_API_KEY_ID`
- `KALSHI_ENV` (`demo` or `prod`)
- `KALSHI_PRIVATE_KEY_PEM`
- `LOG_LEVEL`

Optional:
- `SCAN_INTERVAL_SECONDS`
- `MIN_TRADE_DOLLARS`
- `MAX_TRADE_DOLLARS`
- `MIN_CONVICTION_SCORE`
- `MAX_SPREAD_CENTS`
- `MIN_RECENT_VOLUME`
- `SOCIAL_SIGNAL_ENABLED`
- `SOCIAL_MAX_BONUS_SCORE`
- `SOCIAL_SIZE_BOOST_MAX_PCT`
- `DAILY_MAX_LOSS_DOLLARS`
- `MAX_OPEN_EXPOSURE_DOLLARS`
- `MAX_SIMULTANEOUS_POSITIONS`
- `WITHDRAWAL_ALERT_THRESHOLD_DOLLARS`

## Local run
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/test_kalshi_auth.py
