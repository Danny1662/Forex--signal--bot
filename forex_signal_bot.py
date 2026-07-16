"""
Forex Scalping Signal Bot — single-run version for GitHub Actions.
Crossover signals are confirmed with a lightweight ARIMA forecast
before alerting, to filter out likely-noise signals.

NOT FINANCIAL ADVICE — a rule-based technical alert tool only.
"""

import json
import os
import logging
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("forex_signal_bot")

STATE_PATH = "state.json"


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def fetch_candles(pair, interval, api_key, output_size=100):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": pair,
        "interval": interval,
        "outputsize": output_size,
        "apikey": api_key,
        "order": "ASC",
    }
    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()
    if "values" not in data:
        log.warning(f"No data for {pair}: {data.get('message', data)}")
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df.sort_values("datetime").reset_index(drop=True)


def compute_signal(df, ema_fast=5, ema_slow=13, rsi_period=14):
    if df is None or len(df) < max(ema_slow, rsi_period) + 2:
        return None, None
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    prev, last = df.iloc[-2], df.iloc[-1]
    crossed_up = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
    crossed_down = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]

    signal = None
    if crossed_up:
        signal = "BUY"
    elif crossed_down:
        signal = "SELL"
    return signal, last


def confirm_with_arima(df, signal, steps=3, lookback=100):
    """Fit a quick ARIMA model on recent closes and check the forecast
    direction agrees with the crossover signal. Returns (confirmed, forecast_price)."""
    closes = df["close"].values[-lookback:]
    try:
        model = ARIMA(closes, order=(1, 1, 1))
        fit = model.fit()
        forecast = fit.forecast(steps=steps)
        forecast_price = float(forecast[-1])
        direction = forecast_price - closes[-1]

        if signal == "BUY" and direction > 0:
            return True, forecast_price
        elif signal == "SELL" and direction < 0:
            return True, forecast_price
        else:
            return False, forecast_price
    except Exception as e:
        log.warning(f"ARIMA fit failed, skipping confirmation: {e}")
        return False, None


def send_telegram_message(bot_token, chat_id, text):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    resp = requests.post(url, data=payload, timeout=15)
    if not resp.ok:
        log.error(f"Telegram send failed: {resp.text}")


def format_alert(pair, interval, signal, candle, forecast_price=None):
    local_time = datetime.now(timezone.utc) + timedelta(hours=1)
    ts = local_time.strftime("%I:%M %p") + " WAT"
    forecast_line = ""
    if forecast_price is not None:
        forecast_line = f"ARIMA forecast: {forecast_price:.5f}\n"
    return (
        f"📈 *{signal} {pair}*\n"
        f"Entry time: {ts}\n"
        f"Timeframe: {interval}\n"
        f"Price: {candle['close']:.5f}\n"
        f"RSI(14): {candle['rsi']:.1f}\n"
        f"{forecast_line}\n"
        f"_Rule-based technical alert, not financial advice._"
    )


def main():
    pairs = json.loads(os.environ["PAIRS_JSON"])
    interval = os.environ["INTERVAL"]
    api_key = os.environ["TWELVE_DATA_API_KEY"]
    bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    state = load_state()

    for pair in pairs:
        try:
            df = fetch_candles(pair, interval, api_key)
            signal, candle = compute_signal(df)
            if signal is None:
                log.info(f"{pair}: no crossover signal")
                continue

            candle_ts = str(candle["datetime"])
            if state.get(pair) == candle_ts:
                log.info(f"{pair}: signal already alerted for this candle")
                continue

            confirmed, forecast_price = confirm_with_arima(df, signal)
            if not confirmed:
                log.info(f"{pair}: {signal} crossover not confirmed by ARIMA forecast, skipping")
                state[pair] = candle_ts
                continue

            msg = format_alert(pair, interval, signal, candle, forecast_price)
            send_telegram_message(bot_token, chat_id, msg)
            log.info(f"Sent confirmed {signal} for {pair}")
            state[pair] = candle_ts

        except Exception as e:
            log.error(f"Error processing {pair}: {e}")

    save_state(state)


if __name__ == "__main__":
    main()
