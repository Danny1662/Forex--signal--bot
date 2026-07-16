"""
Forex Scalping Signal Bot — v4
Confidence %, entry time, Nigerian time (WAT), on-demand /start or
/signal command support, and a clean flag-style alert format.

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
WAT_OFFSET = timedelta(hours=1)  # Nigeria is UTC+1

FLAG_MAP = {
    "USD": "🇺🇸", "EUR": "🇪🇺", "GBP": "🇬🇧", "JPY": "🇯🇵",
    "CHF": "🇨🇭", "AUD": "🇦🇺", "CAD": "🇨🇦", "NZD": "🇳🇿",
}


def pair_flags(pair):
    base, quote = pair.split("/")
    return f"{FLAG_MAP.get(base, '')} {base}/{quote} {FLAG_MAP.get(quote, '')}"


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
    closes = df["close"].values[-lookback:]
    try:
        model = ARIMA(closes, order=(1, 1, 1))
        fit = model.fit()
        forecast = fit.forecast(steps=steps)
        forecast_price = float(forecast[-1])
        direction = forecast_price - closes[-1]

        recent_std = pd.Series(closes[-20:]).diff().std()
        if recent_std == 0 or pd.isna(recent_std):
            recent_std = 1e-6
        move_strength = abs(direction) / recent_std
        confidence = max(5, min(95, round(move_strength * 25)))

        if signal == "BUY" and direction > 0:
            return True, forecast_price, confidence
        elif signal == "SELL" and direction < 0:
            return True, forecast_price, confidence
        else:
            return False, forecast_price, confidence
    except Exception as e:
        log.warning(f"ARIMA fit failed, skipping confirmation: {e}")
        return False, None, None


def send_telegram_message(bot_token, chat_id, text):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    resp = requests.post(url, data=payload, timeout=15)
    if not resp.ok:
        log.error(f"Telegram send failed: {resp.text}")


def format_alert(pair, interval, signal, candle, forecast_price=None, confidence=None):
    now_wat = datetime.now(timezone.utc) + WAT_OFFSET
    entry_ts = now_wat.strftime("%I:%M %p") + " WAT"

    direction_box = "🟢" if signal == "BUY" else "🔴"
    conf_line = f"🎯 Confidence: {confidence}%\n" if confidence is not None else ""

    return (
        f"🔔 *NEW SIGNAL!*\n\n"
        f"📊 Trade: {pair_flags(pair)}\n"
        f"⏳ Timer: {interval}\n"
        f"➡️ Entry: {entry_ts}\n"
        f"📈 Direction: {signal} {direction_box}\n"
        f"{conf_line}\n"
        f"_Rule-based technical alert, not financial advice._"
    )


def check_pair(pair, interval, api_key):
    df = fetch_candles(pair, interval, api_key)
    signal, candle = compute_signal(df)
    if signal is None:
        return None, None
    confirmed, forecast_price, confidence = confirm_with_arima(df, signal)
    if not confirmed:
        return signal, None
    msg = format_alert(pair, interval, signal, candle, forecast_price, confidence)
    return signal, msg


def get_telegram_updates(bot_token, last_update_id):
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    params = {"offset": last_update_id + 1, "timeout": 0}
    resp = requests.get(url, params=params, timeout=15)
    if not resp.ok:
        log.error(f"getUpdates failed: {resp.text}")
        return [], last_update_id
    results = resp.json().get("result", [])
    new_last_id = last_update_id
    for r in results:
        new_last_id = max(new_last_id, r["update_id"])
    return results, new_last_id


def handle_on_demand(pairs, interval, api_key, bot_token, chat_id, state):
    last_update_id = state.get("_last_update_id", 0)
    updates, new_last_id = get_telegram_updates(bot_token, last_update_id)
    log.info(f"Checked for Telegram commands: got {len(updates)} update(s), last_update_id was {last_update_id}, now {new_last_id}")
    for u in updates:
        log.info(f"Update content: {u}")
    state["_last_update_id"] = new_last_id

    triggered = False
    for u in updates:
        text = u.get("message", {}).get("text", "")
        if text.strip().lower() in ("/start", "/signal"):
            triggered = True

    if not triggered:
        log.info("No /start or /signal command found, skipping on-demand check")
        return

    log.info("Command detected! Running on-demand check now")
    send_telegram_message(bot_token, chat_id, "🔍 Checking current signals now...")
    for pair in pairs:
        try:
            signal, msg = check_pair(pair, interval, api_key)
            if msg:
                send_telegram_message(bot_token, chat_id, msg)
            elif signal:
                send_telegram_message(bot_token, chat_id, f"{pair}: {signal} crossover seen but not confirmed by forecast — skipping.")
            else:
                send_telegram_message(bot_token, chat_id, f"{pair}: no crossover signal right now.")
        except Exception as e:
            log.error(f"On-demand check failed for {pair}: {e}")


def main():
    pairs = json.loads(os.environ["PAIRS_JSON"])
    interval = os.environ["INTERVAL"]
    api_key = os.environ["TWELVE_DATA_API_KEY"]
    bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    state = load_state()

    handle_on_demand(pairs, interval, api_key, bot_token, chat_id, state)

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

            confirmed, forecast_price, confidence = confirm_with_arima(df, signal)
            if not confirmed:
                log.info(f"{pair}: {signal} crossover not confirmed by ARIMA, skipping")
                state[pair] = candle_ts
                continue

            msg = format_alert(pair, interval, signal, candle, forecast_price, confidence)
            send_telegram_message(bot_token, chat_id, msg)
            log.info(f"Sent confirmed {signal} for {pair}")
            state[pair] = candle_ts

        except Exception as e:
            log.error(f"Error processing {pair}: {e}")

    save_state(state)


if __name__ == "__main__":
    main()
