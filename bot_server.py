"""
Forex Signal Bot — Always-On Server (for Render.com)
Runs continuously, listens for /start or /signal on Telegram in
real time, and replies with the top 3 ranked signals from a scan
of major pairs.

NOT FINANCIAL ADVICE — a rule-based technical alert tool only.
"""

import os
import time
import threading
import logging
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from flask import Flask

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("forex_signal_bot")

WAT_OFFSET = timedelta(hours=1)  # Nigeria is UTC+1

FLAG_MAP = {
    "USD": "🇺🇸", "EUR": "🇪🇺", "GBP": "🇬🇧", "JPY": "🇯🇵",
    "CHF": "🇨🇭", "AUD": "🇦🇺", "CAD": "🇨🇦", "NZD": "🇳🇿",
}

SCAN_UNIVERSE = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "USD/CAD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
    "AUD/JPY", "EUR/AUD",
]

TWELVE_DATA_API_KEY = os.environ["TWELVE_DATA_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
INTERVAL = os.environ.get("INTERVAL", "5min")

_LAST_CALL_TIME = [0.0]
MIN_CALL_INTERVAL = 8  # seconds between API calls — keeps us under 8/min free tier


def _rate_limit():
    elapsed = time.time() - _LAST_CALL_TIME[0]
    if elapsed < MIN_CALL_INTERVAL:
        time.sleep(MIN_CALL_INTERVAL - elapsed)
    _LAST_CALL_TIME[0] = time.time()


def pair_flags(pair):
    base, quote = pair.split("/")
    return f"{FLAG_MAP.get(base, '')} {base}/{quote} {FLAG_MAP.get(quote, '')}"


def fetch_candles(pair, interval, output_size=100):
    _rate_limit()
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": pair,
        "interval": interval,
        "outputsize": output_size,
        "apikey": TWELVE_DATA_API_KEY,
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


def add_indicators(df, ema_fast=5, ema_slow=13, rsi_period=14):
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
    return df


def current_trend(df, ema_fast=5, ema_slow=13, rsi_period=14):
    if df is None or len(df) < max(ema_slow, rsi_period) + 2:
        return None, None
    df = add_indicators(df, ema_fast, ema_slow, rsi_period)
    last = df.iloc[-1]
    signal = "BUY" if last["ema_fast"] > last["ema_slow"] else "SELL"
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


def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    resp = requests.post(url, data=payload, timeout=15)
    if not resp.ok:
        log.error(f"Telegram send failed: {resp.text}")


def format_alert(pair, interval, signal, candle, forecast_price=None, confidence=None):
    now_wat = datetime.now(timezone.utc) + WAT_OFFSET
    entry_ts = now_wat.strftime("%I:%M %p") + " WAT"
    direction_box = "🟢" if signal == "BUY" else "🔴"
    conf_line = f"🎯 Confidence: {confidence}%\n" if confidence is not None else ""
    return (
        f"🔔 *SIGNAL* — {pair_flags(pair)}\n"
        f"⏳ Timer: {interval}\n"
        f"➡️ Entry: {entry_ts}\n"
        f"📈 Direction: {signal} {direction_box}\n"
        f"{conf_line}\n"
        f"_Rule-based technical alert, not financial advice._"
    )


def scan_top_signals(top_n=3):
    candidates = []
    for pair in SCAN_UNIVERSE:
        try:
            df = fetch_candles(pair, INTERVAL)
            signal, candle = current_trend(df)
            if signal is None:
                continue
            confirmed, forecast_price, confidence = confirm_with_arima(df, signal)
            if confirmed and confidence is not None:
                candidates.append((confidence, pair, signal, candle, forecast_price))
        except Exception as e:
            log.error(f"Scan failed for {pair}: {e}")

    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[:top_n]


def run_scan_and_reply():
    send_telegram_message("🔍 Scanning the market for the top signals now...")
    top = scan_top_signals(top_n=3)
    if not top:
        send_telegram_message("No strong signals found right now. Try again shortly.")
        return
    for confidence, pair, signal, candle, forecast_price in top:
        msg = format_alert(pair, INTERVAL, signal, candle, forecast_price, confidence)
        send_telegram_message(msg)

offset = 0
    log.info("Clearing any backlog of old messages on startup")
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        resp = requests.get(url, params={"timeout": 0}, timeout=15)
        results = resp.json().get("result", [])
        for r in results:
            offset = max(offset, r["update_id"])
        log.info(f"Backlog cleared, starting fresh from update_id {offset}")
    except Exception as e:
        log.error(f"Failed to clear backlog: {e}")

    log.info("Starting Telegram long-poll loop")

        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            params = {"offset": offset + 1, "timeout": 30}
            resp = requests.get(url, params=params, timeout=35)
            if not resp.ok:
                log.error(f"getUpdates failed: {resp.text}")
                time.sleep(5)
                continue
            results = resp.json().get("result", [])
            for r in results:
                offset = max(offset, r["update_id"])
                text = r.get("message", {}).get("text", "")
                if text.strip().lower() in ("/start", "/signal"):
                    log.info("Command received, running scan")
                    run_scan_and_reply()
        except Exception as e:
            log.error(f"Poll loop error: {e}")
            time.sleep(5)


app = Flask(__name__)


@app.route("/")
def health():
    return "Forex signal bot is running."


if __name__ == "__main__":
    poll_thread = threading.Thread(target=telegram_poll_loop, daemon=True)
    poll_thread.start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
