"""
Forex Scalping Signal Bot — v3
Adds: confidence %, entry/exit time (WAT), on-demand /start or /signal
command support (checked each scheduled run, so up to ~5 min delay).

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
            return Fa
