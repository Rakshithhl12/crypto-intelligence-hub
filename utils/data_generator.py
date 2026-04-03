import numpy as np
import pandas as pd
from datetime import datetime, timedelta

COINS = {
    "BTC": {"base": 43000, "vol": 0.035, "color": "#f7931a"},
    "ETH": {"base": 2280,  "vol": 0.042, "color": "#627eea"},
    "SOL": {"base": 98,    "vol": 0.065, "color": "#9945ff"},
    "BNB": {"base": 312,   "vol": 0.030, "color": "#f3ba2f"},
    "XRP": {"base": 0.52,  "vol": 0.055, "color": "#00aae4"},
    "ADA": {"base": 0.48,  "vol": 0.060, "color": "#0033ad"},
    "AVAX":{"base": 35,    "vol": 0.070, "color": "#e84142"},
    "DOT": {"base": 7.1,   "vol": 0.055, "color": "#e6007a"},
}

TIMEFRAME_DAYS = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365}


def generate_price_history(coin: str, timeframe: str = "3M", seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed + hash(coin) % 1000)
    days = TIMEFRAME_DAYS.get(timeframe, 90)
    info = COINS.get(coin, COINS["BTC"])
    base = info["base"]
    vol  = info["vol"]

    n = days * 24  # hourly
    dates = pd.date_range(end=datetime.now(), periods=n, freq="h")

    # GBM with mean-reversion + trend
    returns = np.random.normal(0.0001, vol / np.sqrt(24), n)
    # Add small trend
    trend = np.linspace(0, 0.15, n)
    returns += trend / n
    prices = base * np.exp(np.cumsum(returns))

    # Build OHLCV (resample hourly to daily for candles)
    df = pd.DataFrame({"date": dates, "price": prices})
    df = df.set_index("date").resample("D").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
    ).reset_index()
    df["volume"] = np.random.lognormal(np.log(base * 1e4), 0.4, len(df))
    df["coin"] = coin
    return df.dropna().reset_index(drop=True)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].values
    n = len(close)

    # RSI
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=14).mean().values
    avg_l = pd.Series(loss).ewm(span=14).mean().values
    rs = np.where(avg_l == 0, 100, avg_g / (avg_l + 1e-9))
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = pd.Series(df["macd"]).ewm(span=9).mean().values
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    ma20 = pd.Series(close).rolling(20).mean().values
    std20 = pd.Series(close).rolling(20).std().values
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    df["bb_mid"]   = ma20
    df["bb_pct"]   = (close - (ma20 - 2*std20)) / (4 * std20 + 1e-9)

    # Moving averages
    df["ma7"]  = pd.Series(close).rolling(7).mean().values
    df["ma20"] = ma20
    df["ma50"] = pd.Series(close).rolling(50).mean().values
    df["ma200"]= pd.Series(close).rolling(200).mean().values

    # ATR
    high = df["high"].values
    low  = df["low"].values
    tr   = np.maximum(high - low,
           np.maximum(np.abs(high - np.roll(close, 1)),
                      np.abs(low  - np.roll(close, 1))))
    df["atr"] = pd.Series(tr).rolling(14).mean().values

    # Stochastic
    low14  = pd.Series(low).rolling(14).min().values
    high14 = pd.Series(high).rolling(14).max().values
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df["stoch_d"] = pd.Series(df["stoch_k"]).rolling(3).mean().values

    # OBV
    direction = np.sign(np.diff(close, prepend=close[0]))
    df["obv"]  = np.cumsum(direction * df["volume"].values)

    # CCI
    tp  = (high + low + close) / 3
    ma_tp  = pd.Series(tp).rolling(20).mean().values
    mad_tp = pd.Series(tp).rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean()))).values
    df["cci"] = (tp - ma_tp) / (0.015 * mad_tp + 1e-9)

    # Williams %R
    df["williams_r"] = -100 * (high14 - close) / (high14 - low14 + 1e-9)

    return df


def get_all_coins_snapshot() -> pd.DataFrame:
    rows = []
    changes_24h = {"BTC": 2.4, "ETH": -1.1, "SOL": 5.8, "BNB": 0.9,
                   "XRP": -0.3, "ADA": 1.7, "AVAX": -2.8, "DOT": 3.1}
    mcaps = {"BTC": 845e9, "ETH": 274e9, "SOL": 42e9, "BNB": 48e9,
             "XRP": 28e9,  "ADA": 17e9,  "AVAX": 13e9, "DOT": 9e9}
    for coin, info in COINS.items():
        rows.append({
            "coin": coin,
            "price": info["base"] * (1 + np.random.uniform(-0.005, 0.005)),
            "change_24h": changes_24h[coin],
            "market_cap": mcaps[coin],
            "volume_24h": info["base"] * np.random.uniform(8000, 25000),
            "color": info["color"],
        })
    return pd.DataFrame(rows)

