import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from utils.data_generator import add_technical_indicators


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────
def _make_sequences(arr: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback: i])
        y.append(arr[i])
    return np.array(X), np.array(y)


def _simulate_lstm_weights(X_train, y_train, X_test, seed=7):
    """
    Simulate LSTM behaviour with stacked Ridge regressors operating on
    time-lagged features — pure sklearn, no deep-learning framework.
    """
    np.random.seed(seed)
    # Flatten sequences for sklearn
    n_train, lookback, n_feat = X_train.shape
    n_test  = X_test.shape[0]
    Xf_tr = X_train.reshape(n_train, -1)
    Xf_te = X_test.reshape(n_test,  -1)

    # "Hidden layer 1" via PCA + Ridge
    pca = PCA(n_components=min(40, Xf_tr.shape[1]), random_state=seed)
    H1_tr = pca.fit_transform(Xf_tr)
    H1_te = pca.transform(Xf_te)

    # "Hidden layer 2" — gradient boosting captures non-linearity
    gbr = GradientBoostingRegressor(n_estimators=120, max_depth=4,
                                     learning_rate=0.08, subsample=0.8,
                                     random_state=seed)
    gbr.fit(H1_tr, y_train)
    pred = gbr.predict(H1_te)
    return pred, gbr


# ──────────────────────────────────────────
# 1. LSTM-style Price Prediction
# ──────────────────────────────────────────
ML_FEATURES = ["close", "volume", "rsi", "macd", "bb_pct", "atr", "stoch_k", "obv"]

def run_lstm_prediction(df: pd.DataFrame, lookback: int = 10):
    df = add_technical_indicators(df.copy())
    df = df.dropna(subset=ML_FEATURES).reset_index(drop=True)
    # Need enough rows for sequences
    if len(df) < lookback + 20:
        lookback = max(3, len(df) // 5)
    features = ML_FEATURES
    feat_df  = df[features].copy()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feat_df)

    price_scaler = MinMaxScaler()
    price_scaled = price_scaler.fit_transform(df[["close"]].values)

    X, y = _make_sequences(scaled, lookback)
    y_price = price_scaler.fit_transform(df[["close"]].values[lookback:])

    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y_price[:split].ravel(), y_price[split:].ravel()

    pred_scaled, model = _simulate_lstm_weights(X_tr, y_tr, X_te)

    pred = price_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    actual = df["close"].values[lookback + split:]

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae  = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-9))) * 100
    r2   = 1 - np.sum((actual - pred)**2) / (np.sum((actual - np.mean(actual))**2) + 1e-9)

    # Align prediction series with dates
    pred_dates = df["date"].values[lookback + split:]

    return {
        "dates":     pred_dates,
        "actual":    actual,
        "predicted": pred,
        "metrics": {
            "RMSE":     f"${rmse:,.2f}",
            "MAE":      f"${mae:,.2f}",
            "MAPE":     f"{mape:.2f}%",
            "R² Score": f"{max(0, r2):.4f}",
            "Accuracy": f"{max(0, (1-mape/100)*100):.1f}%",
        },
        "rmse": rmse,
    }


# ──────────────────────────────────────────
# 2. ARIMA-style Forecast (manual AR model)
# ──────────────────────────────────────────
def run_arima_forecast(df: pd.DataFrame, forecast_days: int = 30, confidence: int = 95):
    close = df["close"].values.astype(float)
    n     = len(close)

    # Fit AR(p) via OLS
    p = 5
    X_ar, y_ar = [], []
    for i in range(p, n):
        X_ar.append(close[i - p: i])
        y_ar.append(close[i])
    X_ar = np.array(X_ar)
    y_ar = np.array(y_ar)

    model = Ridge(alpha=1.0)
    model.fit(X_ar, y_ar)

    # Residual std for confidence intervals
    fitted    = model.predict(X_ar)
    residuals = y_ar - fitted
    sigma     = np.std(residuals)

    # Multi-step forecast
    history = list(close[-p:])
    forecast = []
    for _ in range(forecast_days):
        x_in = np.array(history[-p:]).reshape(1, -1)
        nxt  = model.predict(x_in)[0]
        # Add small drift noise for realism
        nxt += np.random.normal(0, sigma * 0.1)
        forecast.append(nxt)
        history.append(nxt)

    forecast  = np.array(forecast)
    z_score   = norm.ppf(0.5 + confidence / 200)
    steps     = np.arange(1, forecast_days + 1)
    margin    = z_score * sigma * np.sqrt(steps)
    upper     = forecast + margin
    lower     = np.maximum(forecast - margin, forecast * 0.5)

    future_dates = pd.date_range(start=df["date"].iloc[-1] + timedelta(days=1),
                                  periods=forecast_days, freq="D")

    # Accuracy estimate via cross-validation
    split  = int(n * 0.8)
    X_cv   = X_ar[:split - p]; y_cv_tr = y_ar[:split - p]
    X_cv_t = X_ar[split - p:]; y_cv_te = y_ar[split - p:]
    model2 = Ridge(alpha=1.0).fit(X_cv, y_cv_tr)
    preds_cv = model2.predict(X_cv_t)
    acc = max(0, (1 - np.mean(np.abs((y_cv_te - preds_cv) / (y_cv_te + 1e-9)))) * 100)

    return {
        "forecast":      pd.Series(forecast, index=future_dates),
        "upper":         pd.Series(upper,    index=future_dates),
        "lower":         pd.Series(lower,    index=future_dates),
        "future_dates":  future_dates,
        "accuracy":      acc,
        "rmse":          np.sqrt(mean_squared_error(y_cv_te, preds_cv)),
    }


# ──────────────────────────────────────────
# 3. Random Forest Trading Signals
# ──────────────────────────────────────────
RF_FEATURES = ["rsi", "macd", "macd_hist", "bb_pct", "stoch_k", "stoch_d",
               "atr", "cci", "williams_r", "obv"]

def run_random_forest_signal(df: pd.DataFrame):
    df = add_technical_indicators(df.copy())
    df = df.dropna(subset=RF_FEATURES).reset_index(drop=True)

    feat_cols = RF_FEATURES

    # Label: future 5-bar return without NaN (manual shift)
    close = df["close"].values
    future_close = np.empty_like(close)
    future_close[:-5] = close[5:]
    future_close[-5:]  = close[-5:]          # repeat last for tail rows
    future_ret = (future_close - close) / (close + 1e-9)

    df["future_ret"] = future_ret
    df["label"] = np.where(future_ret > 0.02, 2,
                  np.where(future_ret < -0.02, 0, 1))

    X = df[feat_cols].values
    y = df["label"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    split  = max(1, int(len(X_sc) * 0.75))
    X_tr, y_tr = X_sc[:split], y[:split]

    rf = RandomForestClassifier(n_estimators=150, max_depth=8,
                                 min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    probs  = rf.predict_proba(X_sc)
    preds  = rf.predict(X_sc)

    label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    df["signal"]     = [label_map[p] for p in preds]
    df["confidence"] = probs.max(axis=1) * 100
    importances      = rf.feature_importances_

    return df, feat_cols, importances



# ──────────────────────────────────────────
# 4. Anomaly Detection (Isolation Forest)
# ──────────────────────────────────────────
ANOM_FEATURES = ["close", "volume", "rsi", "macd", "atr", "bb_pct",
                 "stoch_k", "cci", "williams_r", "obv"]

def run_anomaly_detection(df: pd.DataFrame):
    df = add_technical_indicators(df.copy())
    df = df.dropna(subset=ANOM_FEATURES).reset_index(drop=True)

    feat_cols = ANOM_FEATURES
    X = df[feat_cols].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    iso.fit(X_sc)
    df["anomaly_score"] = -iso.score_samples(X_sc)  # higher = more anomalous
    df["is_anomaly"]    = iso.predict(X_sc) == -1

    return df


# ──────────────────────────────────────────
# 5. Portfolio Optimizer (Mean-Variance)
# ──────────────────────────────────────────
def run_portfolio_optimizer(returns_df: pd.DataFrame, n_portfolios: int = 2000):
    """
    Monte Carlo Mean-Variance Optimization (Markowitz).
    returns_df: DataFrame of daily returns, columns = coin names.
    """
    np.random.seed(99)
    mu  = returns_df.mean().values * 252
    cov = returns_df.cov().values * 252
    n   = len(mu)

    port_returns = []
    port_vols    = []
    port_sharpes = []
    port_weights = []

    rf_rate = 0.05  # risk-free rate

    for _ in range(n_portfolios):
        w  = np.random.dirichlet(np.ones(n))
        r  = w @ mu
        v  = np.sqrt(w @ cov @ w)
        sr = (r - rf_rate) / (v + 1e-9)
        port_returns.append(r)
        port_vols.append(v)
        port_sharpes.append(sr)
        port_weights.append(w)

    port_returns = np.array(port_returns)
    port_vols    = np.array(port_vols)
    port_sharpes = np.array(port_sharpes)
    port_weights = np.array(port_weights)

    best_idx = port_sharpes.argmax()
    min_vol_idx = port_vols.argmin()

    return {
        "returns":    port_returns,
        "vols":       port_vols,
        "sharpes":    port_sharpes,
        "weights":    port_weights,
        "best_idx":   best_idx,
        "min_vol_idx":min_vol_idx,
        "best_weights":  port_weights[best_idx],
        "best_return":   port_returns[best_idx],
        "best_vol":      port_vols[best_idx],
        "best_sharpe":   port_sharpes[best_idx],
    }


# ──────────────────────────────────────────
# 6. Sentiment Simulation (rule-based NLP)
# ──────────────────────────────────────────
HEADLINES = {
    "BTC": [
        ("CoinDesk",   "Bitcoin breaks key resistance, analysts target $50K next",          0.82),
        ("Bloomberg",  "Institutional BTC inflows hit record highs this quarter",            0.78),
        ("Reuters",    "Bitcoin ETF sees $2B in weekly inflows amid market optimism",        0.75),
        ("CryptoNews", "SEC delays decision on Bitcoin spot ETF applications again",         0.35),
        ("Forbes",     "BTC mining difficulty hits all-time high, hash rate surges",         0.65),
        ("CNBC",       "Bitcoin tumbles 8% as macro headwinds dampen crypto sentiment",      0.22),
        ("CoinTelegraph","On-chain data suggests BTC accumulation phase underway",           0.71),
        ("The Block",  "Whale wallets add 15,000 BTC in 48 hours, signal bullish outlook",  0.80),
    ],
    "ETH": [
        ("CoinDesk",   "Ethereum staking yields rise to 5.2% post-Dencun upgrade",          0.76),
        ("Decrypt",    "ETH gas fees drop 80% following EIP-4844 implementation",            0.85),
        ("Bloomberg",  "Ethereum layer-2 ecosystem surpasses $40B in total value locked",    0.79),
        ("Reuters",    "Ethereum SEC classification remains uncertain, analysts warn",       0.32),
        ("The Block",  "ETH burn rate increases as DeFi activity picks up pace",             0.68),
        ("CryptoNews", "Ethereum price lags BTC rally, investors rotate to altcoins",        0.41),
    ],
    "SOL": [
        ("CoinTelegraph","Solana processes 65,000 TPS in stress test, beats all rivals",    0.88),
        ("Decrypt",    "SOL surges 12% as memecoin activity floods Solana network",          0.72),
        ("The Block",  "Solana DeFi TVL crosses $5B for the first time in 2024",            0.80),
        ("CryptoNews", "Network outage concerns resurface as Solana congestion spikes",     0.25),
    ],
}
DEFAULT_HEADLINES = [
    ("CoinDesk",    "Crypto market sentiment turns bullish as BTC holds support",          0.70),
    ("Bloomberg",   "Altcoin season signals emerge from technical indicators",              0.65),
    ("Reuters",     "Regulatory clarity concerns weigh on digital asset markets",          0.38),
    ("The Block",   "DeFi protocol exploits hit $200M in Q1, security focus grows",       0.28),
    ("Decrypt",     "Institutional adoption accelerates across crypto asset class",        0.73),
    ("CryptoNews",  "Stablecoin supply reaches new high, signals dry powder buildup",     0.67),
]

def run_sentiment_analysis(coin: str):
    np.random.seed(hash(coin) % 999)
    headlines = HEADLINES.get(coin, DEFAULT_HEADLINES)
    # Jitter scores slightly
    jittered = [(src, title, min(1.0, max(0.0, score + np.random.uniform(-0.05, 0.05))))
                for src, title, score in headlines]

    scores    = [s for _, _, s in jittered]
    positive  = sum(1 for s in scores if s > 0.6)
    neutral   = sum(1 for s in scores if 0.4 <= s <= 0.6)
    negative  = sum(1 for s in scores if s < 0.4)
    overall   = np.mean(scores) * 100

    return {
        "overall_score": overall,
        "positive": positive,
        "neutral":  neutral,
        "negative": negative,
        "headlines": [{"source": s, "title": t, "score": sc} for s, t, sc in jittered],
    }
