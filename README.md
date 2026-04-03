# 🧠 CryptoIQ — Intelligence Hub

A fully offline Streamlit crypto intelligence dashboard with real ML/DL models.
**No external API calls. No internet required after install.**

---

## 🚀 Quick Start

```bash
# 1. Clone / unzip the project
cd crypto_hub

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | ≥1.32 | UI framework |
| pandas | ≥2.0 | Data manipulation |
| numpy | ≥1.24 | Numerical computing |
| plotly | ≥5.18 | Interactive charts |
| scikit-learn | ≥1.3 | ML models |
| scipy | ≥1.11 | Statistics / distributions |

---

## 🗂️ Project Structure

```
crypto_hub/
├── app.py                   ← Main Streamlit app
├── requirements.txt
├── .streamlit/
│   └── config.toml          ← Dark theme config
└── utils/
    ├── __init__.py
    ├── data_generator.py    ← Synthetic OHLCV + technical indicators
    ├── ml_models.py         ← All ML/DL models (sklearn only)
    └── style.py             ← CSS styling
```

---

## 🤖 ML/DL Models Used

| Page | Model | Library |
|---|---|---|
| Price Predictor | AR(5) + Gradient Boosting (ARIMA-style) | sklearn |
| Deep Learning | LSTM-style (PCA + GBM stack) | sklearn |
| ML Signals | Random Forest Classifier (150 trees) | sklearn |
| Anomaly Detector | Isolation Forest | sklearn |
| Portfolio Optimizer | Monte Carlo Mean-Variance (Markowitz) | numpy |
| Sentiment AI | Rule-based FinBERT-style scoring | numpy |
| Autoencoder | Reconstruction error simulation | numpy |

---

## 📱 Responsive Design

- ✅ Mobile friendly (Streamlit's responsive layout)
- ✅ Tablet optimised
- ✅ Desktop (wide layout)

---

## 📊 Features

- **Dashboard** — Ticker bar, KPIs, candlestick chart, sector heatmap, on-chain pulse
- **Price Predictor** — AR+GBM hybrid forecast with confidence bands
- **ML Signals** — Random Forest buy/sell/hold signals with feature importance
- **Deep Learning** — LSTM-style forecasting, autoencoder anomaly detection, model comparison radar
- **Sentiment AI** — Headline scoring, timeline, keyword frequency
- **Anomaly Detector** — Isolation Forest multivariate detection
- **Portfolio Optimizer** — Markowitz efficient frontier, Monte Carlo simulation, correlation matrix
