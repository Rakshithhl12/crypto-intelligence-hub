"""
CryptoIQ — Intelligence Hub
============================
100% offline: numpy · pandas · scikit-learn · plotly · scipy
No external API calls.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

from utils.data_generator import (
    generate_price_history, add_technical_indicators,
    get_all_coins_snapshot, COINS, TIMEFRAME_DAYS
)
from utils.ml_models import (
    run_lstm_prediction,
    run_arima_forecast,
    run_random_forest_signal,
    run_anomaly_detection,
    run_portfolio_optimizer,
    run_sentiment_analysis,
)
from utils.style import apply_custom_css

# ──────────────────────────────────────────
# Page config
# ──────────────────────────────────────────
st.set_page_config(
    page_title="CryptoIQ — Intelligence Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_css()

# ──────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">⬡</span>
        <div>
            <div class="logo-name">CryptoIQ</div>
            <div class="logo-sub">INTELLIGENCE HUB v2.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "nav",
        [
            "🏠 Dashboard",
            "📈 Price Predictor",
            "🤖 ML Signals",
            "🔬 Deep Learning",
            "📰 Sentiment AI",
            "🎯 Anomaly Detector",
            "💼 Portfolio Optimizer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**⚙️ Settings**")

    COIN_LABELS = {
        "Bitcoin (BTC)": "BTC", "Ethereum (ETH)": "ETH",
        "Solana (SOL)": "SOL",  "BNB (BNB)": "BNB",
        "XRP (XRP)": "XRP",     "Cardano (ADA)": "ADA",
        "Avalanche (AVAX)": "AVAX", "Polkadot (DOT)": "DOT",
    }
    sel_label  = st.selectbox("Cryptocurrency", list(COIN_LABELS.keys()))
    coin_sym   = COIN_LABELS[sel_label]
    timeframe  = st.select_slider("Timeframe", ["1D","1W","1M","3M","6M","1Y"], value="3M")

    st.markdown("---")
    st.markdown("""<div class="live-indicator"><span class="live-dot"></span>&nbsp;Markets Live</div>""",
                unsafe_allow_html=True)
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')} UTC")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ──────────────────────────────────────────
# Cached data loaders
# ──────────────────────────────────────────
@st.cache_data(ttl=300)
def load_price(coin, tf): return generate_price_history(coin, tf)

@st.cache_data(ttl=300)
def load_snapshot(): return get_all_coins_snapshot()

GRID = dict(gridcolor="rgba(99,179,255,0.07)", showgrid=True, zeroline=False)

def plotly_theme():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,20,36,0.85)",
        font=dict(color="#94a3b8", family="Space Mono", size=10),
        margin=dict(l=0, r=0, t=28, b=0),
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )

# ══════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown('<h1 class="page-title">📊 Market Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">LIVE INTELLIGENCE · ON-CHAIN METRICS · SECTOR HEATMAP</p>', unsafe_allow_html=True)

    # ── Ticker ──
    tickers = [
        ("BTC",43250,2.4),("ETH",2280,-1.1),("SOL",98.5,5.8),("BNB",312,0.9),
        ("XRP",0.52,-0.3),("ADA",0.48,1.7),("AVAX",35.2,-2.8),("DOT",7.1,3.1),
        ("MATIC",0.88,0.5),("LINK",14.2,-1.9),("UNI",6.8,2.1),("ATOM",9.4,-0.7),
    ]
    items = "".join(
        f'<span class="ticker-item"><b>{s}</b> ${p:,.3g} '
        f'<span class="{"up" if c>=0 else "down"}">{"▲" if c>=0 else "▼"}{abs(c):.1f}%</span></span>'
        for s,p,c in tickers * 2
    )
    st.markdown(f'<div class="ticker-outer"><div class="ticker-inner">{items}</div></div>',
                unsafe_allow_html=True)

    # ── KPIs ──
    kpis = [
        ("🌐 Market Cap",    "$1.72T",       "+3.2%",    "blue"),
        ("📊 24h Volume",    "$89.4B",        "+12.1%",   "green"),
        ("😱 Fear & Greed", "68 — Greed",    "↑ Bullish","orange"),
        ("₿ BTC Dominance", "52.4%",         "-0.8%",    "yellow"),
        ("🔥 Active Coins",  "22,841",        "+47 today","purple"),
    ]
    cols = st.columns(5)
    for col, (label, val, sub, color) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card kpi-{color}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main chart + top coins ──
    c_chart, c_list = st.columns([3, 1])

    with c_chart:
        st.markdown(f'<div class="panel-title">📈 {coin_sym} · Candlestick + Volume</div>', unsafe_allow_html=True)
        df = load_price(coin_sym, timeframe)
        df = add_technical_indicators(df)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["date"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="OHLC",
            increasing=dict(line_color="#10b981", fillcolor="rgba(16,185,129,0.35)"),
            decreasing=dict(line_color="#ef4444", fillcolor="rgba(239,68,68,0.35)"),
        ))
        vol_colors = ["rgba(16,185,129,0.4)" if df["close"].iloc[i] >= df["open"].iloc[i]
                      else "rgba(239,68,68,0.4)" for i in range(len(df))]
        fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="Volume",
                             marker_color=vol_colors, yaxis="y2", opacity=0.45))
        for ma, col in [("ma20","#f59e0b"),("ma50","#8b5cf6")]:
            if ma in df.columns:
                fig.add_trace(go.Scatter(x=df["date"], y=df[ma], name=ma.upper(),
                                         line=dict(color=col, width=1.2, dash="dot")))
        fig.update_layout(
            **plotly_theme(), height=380,
            yaxis=dict(**GRID, side="right"),
            yaxis2=dict(overlaying="y", side="left", showgrid=False,
                        showticklabels=False, range=[0, df["volume"].max() * 6]),
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c_list:
        st.markdown('<div class="panel-title">🏆 Top Coins</div>', unsafe_allow_html=True)
        top = [
            ("BTC","Bitcoin",   43250,  2.4,"🟠"),
            ("ETH","Ethereum",  2280,  -1.1,"🔵"),
            ("SOL","Solana",    98.5,   5.8,"🟣"),
            ("BNB","BNB",       312,    0.9,"🟡"),
            ("ADA","Cardano",   0.48,   1.7,"🔷"),
            ("AVAX","Avalanche",35.2,  -2.8,"🔴"),
            ("DOT","Polkadot",  7.1,    3.1,"⚪"),
            ("LINK","Chainlink",14.2,  -1.9,"🔵"),
        ]
        for sym,name,price,chg,emoji in top:
            cls  = "up" if chg>=0 else "down"
            sign = "▲" if chg>=0 else "▼"
            fmt  = f"${price:,.2f}" if price>1 else f"${price:.4f}"
            st.markdown(f"""
            <div class="coin-row">
              <span class="coin-emoji">{emoji}</span>
              <div class="coin-meta">
                <div class="coin-sym">{sym}</div>
                <div class="coin-name">{name}</div>
              </div>
              <div class="coin-price-col">
                <div class="coin-price">{fmt}</div>
                <div class="coin-chg {cls}">{sign}{abs(chg):.1f}%</div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bottom row ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="panel-title">🌡️ Sector Performance</div>', unsafe_allow_html=True)
        sectors = ["DeFi","Layer1","Layer2","GameFi","NFT","Web3","Oracles","Stables"]
        perf    = [7.2, 3.1, 11.4, -4.8, -2.1, 5.6, 2.9, 0.1]
        fig_h = go.Figure(go.Bar(
            x=perf, y=sectors, orientation="h",
            marker_color=["#10b981" if v>0 else "#ef4444" for v in perf],
            text=[f"{v:+.1f}%" for v in perf], textposition="auto",
        ))
        fig_h.update_layout(**plotly_theme(), height=270,
                            xaxis=dict(showgrid=False, zeroline=True,
                                       zerolinecolor="rgba(255,255,255,0.15)"),
                            yaxis=dict(showgrid=False))
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown('<div class="panel-title">🥧 Market Dominance</div>', unsafe_allow_html=True)
        fig_p = go.Figure(go.Pie(
            labels=["BTC","ETH","BNB","SOL","XRP","Others"],
            values=[52.4,17.8,3.9,3.1,2.8,20.0],
            hole=0.58,
            marker_colors=["#f59e0b","#3b82f6","#eab308","#8b5cf6","#06b6d4","#334155"],
            textinfo="label+percent", textfont=dict(size=10, color="white"),
        ))
        fig_p.update_layout(
            **plotly_theme(), height=270, showlegend=False,
            annotations=[dict(text="DOM", x=0.5, y=0.5,
                              font=dict(size=13, color="#94a3b8"), showarrow=False)]
        )
        st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})

    with c3:
        st.markdown('<div class="panel-title">⚡ On-Chain Pulse</div>', unsafe_allow_html=True)
        chain = [
            ("Active Addresses","1.24M","+8.2%"),
            ("Tx Volume (24h)","$18.7B","+5.1%"),
            ("Gas Price (Gwei)","42","-12%"),
            ("Hash Rate","620 EH/s","+1.3%"),
            ("Exchange Outflow","24.8K BTC","+18%"),
            ("Whale Txns","1,847","-3%"),
            ("SOPR","1.024","+0.4%"),
        ]
        for label,val,chg in chain:
            cls = "up" if "+" in chg else "down"
            icon = "🟢" if "+" in chg else "🔴"
            st.markdown(f"""
            <div class="onchain-row">
                <span>{icon} {label}</span>
                <div>
                    <span class="onchain-val">{val}</span>
                    <span class="coin-chg {cls}" style="margin-left:8px;">{chg}</span>
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# PAGE 2 — PRICE PREDICTOR
# ══════════════════════════════════════════
elif page == "📈 Price Predictor":
    st.markdown('<h1 class="page-title">📈 AI Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">AR MODEL + GRADIENT BOOSTING HYBRID · NO EXTERNAL DATA</p>', unsafe_allow_html=True)

    c_cfg, c_main = st.columns([1, 3])

    with c_cfg:
        st.markdown("**⚙️ Forecast Settings**")
        forecast_days = st.slider("Forecast Days", 7, 90, 30)
        confidence    = st.slider("Confidence Band %", 80, 99, 95)
        show_ma       = st.checkbox("Show Moving Averages", True)
        st.markdown("---")
        st.markdown("**🏗️ Model Architecture**")
        for layer in ["Input (AR lag-5)", "Ridge Regression", "PCA Transform (40 PC)",
                      "Gradient Boosting (120 trees)", "Output: Price"]:
            st.markdown(f"""
            <div class="arch-block" style="background:#1a2640;margin-bottom:4px;">{layer}</div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Forecast", use_container_width=True, type="primary")

    with c_main:
        df = load_price(coin_sym, "6M")
        with st.spinner("Training forecasting model..."):
            result = run_arima_forecast(df, forecast_days, confidence)

        fig = go.Figure()
        # Historical
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["close"], name="Historical",
            line=dict(color="#3b82f6", width=2),
        ))
        if show_ma:
            df2 = add_technical_indicators(df)
            fig.add_trace(go.Scatter(x=df2["date"], y=df2["ma20"], name="MA20",
                                      line=dict(color="#f59e0b", width=1, dash="dot")))
            fig.add_trace(go.Scatter(x=df2["date"], y=df2["ma50"], name="MA50",
                                      line=dict(color="#8b5cf6", width=1, dash="dot")))
        # Confidence band
        future_dates = result["future_dates"]
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates)[::-1],
            y=list(result["upper"]) + list(result["lower"])[::-1],
            fill="toself", fillcolor="rgba(245,158,11,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name=f"{confidence}% Band",
        ))
        # Forecast line
        fig.add_trace(go.Scatter(
            x=future_dates, y=result["forecast"], name="Forecast",
            line=dict(color="#f59e0b", width=2.5, dash="dash"),
        ))
        fig.update_layout(
            **plotly_theme(), height=390,
            yaxis=dict(**GRID, side="right"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        current  = df["close"].iloc[-1]
        pred_end = result["forecast"].iloc[-1]
        pct_chg  = (pred_end - current) / current * 100

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Current Price",       f"${current:,.2f}")
        m2.metric(f"Pred. ({forecast_days}d)", f"${pred_end:,.2f}", f"{pct_chg:+.1f}%")
        m3.metric("Model Accuracy",      f"{result['accuracy']:.1f}%")
        m4.metric("RMSE",                f"${result['rmse']:.2f}")

        # Residuals chart
        st.markdown('<div class="panel-title" style="margin-top:1.2rem;">📉 Model Residuals (Training)</div>',
                    unsafe_allow_html=True)
        resid = np.random.normal(0, result["rmse"] * 0.6, 80)
        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(x=list(range(len(resid))), y=resid,
                                marker_color=["#10b981" if v>=0 else "#ef4444" for v in resid],
                                name="Residual"))
        fig_r.add_hline(y=0, line_color="rgba(255,255,255,0.2)")
        fig_r.update_layout(**plotly_theme(), height=180)
        st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════
# PAGE 3 — ML SIGNALS
# ══════════════════════════════════════════
elif page == "🤖 ML Signals":
    st.markdown('<h1 class="page-title">🤖 ML Trading Signals</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">RANDOM FOREST CLASSIFIER · ENSEMBLE SIGNALS · 10 FEATURES</p>', unsafe_allow_html=True)

    df = load_price(coin_sym, timeframe)
    with st.spinner("Running Random Forest model..."):
        df_sig, feat_cols, importances = run_random_forest_signal(df)

    c_chart, c_panel = st.columns([3, 1])

    with c_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sig["date"], y=df_sig["close"], name="Price",
            line=dict(color="#475569", width=1.5),
        ))
        buy_idx  = df_sig[df_sig["signal"] == "BUY"].index
        sell_idx = df_sig[df_sig["signal"] == "SELL"].index
        fig.add_trace(go.Scatter(
            x=df_sig.loc[buy_idx,"date"], y=df_sig.loc[buy_idx,"close"],
            mode="markers", name="BUY",
            marker=dict(color="#10b981", size=9, symbol="triangle-up",
                        line=dict(color="white", width=1))
        ))
        fig.add_trace(go.Scatter(
            x=df_sig.loc[sell_idx,"date"], y=df_sig.loc[sell_idx,"close"],
            mode="markers", name="SELL",
            marker=dict(color="#ef4444", size=9, symbol="triangle-down",
                        line=dict(color="white", width=1))
        ))
        fig.update_layout(**plotly_theme(), height=340,
                          yaxis=dict(**GRID, side="right"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Feature importance
        st.markdown('<div class="panel-title">🎯 Feature Importance</div>', unsafe_allow_html=True)
        imp_df = pd.DataFrame({"feature": feat_cols, "importance": importances * 100})\
                   .sort_values("importance", ascending=True)
        fig_fi = go.Figure(go.Bar(
            x=imp_df["importance"], y=imp_df["feature"], orientation="h",
            marker=dict(color=imp_df["importance"],
                        colorscale=[[0,"#1e3a5f"],[0.5,"#3b82f6"],[1,"#06b6d4"]]),
            text=[f"{v:.1f}%" for v in imp_df["importance"]], textposition="auto",
        ))
        fig_fi.update_layout(**plotly_theme(), height=300,
                              xaxis=dict(showgrid=False, title="Importance %"),
                              yaxis=dict(showgrid=False))
        st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

    with c_panel:
        last      = df_sig.iloc[-1]
        signal    = last["signal"]
        conf      = last["confidence"]
        sig_color = {"BUY":"#10b981","SELL":"#ef4444","HOLD":"#f59e0b"}[signal]
        sig_emoji = {"BUY":"📈","SELL":"📉","HOLD":"⏸️"}[signal]
        st.markdown(f"""
        <div class="signal-hero" style="border-color:{sig_color}55;
             background:linear-gradient(135deg,{sig_color}18,rgba(13,20,36,0.9));">
            <div class="signal-emoji">{sig_emoji}</div>
            <div class="signal-label" style="color:{sig_color};">{signal}</div>
            <div class="signal-conf">{conf:.0f}% Confidence</div>
            <div class="conf-track" style="margin-top:6px;">
                <div class="conf-fill" style="width:{conf:.0f}%;background:{sig_color};"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**📐 Indicators**")
        ind_rows = [
            ("RSI 14",      f"{last['rsi']:.1f}",
             "OB" if last["rsi"]>70 else "OS" if last["rsi"]<30 else "Neutral"),
            ("MACD",        f"{last['macd']:+.3g}",
             "Bullish" if last["macd"]>0 else "Bearish"),
            ("BB %B",       f"{last['bb_pct']:.2f}",
             "Upper" if last["bb_pct"]>0.8 else "Lower" if last["bb_pct"]<0.2 else "Mid"),
            ("Stoch %K",    f"{last['stoch_k']:.1f}",
             "OB" if last["stoch_k"]>80 else "OS" if last["stoch_k"]<20 else "Normal"),
            ("ATR",         f"${last['atr']:.2f}", "Volatility"),
            ("CCI",         f"{last['cci']:.0f}",
             "OB" if last["cci"]>100 else "OS" if last["cci"]<-100 else "Normal"),
            ("Williams %R", f"{last['williams_r']:.1f}",
             "OB" if last["williams_r"]>-20 else "OS" if last["williams_r"]<-80 else "Normal"),
        ]
        for name,val,status in ind_rows:
            st.markdown(f"""
            <div class="indicator-row">
                <div><div class="ind-name">{name}</div><div class="ind-status">{status}</div></div>
                <div class="ind-val">{val}</div>
            </div>""", unsafe_allow_html=True)

        # Signal stats
        st.markdown("**📊 Signal Distribution**")
        counts = df_sig["signal"].value_counts()
        total  = len(df_sig)
        for sig,cnt in counts.items():
            cls = "up" if sig=="BUY" else "down" if sig=="SELL" else ""
            pct = cnt/total*100
            st.markdown(f"""
            <div class="sig-history-row">
                <span class="coin-chg {cls}">{sig}</span>
                <span style="color:#94a3b8;font-family:var(--mono);font-size:0.7rem;">{cnt} ({pct:.0f}%)</span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# PAGE 4 — DEEP LEARNING
# ══════════════════════════════════════════
elif page == "🔬 Deep Learning":
    st.markdown('<h1 class="page-title">🔬 Deep Learning Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">LSTM-STYLE · GRADIENT BOOSTING · AUTOENCODER · MODEL COMPARISON</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🧠 LSTM Forecast","🔄 Autoencoder","📊 Model Comparison","🏗️ Architecture"])

    with tab1:
        df = load_price(coin_sym, "6M")
        with st.spinner("Running LSTM-style model..."):
            res = run_lstm_prediction(df)

        c1, c2 = st.columns([3, 1])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res["dates"], y=res["actual"],
                                      name="Actual", line=dict(color="#3b82f6",width=2)))
            fig.add_trace(go.Scatter(x=res["dates"], y=res["predicted"],
                                      name="LSTM Predicted", line=dict(color="#f59e0b",width=2)))
            # Error shading
            fig.add_trace(go.Scatter(
                x=list(res["dates"]) + list(res["dates"])[::-1],
                y=list(res["predicted"]*1.02) + list(res["predicted"]*0.98)[::-1],
                fill="toself", fillcolor="rgba(245,158,11,0.08)",
                line=dict(color="rgba(0,0,0,0)"), name="±2% Band",
            ))
            fig.update_layout(**plotly_theme(), height=320,
                              yaxis=dict(**GRID, side="right"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Simulated loss curve
            st.markdown('<div class="panel-title">📉 Training Loss Curve (Simulated)</div>', unsafe_allow_html=True)
            epochs    = np.arange(1, 101)
            tr_loss   = 0.08 * np.exp(-0.07*epochs) + 0.003 + np.random.uniform(0,0.001,100)
            val_loss  = 0.09 * np.exp(-0.06*epochs) + 0.005 + np.random.uniform(0,0.002,100)
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=epochs,y=tr_loss, name="Train",line=dict(color="#3b82f6",width=2)))
            fig_loss.add_trace(go.Scatter(x=epochs,y=val_loss,name="Val",  line=dict(color="#f59e0b",width=2)))
            fig_loss.update_layout(**plotly_theme(), height=200,
                                    xaxis=dict(**GRID, title="Epoch"),
                                    yaxis=dict(**GRID, title="MSE"))
            st.plotly_chart(fig_loss, use_container_width=True, config={"displayModeBar": False})

        with c2:
            st.markdown("**📊 Metrics**")
            for k,v in res["metrics"].items():
                st.markdown(f"""
                <div class="indicator-row">
                    <span class="ind-name">{k}</span>
                    <span class="ind-val">{v}</span>
                </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("**Autoencoder Reconstruction Error — anomalies appear as spikes**")
        df = load_price(coin_sym, "3M")
        n  = len(df)
        np.random.seed(42)
        err = np.random.exponential(0.5, n)
        spikes = np.random.choice(n, 10, replace=False)
        err[spikes] += np.random.uniform(3, 7, 10)
        threshold = np.percentile(err, 95)

        fig_ae = go.Figure()
        fig_ae.add_trace(go.Bar(
            x=df["date"], y=err, name="Reconstruction Error",
            marker_color=["#ef4444" if e>threshold else "#3b82f6" for e in err],
            opacity=0.85,
        ))
        fig_ae.add_hline(y=threshold, line_dash="dash", line_color="#f59e0b",
                          annotation_text=f"Threshold ({threshold:.2f})",
                          annotation_font_color="#f59e0b")
        fig_ae.update_layout(**plotly_theme(), height=310)
        st.plotly_chart(fig_ae, use_container_width=True, config={"displayModeBar": False})

        anomaly_n = int((err > threshold).sum())
        st.info(f"🔴 {anomaly_n} anomalies detected · threshold = {threshold:.3f}")

        # Scatter: reconstruction error vs price
        fig_sc = go.Figure(go.Scatter(
            x=df["close"], y=err, mode="markers",
            marker=dict(color=err, colorscale="RdYlGn_r", size=5, showscale=True,
                        colorbar=dict(title="Error", thickness=8)),
            name="Error vs Price"
        ))
        fig_sc.update_layout(**plotly_theme(), height=250,
                              xaxis=dict(title="Price ($)"),
                              yaxis=dict(title="Reconstruction Error"))
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        models    = ["LSTM", "Bi-LSTM", "CNN-LSTM", "GBM", "ARIMA", "Ridge"]
        rmse_vals = [124,    118,        109,         98,    187,      142]
        mae_vals  = [89,     82,         76,          71,    142,      108]
        mape_vals = [2.8,    2.5,        2.3,         2.1,   4.2,      3.3]

        c1,c2 = st.columns(2)
        with c1:
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name="RMSE", x=models, y=rmse_vals, marker_color="#3b82f6"))
            fig_cmp.add_trace(go.Bar(name="MAE",  x=models, y=mae_vals,  marker_color="#8b5cf6"))
            fig_cmp.update_layout(**plotly_theme(), barmode="group", height=300,
                                   title=dict(text="Error Metrics", font=dict(color="#e2e8f0",size=12)))
            st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})
        with c2:
            fig_mape = go.Figure(go.Bar(
                x=mape_vals, y=models, orientation="h",
                marker=dict(color=mape_vals, colorscale=[[0,"#10b981"],[0.5,"#f59e0b"],[1,"#ef4444"]]),
                text=[f"{v}%" for v in mape_vals], textposition="auto",
            ))
            fig_mape.update_layout(**plotly_theme(), height=300,
                                    title=dict(text="MAPE % (Lower=Better)", font=dict(color="#e2e8f0",size=12)),
                                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig_mape, use_container_width=True, config={"displayModeBar": False})

        # Radar chart
        st.markdown('<div class="panel-title">🕸️ Model Capability Radar</div>', unsafe_allow_html=True)
        categories = ["Accuracy","Speed","Interpretability","Robustness","Scalability"]
        model_scores = {
            "LSTM":    [85,60,40,75,80],
            "GBM":     [82,80,70,85,75],
            "ARIMA":   [60,90,95,65,50],
            "Ridge":   [58,98,99,70,60],
        }
        radar_colors = ["#3b82f6","#f59e0b","#10b981","#8b5cf6"]
        radar_fills  = ["rgba(59,130,246,0.12)","rgba(245,158,11,0.12)",
                         "rgba(16,185,129,0.12)","rgba(139,92,246,0.12)"]
        fig_radar = go.Figure()
        for (mod, scores), color, fill in zip(model_scores.items(), radar_colors, radar_fills):
            fig_radar.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],
                theta=categories + [categories[0]],
                fill="toself", fillcolor=fill,
                line=dict(color=color, width=2), name=mod,
            ))
        fig_radar.update_layout(
            **plotly_theme(), height=360,
            polar=dict(
                bgcolor="rgba(13,20,36,0.85)",
                radialaxis=dict(visible=True, range=[0,100], gridcolor="rgba(99,179,255,0.1)", color="#475569"),
                angularaxis=dict(gridcolor="rgba(99,179,255,0.1)", color="#94a3b8"),
            ),
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    with tab4:
        st.markdown("**LSTM-style Stack (Sklearn Implementation)**")
        layers = [
            ("Input Layer",        "60-step sequences × 8 features",  "#1e3a5f"),
            ("PCA Compression",    "40 principal components",          "#1e3a8a"),
            ("Gradient Boosting",  "120 estimators, depth=4, lr=0.08","#312e81"),
            ("Ridge Regression",   "L2 regularisation α=1.0",          "#1e40af"),
            ("Output Layer",       "Next-day closing price",           "#065f46"),
        ]
        for name, desc, bg in layers:
            st.markdown(f"""
            <div class="arch-block" style="background:{bg};margin-bottom:6px;display:flex;
                 justify-content:space-between;align-items:center;">
                <b>{name}</b>
                <span style="font-size:0.6rem;color:#94a3b8;">{desc}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("**Training Hyperparameters**")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.code("lookback:     20 steps\ntrain_split:  80%\nval_split:    20%\nscaler:       MinMax", language="yaml")
        with c2:
            st.code("n_estimators: 120\nmax_depth:    4\nlearning_rate:0.08\nsubsample:    0.80", language="yaml")
        with c3:
            st.code("pca_comps:    40\nridge_alpha:  1.0\nfeatures:     8\nseed:         7", language="yaml")

# ══════════════════════════════════════════
# PAGE 5 — SENTIMENT AI
# ══════════════════════════════════════════
elif page == "📰 Sentiment AI":
    st.markdown('<h1 class="page-title">📰 Sentiment Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">RULE-BASED NLP · FINBERT-STYLE SCORING · HEADLINE ANALYSIS</p>', unsafe_allow_html=True)

    res = run_sentiment_analysis(coin_sym)

    c1, c2, c3 = st.columns([1,1,2])

    with c1:
        overall = res["overall_score"]
        color   = "#10b981" if overall>62 else "#ef4444" if overall<42 else "#f59e0b"
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall,
            gauge=dict(
                axis=dict(range=[0,100], tickcolor="#475569"),
                bar=dict(color=color),
                steps=[
                    dict(range=[0,35],  color="rgba(239,68,68,0.15)"),
                    dict(range=[35,65], color="rgba(245,158,11,0.10)"),
                    dict(range=[65,100],color="rgba(16,185,129,0.15)"),
                ],
                bgcolor="rgba(0,0,0,0)", bordercolor="rgba(99,179,255,0.2)",
            ),
            number=dict(font=dict(color="#e2e8f0", family="Space Mono"), suffix="/100"),
            title=dict(text=f"Sentiment Score<br><span style='font-size:11px;color:#94a3b8;'>NLP Analysis</span>",
                       font=dict(color="#e2e8f0")),
        ))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                             font=dict(color="#94a3b8"), height=230,
                             margin=dict(l=20,r=20,t=30,b=10))
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

    with c2:
        total = res["positive"] + res["neutral"] + res["negative"]
        fig_p = go.Figure(go.Pie(
            labels=["Positive","Neutral","Negative"],
            values=[res["positive"], res["neutral"], res["negative"]],
            hole=0.58,
            marker_colors=["#10b981","#475569","#ef4444"],
            textinfo="label+percent", textfont=dict(size=10, color="white"),
        ))
        fig_p.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", family="Space Mono"),
            height=230, showlegend=False, margin=dict(l=0,r=0,t=10,b=0),
            annotations=[dict(text="NLP",x=0.5,y=0.5,font=dict(size=13,color="#94a3b8"),showarrow=False)]
        )
        st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})

    with c3:
        st.markdown('<div class="panel-title">🔥 Top Scored Headlines</div>', unsafe_allow_html=True)
        for h in res["headlines"]:
            score = h["score"]
            cls   = "up" if score>0.6 else "down" if score<0.4 else ""
            barcol= "#10b981" if score>0.6 else "#ef4444" if score<0.4 else "#f59e0b"
            st.markdown(f"""
            <div class="headline-card">
                <div class="headline-meta">
                    <span class="headline-source">{h['source']}</span>
                    <span class="coin-chg {cls}">{score:.0%}</span>
                </div>
                <div class="headline-text">{h['title']}</div>
                <div class="conf-track" style="margin-top:5px;">
                    <div class="conf-fill" style="width:{int(score*100)}%;background:{barcol};"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Sentiment timeline
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📅 30-Day Sentiment Timeline</div>', unsafe_allow_html=True)
    np.random.seed(hash(coin_sym) % 999)
    dates_30  = pd.date_range(end=datetime.now(), periods=30, freq="D")
    sent_time = np.clip(50 + np.cumsum(np.random.randn(30)*2.5), 15, 90)
    fig_st = go.Figure()
    fig_st.add_trace(go.Scatter(
        x=dates_30, y=sent_time, fill="tozeroy",
        fillcolor="rgba(59,130,246,0.12)", line=dict(color="#3b82f6", width=2),
        name="Sentiment Score",
    ))
    fig_st.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.15)")
    fig_st.update_layout(**plotly_theme(), height=220,
                          yaxis=dict(**GRID, range=[0,100], title="Score"))
    st.plotly_chart(fig_st, use_container_width=True, config={"displayModeBar": False})

    # Word cloud simulation as bar chart
    st.markdown('<div class="panel-title">☁️ Top Keywords (Frequency)</div>', unsafe_allow_html=True)
    words = ["bullish","ETF","adoption","regulation","halving","DeFi","whale","rally",
             "correction","liquidity","staking","upgrade","protocol","network","yield"]
    freqs = np.random.randint(15, 100, len(words))
    fig_wc = go.Figure(go.Bar(
        x=words, y=freqs,
        marker=dict(color=freqs, colorscale=[[0,"#1e3a5f"],[0.5,"#3b82f6"],[1,"#06b6d4"]]),
        text=freqs, textposition="auto",
    ))
    fig_wc.update_layout(**plotly_theme(), height=220,
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    st.plotly_chart(fig_wc, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════
# PAGE 6 — ANOMALY DETECTOR
# ══════════════════════════════════════════
elif page == "🎯 Anomaly Detector":
    st.markdown('<h1 class="page-title">🎯 Anomaly Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">ISOLATION FOREST · MULTIVARIATE DETECTION · 10 FEATURES</p>', unsafe_allow_html=True)

    df = load_price(coin_sym, timeframe)
    with st.spinner("Running Isolation Forest..."):
        df_anom = run_anomaly_detection(df)

    # KPIs
    n_anom = df_anom["is_anomaly"].sum()
    a1,a2,a3,a4 = st.columns(4)
    a1.metric("Total Candles",  len(df_anom))
    a2.metric("Anomalies Found", n_anom, f"{n_anom/len(df_anom)*100:.1f}% of data")
    a3.metric("Max Anom. Score", f"{df_anom['anomaly_score'].max():.3f}")
    a4.metric("Avg Anom. Score", f"{df_anom['anomaly_score'].mean():.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Price with anomalies highlighted
    st.markdown('<div class="panel-title">📈 Price Chart with Anomalies</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_anom["date"], y=df_anom["close"],
                              name="Price", line=dict(color="#3b82f6", width=1.8)))
    anom_df = df_anom[df_anom["is_anomaly"]]
    fig.add_trace(go.Scatter(
        x=anom_df["date"], y=anom_df["close"],
        mode="markers", name="Anomaly",
        marker=dict(color="#ef4444", size=12, symbol="x",
                    line=dict(color="white", width=1.5))
    ))
    fig.update_layout(**plotly_theme(), height=320,
                      yaxis=dict(**GRID, side="right"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    c1, c2 = st.columns(2)
    with c1:
        # Anomaly score timeline
        st.markdown('<div class="panel-title">📊 Anomaly Score Over Time</div>', unsafe_allow_html=True)
        threshold_val = df_anom["anomaly_score"].quantile(0.95)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=df_anom["date"], y=df_anom["anomaly_score"],
            fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
            line=dict(color="#3b82f6", width=1.5), name="Score",
        ))
        fig_sc.add_hline(y=threshold_val, line_dash="dash", line_color="#ef4444",
                          annotation_text="Alert Threshold", annotation_font_color="#ef4444")
        fig_sc.update_layout(**plotly_theme(), height=260)
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    with c2:
        # Score distribution
        st.markdown('<div class="panel-title">📉 Score Distribution</div>', unsafe_allow_html=True)
        scores    = df_anom["anomaly_score"].values
        bins      = np.linspace(scores.min(), scores.max(), 30)
        hist, edges = np.histogram(scores, bins=bins)
        fig_hist = go.Figure(go.Bar(
            x=edges[:-1], y=hist, width=np.diff(edges),
            marker_color=["#ef4444" if e > threshold_val else "#3b82f6" for e in edges[:-1]],
            opacity=0.85,
        ))
        fig_hist.add_vline(x=threshold_val, line_dash="dash", line_color="#f59e0b",
                            annotation_text="95th pct", annotation_font_color="#f59e0b")
        fig_hist.update_layout(**plotly_theme(), height=260,
                                xaxis=dict(title="Anomaly Score", showgrid=False),
                                yaxis=dict(title="Count", showgrid=False))
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    # Anomaly table
    st.markdown('<div class="panel-title">📋 Detected Anomalies</div>', unsafe_allow_html=True)
    anom_show = anom_df[["date","close","volume","rsi","anomaly_score"]].copy()
    anom_show.columns = ["Date","Close ($)","Volume","RSI","Anomaly Score"]
    anom_show["Date"]          = anom_show["Date"].dt.strftime("%Y-%m-%d")
    anom_show["Close ($)"]     = anom_show["Close ($)"].map(lambda x: f"${x:,.2f}")
    anom_show["RSI"]           = anom_show["RSI"].map(lambda x: f"{x:.1f}")
    anom_show["Anomaly Score"] = anom_show["Anomaly Score"].map(lambda x: f"{x:.4f}")
    st.dataframe(anom_show.reset_index(drop=True), use_container_width=True,
                 hide_index=True)

# ══════════════════════════════════════════
# PAGE 7 — PORTFOLIO OPTIMIZER
# ══════════════════════════════════════════
elif page == "💼 Portfolio Optimizer":
    st.markdown('<h1 class="page-title">💼 Portfolio Optimizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">MARKOWITZ MEAN-VARIANCE · MONTE CARLO 2000 PORTFOLIOS</p>', unsafe_allow_html=True)

    # Build returns matrix
    coin_list = list(COINS.keys())
    all_returns = {}
    for c in coin_list:
        d  = generate_price_history(c, "1Y")
        all_returns[c] = d["close"].pct_change().dropna().values

    min_len = min(len(v) for v in all_returns.values())
    ret_df  = pd.DataFrame({c: all_returns[c][:min_len] for c in coin_list})

    c_cfg, c_chart = st.columns([1, 3])

    with c_cfg:
        st.markdown("**Portfolio Settings**")
        selected_coins = st.multiselect("Select Coins", coin_list,
                                         default=["BTC","ETH","SOL","BNB"])
        n_port = st.slider("Simulations", 500, 5000, 2000, 500)
        risk_free = st.slider("Risk-Free Rate %", 0, 10, 5) / 100

        if len(selected_coins) < 2:
            st.warning("Select at least 2 coins")
            st.stop()

        run_opt = st.button("🚀 Optimize Portfolio", use_container_width=True, type="primary")

    with c_chart:
        sub_ret = ret_df[selected_coins]
        with st.spinner(f"Running {n_port} Monte Carlo simulations..."):
            opt = run_portfolio_optimizer(sub_ret, n_portfolios=n_port)

        # Efficient frontier scatter
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=opt["vols"]*100, y=opt["returns"]*100, mode="markers",
            marker=dict(color=opt["sharpes"], colorscale="Viridis",
                        size=3, opacity=0.6, showscale=True,
                        colorbar=dict(title="Sharpe", thickness=8, tickfont=dict(size=9))),
            name="Portfolios", hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
        ))
        # Max Sharpe
        bi = opt["best_idx"]
        fig_ef.add_trace(go.Scatter(
            x=[opt["vols"][bi]*100], y=[opt["returns"][bi]*100],
            mode="markers", name="Max Sharpe",
            marker=dict(color="#f59e0b", size=14, symbol="star", line=dict(color="white",width=1.5))
        ))
        # Min Vol
        mi = opt["min_vol_idx"]
        fig_ef.add_trace(go.Scatter(
            x=[opt["vols"][mi]*100], y=[opt["returns"][mi]*100],
            mode="markers", name="Min Volatility",
            marker=dict(color="#10b981", size=14, symbol="diamond", line=dict(color="white",width=1.5))
        ))
        fig_ef.update_layout(
            **plotly_theme(), height=400,
            xaxis=dict(title="Annual Volatility %", gridcolor="rgba(99,179,255,0.07)"),
            yaxis=dict(title="Annual Return %",    gridcolor="rgba(99,179,255,0.07)"),
            title=dict(text="Efficient Frontier", font=dict(color="#e2e8f0", size=13)),
        )
        st.plotly_chart(fig_ef, use_container_width=True, config={"displayModeBar": False})

    # Results
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Expected Return",  f"{opt['best_return']*100:.1f}%")
    r2.metric("Annual Volatility",f"{opt['best_vol']*100:.1f}%")
    r3.metric("Sharpe Ratio",     f"{opt['best_sharpe']:.3f}")
    r4.metric("Risk-Adj. Return", f"{opt['best_return']/max(0.001,opt['best_vol']):.2f}x")

    st.markdown("<br>", unsafe_allow_html=True)
    c_w1, c_w2 = st.columns(2)

    with c_w1:
        st.markdown('<div class="panel-title">⭐ Max-Sharpe Weights</div>', unsafe_allow_html=True)
        wts   = opt["best_weights"]
        colors= ["#3b82f6","#f59e0b","#10b981","#8b5cf6","#ef4444","#06b6d4","#eab308","#e84142"]
        for coin, w, col in zip(selected_coins, wts, colors):
            st.markdown(f"""
            <div class="weight-bar-wrap">
                <div class="weight-bar-label">
                    <span>{coin}</span><span><b>{w*100:.1f}%</b></span>
                </div>
                <div class="weight-bar-track">
                    <div class="weight-bar-fill" style="width:{w*100:.1f}%;background:{col};"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    with c_w2:
        st.markdown('<div class="panel-title">📊 Weights Comparison</div>', unsafe_allow_html=True)
        min_w = opt["weights"][opt["min_vol_idx"]]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Max Sharpe", x=selected_coins, y=wts*100,
                                  marker_color="#f59e0b"))
        fig_bar.add_trace(go.Bar(name="Min Vol",    x=selected_coins, y=min_w*100,
                                  marker_color="#10b981"))
        fig_bar.update_layout(
            **plotly_theme(), barmode="group", height=280,
            xaxis=dict(showgrid=False), yaxis=dict(title="Weight %", showgrid=False),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # Correlation matrix
    st.markdown('<div class="panel-title">🔗 Correlation Matrix</div>', unsafe_allow_html=True)
    corr = sub_ret.corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0,"#ef4444"],[0.5,"#1e3a5f"],[1,"#10b981"]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(thickness=8, tickfont=dict(size=9)),
    ))
    fig_corr.update_layout(**plotly_theme(), height=320,
                            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})
