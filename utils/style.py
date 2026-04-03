import streamlit as st


def apply_custom_css():
    st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg:       #080c14;
    --bg2:      #0d1420;
    --bg3:      #111b2d;
    --bg4:      #1a2640;
    --border:   rgba(99,179,255,0.12);
    --border2:  rgba(99,179,255,0.25);
    --accent:   #3b82f6;
    --accent2:  #06b6d4;
    --accent3:  #8b5cf6;
    --green:    #10b981;
    --red:      #ef4444;
    --amber:    #f59e0b;
    --text:     #e2e8f0;
    --text2:    #94a3b8;
    --text3:    #475569;
    --mono:     'Space Mono', monospace;
    --display:  'Syne', sans-serif;
}

/* ── Global App ── */
html, body, [class*="css"] {
    font-family: var(--display) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg); }
.block-container { padding-top: 1.5rem !important; padding-bottom: 3rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text2) !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.85rem !important; padding: 0.35rem 0.5rem; border-radius: 6px; transition: background 0.2s; }
[data-testid="stSidebar"] .stRadio label:hover { background: var(--bg4) !important; color: var(--text) !important; }
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }

/* ── Sidebar Logo ── */
.sidebar-logo {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.5rem 0 0.5rem;
}
.logo-icon {
    font-size: 1.8rem; line-height: 1;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.logo-name  { font-size: 1.3rem; font-weight: 800; color: #e2e8f0 !important; letter-spacing: -0.03em; }
.logo-sub   { font-size: 0.62rem; font-family: var(--mono); color: var(--text3) !important; letter-spacing: 0.08em; }

/* Live indicator */
.live-indicator {
    display: flex; align-items: center; gap: 0.5rem;
    font-family: var(--mono); font-size: 0.72rem; color: #10b981 !important;
    border: 1px solid rgba(16,185,129,0.3); border-radius: 4px;
    padding: 0.35rem 0.7rem; margin-bottom: 0.5rem;
}
.live-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #10b981; display: inline-block;
    animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ── Page Titles ── */
.page-title {
    font-size: 2rem; font-weight: 800; letter-spacing: -0.03em;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.15rem; line-height: 1.15;
}
.page-sub {
    font-family: var(--mono); font-size: 0.72rem;
    color: var(--text3); letter-spacing: 0.06em; margin-bottom: 1.2rem;
}
.panel-title {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--text2);
    margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.4rem;
}

/* ── KPI Cards ── */
.kpi-card {
    border-radius: 12px; padding: 1rem 1.1rem;
    border: 1px solid var(--border); position: relative; overflow: hidden;
    background: var(--bg2); margin-bottom: 0.25rem;
}
.kpi-card::before { content:''; position:absolute; top:0;left:0;right:0;height:3px; border-radius:12px 12px 0 0; }
.kpi-blue::before   { background: var(--accent); }
.kpi-green::before  { background: var(--green); }
.kpi-orange::before { background: var(--amber); }
.kpi-yellow::before { background: #eab308; }
.kpi-purple::before { background: var(--accent3); }
.kpi-label { font-family:var(--mono); font-size:0.62rem; color:var(--text3); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.3rem; }
.kpi-value { font-size:1.25rem; font-weight:800; letter-spacing:-0.02em; color:var(--text); margin-bottom:0.15rem; }
.kpi-sub   { font-family:var(--mono); font-size:0.62rem; color:var(--text3); }

/* ── Ticker ── */
.ticker-outer {
    overflow: hidden; background: var(--bg2);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 0.5rem 0; margin-bottom: 1.2rem; white-space: nowrap;
}
.ticker-inner {
    display: inline-flex; gap: 2.5rem;
    animation: ticker 40s linear infinite;
    padding: 0 1rem;
}
@keyframes ticker { from{transform:translateX(0)} to{transform:translateX(-50%)} }
.ticker-item { display:inline-flex; align-items:center; gap:0.5rem; font-family:var(--mono); font-size:0.72rem; flex-shrink:0; color:var(--text2); }
.ticker-item b { color: var(--text); }
.up   { color: #10b981 !important; font-weight:700; }
.down { color: #ef4444 !important; font-weight:700; }

/* ── Coin rows ── */
.coin-row {
    display:flex; align-items:center; gap:0.75rem;
    padding:0.55rem 0.5rem; border-bottom:1px solid var(--border);
    border-radius:6px; cursor:pointer; transition:background 0.15s;
}
.coin-row:hover { background: var(--bg3); }
.coin-row:last-child { border-bottom: none; }
.coin-emoji { font-size: 1.2rem; flex-shrink:0; }
.coin-meta  { flex:1; min-width:0; }
.coin-sym   { font-size:0.8rem; font-weight:700; color:var(--text); }
.coin-name  { font-family:var(--mono); font-size:0.62rem; color:var(--text3); }
.coin-price-col { text-align:right; }
.coin-price { font-family:var(--mono); font-size:0.8rem; font-weight:700; color:var(--text); }
.coin-chg   { font-family:var(--mono); font-size:0.65rem; font-weight:700; }

/* ── On-chain rows ── */
.onchain-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:0.45rem 0; border-bottom:1px solid var(--border);
    font-size:0.78rem; color:var(--text2);
}
.onchain-row:last-child { border-bottom:none; }
.onchain-val { font-family:var(--mono); font-size:0.78rem; font-weight:700; color:var(--text); }

/* ── Signal Hero ── */
.signal-hero {
    border: 1px solid; border-radius:12px; padding:1.2rem;
    text-align:center; margin-bottom:1rem;
}
.signal-emoji { font-size:2.5rem; margin-bottom:0.4rem; }
.signal-label { font-size:1.8rem; font-weight:800; letter-spacing:-0.02em; }
.signal-conf  { font-family:var(--mono); font-size:0.72rem; color:var(--text3); margin:0.4rem 0; }

/* ── Indicator rows ── */
.indicator-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:0.45rem 0; border-bottom:1px solid var(--border);
    font-size:0.78rem;
}
.indicator-row:last-child { border-bottom:none; }
.ind-name   { color:var(--text2); font-size:0.75rem; }
.ind-status { font-family:var(--mono); font-size:0.6rem; color:var(--text3); }
.ind-val    { font-family:var(--mono); font-size:0.78rem; font-weight:700; color:var(--text); }

/* ── Confidence bar ── */
.conf-track { height:4px; background:var(--bg4); border-radius:2px; overflow:hidden; margin-top:3px; }
.conf-fill  { height:100%; border-radius:2px; transition:width 0.8s; }

/* ── Signal history ── */
.sig-history-row {
    display:flex; justify-content:space-between;
    padding:0.35rem 0; border-bottom:1px solid var(--border);
    font-family:var(--mono); font-size:0.72rem;
}

/* ── Headline cards ── */
.headline-card {
    background:var(--bg3); border:1px solid var(--border);
    border-radius:8px; padding:0.7rem 0.9rem; margin-bottom:0.5rem;
}
.headline-meta { display:flex; justify-content:space-between; margin-bottom:0.3rem; }
.headline-source { font-family:var(--mono); font-size:0.6rem; color:var(--text3); text-transform:uppercase; letter-spacing:0.08em; }
.headline-text   { font-size:0.78rem; color:var(--text); line-height:1.4; }

/* ── Arch boxes ── */
.arch-block {
    border-radius:8px; padding:0.5rem 0.8rem;
    font-family:var(--mono); font-size:0.65rem;
    color:white; text-align:center; line-height:1.4;
    border:1px solid rgba(99,179,255,0.2);
}
.arch-arrow { color:var(--text3); font-size:1rem; }

/* ── Buttons ── */
.stButton>button {
    background: var(--accent) !important; color:white !important;
    border: none !important; border-radius:8px !important;
    font-family:var(--display) !important; font-weight:700 !important;
    transition: opacity 0.2s !important;
}
.stButton>button:hover { opacity:0.85 !important; }

/* ── Selectbox / Slider ── */
.stSelectbox>div>div, .stSlider { color:var(--text2) !important; }
[data-testid="stSlider"] .stSlider { background:var(--bg4); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background:var(--bg2) !important; border-radius:8px; gap:0.25rem; border:1px solid var(--border); padding:0.25rem; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:var(--text3) !important; border-radius:6px !important; font-family:var(--display) !important; font-size:0.78rem !important; font-weight:600 !important; }
.stTabs [aria-selected="true"] { background:var(--accent) !important; color:white !important; }

/* ── Metric ── */
[data-testid="stMetric"] { background:var(--bg2); border:1px solid var(--border); border-radius:10px; padding:0.8rem 1rem; }
[data-testid="stMetricLabel"] { font-family:var(--mono); font-size:0.65rem !important; color:var(--text3) !important; letter-spacing:0.06em; }
[data-testid="stMetricValue"] { font-size:1.4rem !important; font-weight:800 !important; color:var(--text) !important; }

/* ── Code blocks ── */
.stCode { background:var(--bg3) !important; border:1px solid var(--border) !important; border-radius:8px !important; font-family:var(--mono) !important; }

/* ── Info/Success boxes ── */
.stInfo    { background:rgba(59,130,246,0.1) !important; border-color:var(--accent) !important; border-radius:8px !important; }
.stSuccess { background:rgba(16,185,129,0.1) !important; border-radius:8px !important; }
.stWarning { background:rgba(245,158,11,0.1) !important; border-radius:8px !important; }

/* ── Spinner ── */
.stSpinner { color: var(--accent) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:var(--bg2); }
::-webkit-scrollbar-thumb { background:var(--bg4); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--accent); }

/* ── Portfolio weights bars ── */
.weight-bar-wrap { margin-bottom:0.6rem; }
.weight-bar-label { display:flex; justify-content:space-between; font-size:0.75rem; color:var(--text2); margin-bottom:3px; }
.weight-bar-track { height:10px; background:var(--bg4); border-radius:5px; overflow:hidden; }
.weight-bar-fill  { height:100%; border-radius:5px; }

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .page-title { font-size: 1.4rem !important; }
    .kpi-value  { font-size: 1rem !important; }
    .block-container { padding: 0.75rem 0.5rem !important; }
}
</style>
""", unsafe_allow_html=True)
