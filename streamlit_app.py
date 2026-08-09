"""
ShahabPrime Quant Dashboard - BYBIT VERSION
==============================================
A real-time market microstructure dashboard for crypto (Bybit, via ccxt).
Fixed for Streamlit Cloud deployment.
"""

import time
from collections import deque
import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# PAGE SETUP
st.set_page_config(page_title="ShahabPrime Quant Dashboard", layout="wide", page_icon="📊")

DARK_TEMPLATE = "plotly_dark"

st.markdown(
    """
    <style>
    .stApp { background-color: #05070c; }
    .alert-box {
        padding: 12px 16px; border-radius: 8px; margin-top: 8px;
        font-family: monospace; font-size: 13px;
    }
    .alert-warn { background: rgba(255,176,32,0.1); border: 1px solid rgba(255,176,32,0.4); color: #ffb020; }
    .alert-danger { background: rgba(255,56,96,0.1); border: 1px solid rgba(255,56,96,0.4); color: #ff3860; }
    .alert-ok { background: rgba(0,255,163,0.08); border: 1px solid rgba(0,255,163,0.35); color: #00ffa3; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 ShahabPrime — Quant Trading Dashboard")
st.caption("Real-time market microstructure for crypto, via Bybit. Educational tool — not financial advice.")

# SIDEBAR CONTROLS
with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.text_input("Symbol (e.g., BTC/USDT)", value="BTC/USDT")
    refresh_enabled = st.checkbox("Auto-refresh enabled", value=False)

# CACHE & EXCHANGE - BYBIT
@st.cache_resource
def get_exchange():
    try:
        return ccxt.bybit({"enableRateLimit": True})
    except Exception as e:
        st.error(f"Failed to initialize Bybit: {e}")
        return None

exchange = get_exchange()

if exchange is None:
    st.error("❌ Could not connect to Bybit")
    st.stop()

# SESSION STATE
if "trade_buffer" not in st.session_state:
    st.session_state.trade_buffer = deque(maxlen=1000)
if "cvd_history" not in st.session_state:
    st.session_state.cvd_history = []
if "cvd_running" not in st.session_state:
    st.session_state.cvd_running = 0.0

# DATA FETCHERS
def fetch_order_book(sym, limit=20):
    """Fetch order book from Bybit safely"""
    try:
        return exchange.fetch_order_book(sym, limit=limit)
    except Exception as e:
        st.error(f"Order book fetch failed: {e}")
        return None

def fetch_ticker(sym):
    """Fetch ticker info from Bybit safely"""
    try:
        return exchange.fetch_ticker(sym)
    except Exception as e:
        st.error(f"Ticker fetch failed: {e}")
        return None

def fetch_recent_trades(sym, limit=100):
    """Fetch recent trades from Bybit safely"""
    try:
        return exchange.fetch_trades(sym, limit=limit)
    except Exception as e:
        st.error(f"Trades fetch failed: {e}")
        return []

# MODULE 1: ORDER BOOK IMBALANCE
def compute_imbalance(order_book, depth=5):
    bids = order_book["bids"][:depth]
    asks = order_book["asks"][:depth]
    B = sum(q for _, q in bids)
    A = sum(q for _, q in asks)
    if B + A == 0:
        return 0.5, B, A
    return B / (B + A), B, A

# MODULE 2: VPIN METER
def compute_vpin(trade_buffer, bucket_size=5.0, n_buckets=10):
    if not trade_buffer:
        return None
    buckets = []
    cur_buy, cur_sell, cur_vol = 0.0, 0.0, 0.0
    
    for ts, price, amount, side in trade_buffer:
        remaining = amount
        while remaining > 1e-12:
            take = min(remaining, bucket_size - cur_vol)
            if side == "buy":
                cur_buy += take
            else:
                cur_sell += take
            cur_vol += take
            remaining -= take
            if cur_vol >= bucket_size - 1e-9:
                buckets.append((cur_buy, cur_sell))
                cur_buy, cur_sell, cur_vol = 0.0, 0.0, 0.0
    
    if len(buckets) < 3:
        return None
    
    recent = buckets[-int(n_buckets):]
    imbalances = [abs(b - s) for b, s in recent]
    vpin = sum(imbalances) / (len(recent) * bucket_size) if bucket_size > 0 else 0.0
    return vpin

# MODULE 3: CVD
def update_cvd(trades, symbol):
    if not trades:
        return
    
    prev_price = None
    for t in trades:
        side = t.get("side")
        if side is None:
            if prev_price is not None:
                side = "buy" if t["price"] >= prev_price else "sell"
            else:
                side = "buy"
        prev_price = t["price"]
        
        signed = t["amount"] if side == "buy" else -t["amount"]
        st.session_state.cvd_running += signed
        st.session_state.cvd_history.append((t["timestamp"], t["price"], st.session_state.cvd_running))
    
    if len(st.session_state.cvd_history) > 1000:
        st.session_state.cvd_history = st.session_state.cvd_history[-1000:]

# MODULE 4: KELLY CRITERION
def kelly_fraction(win_rate, avg_win, avg_loss):
    if avg_loss <= 0:
        return 0.0, 0.0
    R = avg_win / avg_loss
    f_star = win_rate - (1 - win_rate) / R
    f_star = max(0.0, f_star)
    return f_star, f_star / 2

# MAIN UI - TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📖 Order Book", "🌡️ VPIN", "💰 CVD", "🎯 Kelly", "🪙 Ticker"]
)

# Fetch data from Bybit
order_book = fetch_order_book(symbol)
ticker = fetch_ticker(symbol)
recent_trades = fetch_recent_trades(symbol, limit=100)

if recent_trades:
    for t in recent_trades:
        st.session_state.trade_buffer.append((t["timestamp"], t["price"], t["amount"], t.get("side", "buy")))
    update_cvd(recent_trades, symbol)

# ---- TAB 1: ORDER BOOK ----
with tab1:
    st.subheader("Order Book Imbalance")
    if order_book:
        imbalance, B, A = compute_imbalance(order_book, depth=5)
        c1, c2, c3 = st.columns(3)
        c1.metric("Bid Volume (top 5)", f"{B:.6f}")
        c2.metric("Ask Volume (top 5)", f"{A:.6f}")
        c3.metric("Imbalance i = B/(B+A)", f"{imbalance:.3f}")
        st.progress(imbalance)
        
        if imbalance > 0.65:
            st.markdown('<div class="alert-box alert-ok">✅ Bid-heavy — buyers dominate</div>', unsafe_allow_html=True)
        elif imbalance < 0.35:
            st.markdown('<div class="alert-box alert-danger">⚠️ Ask-heavy — sellers dominate</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-box">⚖️ Balanced book</div>', unsafe_allow_html=True)
        
        bids_df = pd.DataFrame(order_book["bids"][:15], columns=["price", "qty"])
        asks_df = pd.DataFrame(order_book["asks"][:15], columns=["price", "qty"])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bids_df["price"], y=bids_df["qty"], name="Bids", marker_color="#00ffa3"))
        fig.add_trace(go.Bar(x=asks_df["price"], y=asks_df["qty"], name="Asks", marker_color="#ff3860"))
        fig.update_layout(template=DARK_TEMPLATE, title="Order Book Levels", height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Order book unavailable")

# ---- TAB 2: VPIN ----
with tab2:
    st.subheader("VPIN Meter (Volume-Synchronized Probability of Informed Trading)")
    vpin = compute_vpin(st.session_state.trade_buffer, bucket_size=5.0, n_buckets=10)
    
    if vpin is None:
        st.info(f"Accumulating trades... ({len(st.session_state.trade_buffer)} captured)")
    else:
        gauge_color = "#00ffa3" if vpin < 0.4 else "#ffb020" if vpin < 0.7 else "#ff3860"
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=vpin, number={"valueformat": ".3f"},
            gauge={
                "axis": {"range": [0, 1]}, "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 0.4], "color": "rgba(0,255,163,0.15)"},
                    {"range": [0.4, 0.7], "color": "rgba(255,176,32,0.15)"},
                    {"range": [0.7, 1], "color": "rgba(255,56,96,0.15)"},
                ],
            },
        ))
        fig.update_layout(template=DARK_TEMPLATE, height=320)
        st.plotly_chart(fig, use_container_width=True)
        
        if vpin > 0.70:
            st.markdown('<div class="alert-box alert-danger">⚠️ VPIN ELEVATED (>0.70) - High order-flow toxicity</div>', unsafe_allow_html=True)
        elif vpin > 0.4:
            st.markdown('<div class="alert-box alert-warn">⚡ Moderate toxicity</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-box alert-ok">✅ Low toxicity — balanced trading</div>', unsafe_allow_html=True)

# ---- TAB 3: CVD ----
with tab3:
    st.subheader("Cumulative Volume Delta (CVD) vs Price")
    if len(st.session_state.cvd_history) < 5:
        st.info(f"Accumulating trades... ({len(st.session_state.cvd_history)} trades captured)")
    else:
        df = pd.DataFrame(st.session_state.cvd_history, columns=["ts", "price", "cvd"])
        df["time"] = pd.to_datetime(df["ts"], unit="ms")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["time"], y=df["price"], name="Price", line=dict(color="#00e5ff"), yaxis="y1"))
        fig.add_trace(go.Scatter(x=df["time"], y=df["cvd"], name="CVD", line=dict(color="#ffb020"), yaxis="y2"))
        fig.update_layout(
            template=DARK_TEMPLATE, height=420,
            yaxis=dict(title="Price"), yaxis2=dict(title="CVD", overlaying="y", side="right"),
            title="Price vs Cumulative Volume Delta",
        )
        st.plotly_chart(fig, use_container_width=True)

# ---- TAB 4: KELLY ----
with tab4:
    st.subheader("Kelly Criterion Position Sizing")
    c1, c2, c3 = st.columns(3)
    win_rate = c1.number_input("Win rate (%)", 1.0, 99.0, 55.0, step=1.0) / 100
    avg_win = c2.number_input("Avg win (%)", 0.01, value=2.5, step=0.1)
    avg_loss = c3.number_input("Avg loss (%)", 0.01, value=1.5, step=0.1)
    
    full_kelly, half_kelly = kelly_fraction(win_rate, avg_win, avg_loss)
    
    c1, c2 = st.columns(2)
    c1.metric("Full Kelly", f"{full_kelly*100:.1f}%")
    c2.metric("Half Kelly (Recommended)", f"{half_kelly*100:.1f}%")
    
    st.info("💡 Half Kelly is safer and protects against model errors")

# ---- TAB 5: TICKER ----
with tab5:
    st.subheader(f"Ticker Info for {symbol} (Bybit)")
    if ticker:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${ticker.get('last', 0):.2f}")
        c2.metric("24h Change %", f"{ticker.get('percentage', 0):.2f}%")
        c3.metric("24h High", f"${ticker.get('high', 0):.2f}")
        c4.metric("24h Low", f"${ticker.get('low', 0):.2f}")
        
        st.write("---")
        st.metric("24h Volume", f"${ticker.get('quoteVolume', 0):.2f}")
    else:
        st.warning("Ticker data unavailable")

st.divider()
st.caption("Data: Bybit via ccxt | Real-time microstructure analysis | Not financial advice")
st.caption("⚠️ DISCLAIMER: This is an educational tool only. Always DYOR. Never trade with money you can't afford to lose.")

if refresh_enabled:
    time.sleep(5)
    st.rerun()
