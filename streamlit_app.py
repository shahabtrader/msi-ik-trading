"""
ShahabPrime Quant Dashboard
===========================
A real-time market microstructure dashboard for crypto (Binance, via ccxt).
"""

import time
from collections import deque
import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from math import erf, sqrt

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
st.caption("Real-time market microstructure for crypto, via ccxt/Binance. Educational tool — not financial advice.")

# SIDEBAR CONTROLS
with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.text_input("Symbol (ccxt format)", value="BTC/USDT")
    refresh_secs = st.slider("Refresh interval (seconds)", 3, 30, 6)
    vpin_bucket_size = st.number_input("VPIN volume bucket size", min_value=0.1, value=5.0, step=0.5)
    vpin_n_buckets = st.number_input("VPIN window (# buckets)", min_value=5, value=20, step=1)
    auto_refresh = st.checkbox("Auto-refresh", value=True)

# CACHE & EXCHANGE
@st.cache_resource
def get_exchange():
    return ccxt.binance({"enableRateLimit": True})

exchange = get_exchange()

# SESSION STATE
if "trade_buffer" not in st.session_state:
    st.session_state.trade_buffer = deque(maxlen=5000)
if "cvd_history" not in st.session_state:
    st.session_state.cvd_history = []
if "cvd_running" not in st.session_state:
    st.session_state.cvd_running = 0.0
if "last_trade_id" not in st.session_state:
    st.session_state.last_trade_id = None
if "book_history" not in st.session_state:
    st.session_state.book_history = deque(maxlen=60)

# DATA FETCHERS
def fetch_order_book(sym, limit=20):
    try:
        return exchange.fetch_order_book(sym, limit=limit)
    except Exception as e:
        st.error(f"Order book fetch failed: {e}")
        return None

def fetch_recent_trades(sym, limit=200):
    try:
        return exchange.fetch_trades(sym, limit=limit)
    except Exception as e:
        st.error(f"Trades fetch failed: {e}")
        return []

def classify_trade_side(trade):
    if trade.get("side") in ("buy", "sell"):
        return trade["side"]
    return None

def update_trade_buffer(sym):
    trades = fetch_recent_trades(sym, limit=200)
    if not trades:
        return
    prev_price = None
    for t in trades:
        tid = t.get("id") or t.get("timestamp")
        if st.session_state.last_trade_id is not None and tid == st.session_state.last_trade_id:
            continue
        side = classify_trade_side(t)
        if side is None:
            if prev_price is not None:
                side = "buy" if t["price"] >= prev_price else "sell"
            else:
                side = "buy"
        prev_price = t["price"]
        st.session_state.trade_buffer.append((t["timestamp"], t["price"], t["amount"], side))
        signed = t["amount"] if side == "buy" else -t["amount"]
        st.session_state.cvd_running += signed
        st.session_state.cvd_history.append((t["timestamp"], t["price"], st.session_state.cvd_running))
    if trades:
        st.session_state.last_trade_id = trades[-1].get("id") or trades[-1].get("timestamp")
    if len(st.session_state.cvd_history) > 3000:
        st.session_state.cvd_history = st.session_state.cvd_history[-3000:]

# MODULE 1: ORDER BOOK IMBALANCE & SPOOFING RADAR
def compute_imbalance(order_book, depth=5):
    bids = order_book["bids"][:depth]
    asks = order_book["asks"][:depth]
    B = sum(q for _, q in bids)
    A = sum(q for _, q in asks)
    if B + A == 0:
        return 0.5, B, A
    return B / (B + A), B, A

def spoofing_heuristic(book_history):
    if len(book_history) < 3:
        return None
    latest = book_history[-1]
    prev = book_history[-3]
    alerts = []
    for side, label in [("bids", "BID"), ("asks", "ASK")]:
        prev_levels = {round(p, 2): q for p, q in prev[side][:5]}
        latest_levels = {round(p, 2): q for p, q in latest[side][:5]}
        for price, qty in prev_levels.items():
            other_sizes = [v for k, v in prev_levels.items() if k != price]
            median_other = np.median(other_sizes) if other_sizes else qty
            was_wall = qty > 4 * max(median_other, 1e-9)
            still_there = latest_levels.get(price, 0)
            if was_wall and still_there < qty * 0.15:
                alerts.append(f"{label} wall of {qty:.3f} @ {price} appeared then vanished (down to {still_there:.3f})")
    return alerts

# MODULE 2: VPIN METER
def compute_vpin(trade_buffer, bucket_size, n_buckets):
    if not trade_buffer:
        return None, []
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
        return None, buckets
    recent = buckets[-int(n_buckets):]
    imbalances = [abs(b - s) for b, s in recent]
    vpin = sum(imbalances) / (len(recent) * bucket_size)
    return vpin, recent

# MODULE 3: CVD & DIVERGENCE
def detect_cvd_divergence(cvd_history, lookback=60):
    if len(cvd_history) < lookback:
        return None
    window = cvd_history[-lookback:]
    prices = np.array([p for _, p, _ in window])
    cvds = np.array([c for _, _, c in window])
    peak_idx = []
    for i in range(2, len(prices) - 2):
        if prices[i] == max(prices[max(0, i - 5): i + 5]):
            peak_idx.append(i)
    peak_idx = sorted(set(peak_idx))
    if len(peak_idx) < 2:
        return None
    i1, i2 = peak_idx[-2], peak_idx[-1]
    price_hh = prices[i2] > prices[i1]
    cvd_lh = cvds[i2] < cvds[i1]
    price_ll = prices[i2] < prices[i1]
    cvd_hl = cvds[i2] > cvds[i1]
    if price_hh and cvd_lh:
        return "bearish", i1, i2
    if price_ll and cvd_hl:
        return "bullish", i1, i2
    return None

# MODULE 4: KELLY CRITERION
def kelly_fraction(win_rate, avg_win, avg_loss):
    if avg_loss <= 0:
        return 0.0, 0.0
    R = avg_win / avg_loss
    f_star = win_rate - (1 - win_rate) / R
    f_star = max(0.0, f_star)
    return f_star, f_star / 2

# MODULE 5: MULTI-COIN COMPARISON
def fetch_coin_snapshot(sym):
    try:
        ob = exchange.fetch_order_book(sym, limit=10)
        imbalance, B, A = compute_imbalance(ob, depth=10)
        ticker = exchange.fetch_ticker(sym)
        funding = exchange.fetch_funding_rate(sym)
        return {
            "symbol": sym,
            "price": ticker.get("last"),
            "change_24h_pct": ticker.get("percentage"),
            "imbalance": imbalance,
            "funding_rate": funding.get("fundingRate", 0.0),
            "volume_24h": ticker.get("quoteVolume"),
        }
    except Exception as e:
        return {"symbol": sym, "error": str(e)}

# MODULE 6: HISTORICAL BACKTEST
def fetch_klines_with_taker_volume(sym, timeframe="1h", limit=720):
    market_symbol = sym.replace("/", "")
    try:
        raw = exchange.fapiPublicGetKlines({"symbol": market_symbol, "interval": timeframe, "limit": limit})
    except:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "num_trades", "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume", "taker_buy_base_vol"]:
        df[col] = df[col].astype(float)
    df["taker_sell_base_vol"] = df["volume"] - df["taker_buy_base_vol"]
    df["imbalance"] = (df["taker_buy_base_vol"] - df["taker_sell_base_vol"]) / df["volume"].replace(0, np.nan)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def backtest_imbalance_vs_returns(df, forward_bars=4, n_quintiles=5):
    d = df.copy()
    d["fwd_return_pct"] = (d["close"].shift(-forward_bars) - d["close"]) / d["close"] * 100
    d = d.dropna(subset=["imbalance", "fwd_return_pct"])
    if len(d) < 30:
        return None, None, None
    d["quintile"] = pd.qcut(d["imbalance"], n_quintiles, labels=False, duplicates="drop")
    summary = d.groupby("quintile").agg(
        mean_imbalance=("imbalance", "mean"),
        mean_fwd_return_pct=("fwd_return_pct", "mean"),
        win_rate_pct=("fwd_return_pct", lambda x: (x > 0).mean() * 100),
        n=("fwd_return_pct", "count"),
    ).reset_index()
    corr = d["imbalance"].corr(d["fwd_return_pct"])
    return summary, corr, len(d)

# MODULE 7: FUNDING RATE & VOLATILITY
def fetch_funding_history(sym, limit=200):
    try:
        hist = exchange.fetch_funding_rate_history(sym, limit=limit)
        df = pd.DataFrame(hist)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["fundingRate_pct"] = df["fundingRate"] * 100
        return df[["time", "fundingRate_pct"]]
    except Exception as e:
        st.warning(f"Funding history fetch failed: {e}")
        return pd.DataFrame()

def compute_realized_volatility(df, window=24, periods_per_year=None):
    d = df.copy()
    d["log_ret"] = np.log(d["close"] / d["close"].shift(1))
    d["realized_vol"] = d["log_ret"].rolling(window).std() * np.sqrt(periods_per_year or 365)
    return d

# MODULE 8: RISK METRICS
def compute_return_series(df):
    d = df.copy()
    d["ret"] = d["close"].pct_change()
    d["log_ret"] = np.log(d["close"] / d["close"].shift(1))
    return d.dropna(subset=["ret"])

def sharpe_ratio(returns, periods_per_year, risk_free_rate=0.0):
    excess = returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)

def sortino_ratio(returns, periods_per_year, risk_free_rate=0.0):
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or pd.isna(downside_std):
        return 0.0
    return (excess.mean() / downside_std) * np.sqrt(periods_per_year)

def max_drawdown(prices):
    cum_max = prices.cummax()
    drawdown = (prices - cum_max) / cum_max
    return drawdown.min()

def compute_zscore(prices, window=20):
    roll_mean = prices.rolling(window).mean()
    roll_std = prices.rolling(window).std()
    return (prices - roll_mean) / roll_std

def monte_carlo_bootstrap(returns, n_sims=2000, n_periods=100, starting_capital=10000):
    returns_arr = returns.dropna().values
    if len(returns_arr) < 20:
        return None
    ending_capitals = []
    max_drawdowns = []
    for _ in range(n_sims):
        sampled = np.random.choice(returns_arr, size=n_periods, replace=True)
        equity = starting_capital * np.cumprod(1 + sampled)
        ending_capitals.append(equity[-1])
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        max_drawdowns.append(dd.min())
    return {
        "ending_capitals": np.array(ending_capitals),
        "max_drawdowns": np.array(max_drawdowns),
    }

# MAIN UI - TABS
order_book = fetch_order_book(symbol)
if order_book:
    st.session_state.book_history.append(order_book)
update_trade_buffer(symbol)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["📖 Order Book", "🌡️ VPIN", "💰 CVD", "🎯 Kelly", "🪙 Multi-Coin", "📈 Backtest", "💸 Funding", "📐 Risk"]
)

# ---- TAB 1: ORDER BOOK ----
with tab1:
    st.subheader("Order Book Imbalance")
    if order_book:
        imbalance, B, A = compute_imbalance(order_book, depth=5)
        c1, c2, c3 = st.columns(3)
        c1.metric("Bid Volume (top 5)", f"{B:.3f}")
        c2.metric("Ask Volume (top 5)", f"{A:.3f}")
        c3.metric("Imbalance i = B/(B+A)", f"{imbalance:.3f}")
        st.progress(imbalance)
        if imbalance > 0.65:
            st.markdown('<div class="alert-box alert-ok">✅ Bid-heavy — buyers dominate</div>', unsafe_allow_html=True)
        elif imbalance < 0.35:
            st.markdown('<div class="alert-box alert-danger">⚠️ Ask-heavy — sellers dominate</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-box">⚖️ Balanced book</div>', unsafe_allow_html=True)
        st.subheader("Spoofing Radar")
        alerts = spoofing_heuristic(st.session_state.book_history)
        if alerts:
            for a in alerts:
                st.markdown(f'<div class="alert-box alert-warn">⚠️ {a}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-box alert-ok">✅ No spoofing patterns detected</div>', unsafe_allow_html=True)
        bids_df = pd.DataFrame(order_book["bids"][:20], columns=["price", "qty"])
        asks_df = pd.DataFrame(order_book["asks"][:20], columns=["price", "qty"])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bids_df["price"], y=bids_df["qty"], name="Bids", marker_color="#00ffa3"))
        fig.add_trace(go.Bar(x=asks_df["price"], y=asks_df["qty"], name="Asks", marker_color="#ff3860"))
        fig.update_layout(template=DARK_TEMPLATE, title="Top 20 Order Book Levels", height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Order book unavailable")

# ---- TAB 2: VPIN ----
with tab2:
    st.subheader("VPIN Meter")
    vpin, buckets = compute_vpin(st.session_state.trade_buffer, vpin_bucket_size, vpin_n_buckets)
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
        st.info("Accumulating trades for CVD...")
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
        div = detect_cvd_divergence(st.session_state.cvd_history)
        if div is None:
            st.info("No divergence detected")
        elif div[0] == "bearish":
            st.markdown('<div class="alert-box alert-danger">⚠️ BEARISH DIVERGENCE - Higher price, lower CVD</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-box alert-ok">✅ BULLISH DIVERGENCE - Lower price, higher CVD</div>', unsafe_allow_html=True)

# ---- TAB 4: KELLY ----
with tab4:
    st.subheader("Kelly Criterion Position Sizing")
    c1, c2, c3 = st.columns(3)
    win_rate = c1.number_input("Win rate (%)", 1.0, 99.0, 50.0, step=1.0) / 100
    avg_win = c2.number_input("Avg win (%)", 0.01, value=2.0, step=0.1)
    avg_loss = c3.number_input("Avg loss (%)", 0.01, value=1.0, step=0.1)
    full_kelly, half_kelly = kelly_fraction(win_rate, avg_win, avg_loss)
    c1, c2 = st.columns(2)
    c1.metric("Full Kelly", f"{full_kelly*100:.1f}%")
    c2.metric("Half Kelly (Recommended)", f"{half_kelly*100:.1f}%")

# ---- TAB 5: MULTI-COIN ----
with tab5:
    st.subheader("Multi-Coin Comparison")
    default_list = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
    coins = st.multiselect("Select coins", default_list, default=default_list)
    if coins:
        rows = [fetch_coin_snapshot(c) for c in coins]
        comp_df = pd.DataFrame(rows)
        if "error" in comp_df.columns:
            bad = comp_df[comp_df["error"].notna()]
            if not bad.empty:
                st.warning(f"Could not fetch: {', '.join(bad['symbol'])}")
            comp_df = comp_df[comp_df["error"].isna()].drop(columns=["error"])
        if not comp_df.empty:
            comp_df["imbalance"] = comp_df["imbalance"].round(3)
            comp_df["funding_rate_pct"] = (comp_df["funding_rate"] * 100).round(4)
            comp_df["change_24h_pct"] = comp_df["change_24h_pct"].round(2)
            show_df = comp_df[["symbol", "price", "change_24h_pct", "imbalance", "funding_rate_pct"]]
            st.dataframe(show_df, use_container_width=True, hide_index=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=comp_df["symbol"], y=comp_df["imbalance"], marker_color="#00e5ff"))
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
            fig.update_layout(template=DARK_TEMPLATE, height=300, title="Order Book Imbalance by Coin")
            st.plotly_chart(fig, use_container_width=True)

# ---- TAB 6: BACKTEST ----
with tab6:
    st.subheader("Historical Backtest")
    c1, c2, c3 = st.columns(3)
    bt_timeframe = c1.selectbox("Timeframe", ["15m", "1h", "4h"], index=1)
    bt_limit = c2.slider("Candles back", 100, 1500, 720)
    bt_forward = c3.slider("Forward window", 1, 20, 4)
    if st.button("Run Backtest"):
        with st.spinner("Downloading historical data..."):
            try:
                hist_df = fetch_klines_with_taker_volume(symbol, timeframe=bt_timeframe, limit=bt_limit)
                summary, corr, n = backtest_imbalance_vs_returns(hist_df, forward_bars=bt_forward)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                summary, corr, n = None, None, None
        if summary is not None:
            st.metric("Correlation", f"{corr:.3f}")
            st.markdown(f"**Sample size:** {n} candles")
            fig = go.Figure(go.Bar(
                x=[f"Q{int(q)+1}" for q in summary["quintile"]],
                y=summary["mean_fwd_return_pct"],
                marker_color=np.where(summary["mean_fwd_return_pct"] >= 0, "#00ffa3", "#ff3860"),
            ))
            fig.update_layout(template=DARK_TEMPLATE, height=360, title="Mean Forward Return by Imbalance Quintile")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data for backtest")

# ---- TAB 7: FUNDING ----
with tab7:
    st.subheader("Funding Rate History")
    fh = fetch_funding_history(symbol, limit=200)
    if not fh.empty:
        fig = go.Figure(go.Scatter(x=fh["time"], y=fh["fundingRate_pct"], line=dict(color="#ffb020"), fill="tozeroy"))
        fig.add_hline(y=0, line_color="gray")
        fig.update_layout(template=DARK_TEMPLATE, height=320, title="Funding Rate History (%)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Funding history unavailable")

# ---- TAB 8: RISK METRICS ----
with tab8:
    st.subheader("Risk Metrics")
    c1, c2, c3 = st.columns(3)
    rm_timeframe = c1.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1, key="rm_tf")
    rm_limit = c2.slider("Candles", 100, 1000, 400, key="rm_limit")
    rm_zwindow = c3.slider("Z-score window", 5, 60, 20)
    periods_map = {"15m": 4*24*365, "1h": 24*365, "4h": 6*365, "1d": 365}
    try:
        rm_df = fetch_klines_with_taker_volume(symbol, timeframe=rm_timeframe, limit=rm_limit)
        rm_df = compute_return_series(rm_df)
        ppy = periods_map[rm_timeframe]
        sr = sharpe_ratio(rm_df["ret"], ppy)
        so = sortino_ratio(rm_df["ret"], ppy)
        mdd = max_drawdown(rm_df["close"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe Ratio", f"{sr:.2f}")
        c2.metric("Sortino Ratio", f"{so:.2f}")
        c3.metric("Max Drawdown", f"{mdd*100:.1f}%")
    except Exception as e:
        st.error(f"Risk metrics failed: {e}")

st.divider()
st.caption("Data: Binance via ccxt | Real-time microstructure analysis | Not financial advice")

if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()
