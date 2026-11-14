import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from scipy.stats import norm

st.set_page_config(page_title="AI Market Risk Engine", layout="wide")
st.title("📉 AI/ML Market Risk Management System")

# --- SIDEBAR: INTERACTIVE USER INPUTS ---
st.sidebar.header("1. Market & Portfolio Inputs")
market = st.sidebar.selectbox("Select Exchange", ["India (NSE)", "India (BSE)", "US / Global"])
raw_ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE, TCS, AAPL)", "RELIANCE").strip().upper()

# Automatically attach .NS or .BO for Indian markets
if market == "India (NSE)" and not raw_ticker.endswith(".NS") and not raw_ticker.startswith("^"):
    ticker = f"{raw_ticker}.NS"
elif market == "India (BSE)" and not raw_ticker.endswith(".BO") and not raw_ticker.startswith("^"):
    ticker = f"{raw_ticker}.BO"
else:
    ticker = raw_ticker

currency = "₹" if "India" in market else "$"
default_val = 1000000 if "India" in market else 100000

portfolio_val = st.sidebar.number_input(f"Portfolio Value ({currency})", min_value=1000, value=default_val, step=10000)
confidence = st.sidebar.slider("VaR Confidence Level (%)", 90, 99, 95)

st.sidebar.header("2. Stress Test Scenario")
market_shock = st.sidebar.slider("Simulate Market Crash (%)", -50, 0, -15)
vol_multiplier = st.sidebar.slider("Volatility Spike Multiplier", 1.0, 4.0, 2.0)

# --- LOAD & CLEAN DATA ---
@st.cache_data(ttl=3600)
def load_data(symbol):
    df = yf.download(symbol, period="5y", progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Return'] = df['Close'].pct_change()
    df['Volatility_21d'] = df['Return'].rolling(21).std()
    # Replace 0 volume with NaN before pct_change to prevent infinity errors on Indian stocks
    df['Vol_Change'] = df['Volume'].replace(0, np.nan).pct_change().fillna(0)
    return df.dropna(subset=['Return', 'Volatility_21d'])

df = load_data(ticker)

if df.empty:
    st.error(f"Could not fetch data for '{ticker}'. Check the exact NSE/BSE symbol (e.g., HDFCBANK, INFY, TATAMOTORS).")
    st.stop()

# --- 1. ML VOLATILITY FORECASTING (Random Forest) ---
df['Target_Vol'] = df['Volatility_21d'].shift(-1)
train_df = df.dropna(subset=['Target_Vol'])

features = ['Return', 'Volatility_21d', 'Vol_Change']
X = train_df[features].replace([np.inf, -np.inf], 0)
y = train_df['Target_Vol']

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

latest_features = df[features].iloc[[-1]].replace([np.inf, -np.inf], 0)
predicted_daily_vol = rf_model.predict(latest_features)[0]
annualized_vol = predicted_daily_vol * np.sqrt(252)

# --- 2. VALUE AT RISK (VaR) CALCULATION ---
z_score = norm.ppf(1 - (confidence / 100))
var_pct = abs(z_score * predicted_daily_vol)
var_amount = portfolio_val * var_pct

# --- 3. ANOMALY DETECTION (Isolation Forest) ---
iso = IsolationForest(contamination=0.02, random_state=42)
df['Anomaly'] = iso.fit_predict(df[['Return', 'Volatility_21d']])
anomalies = df[df['Anomaly'] == -1]

# --- 4. STRESS TEST CALCULATION ---
stressed_var = var_amount * vol_multiplier
stressed_loss = (portfolio_val * abs(market_shock / 100)) + stressed_var

# --- DISPLAY METRICS ---
st.caption(f"Showing live risk analytics for **{ticker}** | Latest Close: **{currency}{df['Close'].iloc[-1]:,.2f}**")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted Annual Volatility", f"{annualized_vol:.2%}")
col2.metric(f"1-Day {confidence}% VaR (Max Loss)", f"{currency}{var_amount:,.2f}", f"-{var_pct:.2%}")
col3.metric("Abnormal Days Detected (5Y)", f"{len(anomalies)} days")
col4.metric("Stress Test Projected Loss", f"{currency}{stressed_loss:,.2f}", f"{market_shock}% Shock")

# --- INTERACTIVE CHARTS ---
st.subheader(f"Abnormal Market Movements Detected: {ticker}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig.add_trace(go.Scatter(
    x=anomalies.index, y=anomalies['Close'], mode='markers',
    marker=dict(color='red', size=8), name='ML Anomaly Flag'
))
fig.update_layout(yaxis_title=f"Price ({currency})", xaxis_title="Date")
st.plotly_chart(fig, use_container_width=True)
