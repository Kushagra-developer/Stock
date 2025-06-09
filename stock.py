import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import pandas as pd
import pandas_ta as ta # Import pandas-ta
import os
os.system("pip install plotly")
# --- Page Configuration ---
st.set_page_config(layout="wide")
st.title("📈 NIFTY 100 - Intraday Chart + Historical Data with Technical Indicators")

# --- Stock Data ---
nifty_100 = {
    "ADANIENT": "ADANIENT.NS", "ADANIGREEN": "ADANIGREEN.NS", "ADANIPORTS": "ADANIPORTS.NS",
    "APOLLOHOSP": "APOLLOHOSP.NS", "ASIANPAINT": "ASIANPAINT.NS", "AXISBANK": "AXISBANK.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS", "BAJFINANCE": "BAJFINANCE.NS", "BAJAJFINSV": "BAJAJFINSV.NS",
    "BPCL": "BPCL.NS", "BHARTIARTL": "BHARTIARTL.NS", "BRITANNIA": "BRITANNIA.NS",
    "CIPLA": "CIPLA.NS", "COALINDIA": "COALINDIA.NS", "DIVISLAB": "DIVISLAB.NS",
    "DRREDDY": "DRREDDY.NS", "EICHERMOT": "EICHERMOT.NS", "GRASIM": "GRASIM.NS",
    "HCLTECH": "HCLTECH.NS", "HDFCBANK": "HDFCBANK.NS", "HDFCLIFE": "HDFCLIFE.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS", "HINDALCO": "HINDALCO.NS", "HINDUNILVR": "HINDUNILVR.NS",
    "ICICIBANK": "ICICIBANK.NS", "ITC": "ITC.NS", "INDUSINDBK": "INDUSINDBK.NS",
    "INFY": "INFY.NS", "JSWSTEEL": "JSWSTEEL.NS", "KOTAKBANK": "KOTAKBANK.NS",
    "LT": "LT.NS", "M&M": "M&M.NS", "MARUTI": "MARUTI.NS", "NTPC": "NTPC.NS",
    "NESTLEIND": "NESTLEIND.NS", "ONGC": "ONGC.NS", "POWERGRID": "POWERGRID.NS",
    "RELIANCE": "RELIANCE.NS", "SBILIFE": "SBILIFE.NS", "SBIN": "SBIN.NS",
    "SUNPHARMA": "SUNPHARMA.NS", "TCS": "TCS.NS", "TATACONSUM": "TATACONSUM.NS",
    "TATAMOTORS": "TATAMOTORS.NS", "TATASTEEL": "TATASTEEL.NS", "TECHM": "TECHM.NS",
    "TITAN": "TITAN.NS", "ULTRACEMCO": "ULTRACEMCO.NS", "UPL": "UPL.NS", "WIPRO": "WIPRO.NS"
    # Note: Some tickers from the original list were outdated or delisted.
    # I've updated the list to be more current with the NIFTY 100 components.
}


# --- Sidebar for User Input ---
st.sidebar.header("Controls")
selected_company = st.sidebar.selectbox("Select Company", list(nifty_100.keys()))
ticker = nifty_100[selected_company]

end_date = datetime.today()
start_10yr = end_date - timedelta(days=365 * 10)

# --- Data Fetching and Caching ---
@st.cache_data(ttl=3600)
def fetch_daily_data(ticker, start, end):
    """Fetches daily historical data."""
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

@st.cache_data(ttl=300)
def fetch_intraday_data(ticker):
    """Fetches 1-minute intraday data for the current day."""
    df = yf.download(ticker, period="1d", interval="1m")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

# --- Indicator and Signal Calculation ---
def add_indicators_and_signals(df):
    """Calculates technical indicators and generates trading signals."""
    if df.empty:
        return df

    # Calculate Indicators using pandas_ta
    df.ta.supertrend(append=True, length=7, multiplier=3)
    df.ta.macd(append=True, fast=12, slow=26, signal=9)
    df.ta.rsi(append=True, length=14)
    df.ta.aroon(append=True, length=14)

    # --- Generate Buy/Sell Signals (1 for Buy, -1 for Sell, 0 for Hold) ---

    # Supertrend Signal
    st_col = 'SUPERTd_7_3.0'
    df['ST_Signal'] = 0
    if st_col in df.columns:
        # Buy signal: Supertrend flips from -1 to 1
        df.loc[(df[st_col] == 1) & (df[st_col].shift(1) == -1), 'ST_Signal'] = 1
        # Sell signal: Supertrend flips from 1 to -1
        df.loc[(df[st_col] == -1) & (df[st_col].shift(1) == 1), 'ST_Signal'] = -1

    # MACD Signal (Crossover)
    macd_col = 'MACD_12_26_9'
    macds_col = 'MACDs_12_26_9'
    df['MACD_Signal'] = 0
    if macd_col in df.columns and macds_col in df.columns:
        # Buy signal: MACD crosses above MACD_Signal
        df.loc[(df[macd_col] > df[macds_col]) & (df[macd_col].shift(1) <= df[macds_col].shift(1)), 'MACD_Signal'] = 1
        # Sell signal: MACD crosses below MACD_Signal
        df.loc[(df[macd_col] < df[macds_col]) & (df[macd_col].shift(1) >= df[macds_col].shift(1)), 'MACD_Signal'] = -1

    # RSI Signal (Overbought/Oversold)
    rsi_col = 'RSI_14'
    df['RSI_Signal'] = 0
    if rsi_col in df.columns:
        # Buy signal: RSI crosses above 30 (oversold to neutral)
        df.loc[(df[rsi_col] > 30) & (df[rsi_col].shift(1) <= 30), 'RSI_Signal'] = 1
        # Sell signal: RSI crosses below 70 (overbought to neutral)
        df.loc[(df[rsi_col] < 70) & (df[rsi_col].shift(1) >= 70), 'RSI_Signal'] = -1

    # Aroon Signal (Crossover)
    aroonu_col = 'AROONU_14'
    aroond_col = 'AROOND_14'
    df['Aroon_Signal'] = 0
    if aroonu_col in df.columns and aroond_col in df.columns:
        # Buy signal: AroonUp crosses above AroonDown
        df.loc[(df[aroonu_col] > df[aroond_col]) & (df[aroonu_col].shift(1) <= df[aroond_col].shift(1)), 'Aroon_Signal'] = 1
        # Sell signal: AroonDown crosses above AroonUp
        df.loc[(df[aroonu_col] < df[aroond_col]) & (df[aroonu_col].shift(1) >= df[aroond_col].shift(1)), 'Aroon_Signal'] = -1

    # --- Consolidate Signals ---
    df['Final_Signal_Score'] = df['ST_Signal'] + df['MACD_Signal'] + df['RSI_Signal'] + df['Aroon_Signal']

    def map_signal(score):
        if score > 0:
            return 'BUY'
        elif score < 0:
            return 'SELL'
        else:
            return 'HOLD'

    df['Final_Signal'] = df['Final_Signal_Score'].apply(map_signal)

    return df

# --- Main App Logic ---
df_daily = fetch_daily_data(ticker, start_10yr, end_date)
df_intraday = fetch_intraday_data(ticker)

# Add indicators to the daily data
df_daily_with_indicators = add_indicators_and_signals(df_daily.copy())

# Add indicators to the intraday data
df_intraday_with_indicators = add_indicators_and_signals(df_intraday.copy())

# --- Intraday Chart Section ---
india_tz = pytz.timezone('Asia/Kolkata')
today_ist = datetime.now(india_tz).date()

if not df_intraday_with_indicators.empty:
    if df_intraday_with_indicators.index.tz is not None:
        df_intraday_with_indicators.index = df_intraday_with_indicators.index.tz_convert('Asia/Kolkata')
    else:
        df_intraday_with_indicators.index = df_intraday_with_indicators.index.tz_localize('UTC').tz_convert('Asia/Kolkata')

    df_today = df_intraday_with_indicators[df_intraday_with_indicators.index.date == today_ist]
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    intraday_data_valid = (not df_today.empty) and all(col in df_today.columns for col in required_cols)

    if intraday_data_valid:
        st.subheader(f"📊 Intraday 1-Minute Candlestick Chart for {selected_company} ({today_ist})")

        def plot_today_candle(df, ticker_name):
            fig = go.Figure(data=[go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                increasing_line_color='#26A69A', decreasing_line_color='#EF5350'
            )])
            fig.update_layout(
                title=f"Live Intraday Data for {ticker_name}",
                xaxis_rangeslider_visible=False,
                height=500,
                template="plotly_dark"
            )
            return fig

        st.plotly_chart(plot_today_candle(df_today, selected_company), use_container_width=True)

    else:
        st.warning("Intraday data not available or market is closed. Showing only historical data.")
else:
    st.warning("Could not fetch intraday data. The market may be closed or there might be a data issue.")


# --- Intraday Technical Data Table ---
st.subheader(f"📋 Intraday 1-Minute Technical Data & Signals for {selected_company} ({today_ist})")
st.info("This table includes 1-minute intraday price data along with Supertrend, MACD, RSI, and Aroon indicators and their respective signals.")

if not df_intraday_with_indicators.empty:
    # Select columns to display for intraday data
    intraday_display_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SUPERT_7_3.0', 'MACD_12_26_9', 'RSI_14', 'AROONU_14', 'AROOND_14',
        'Final_Signal'
    ]
    # Filter out columns that might not exist
    intraday_display_cols = [col for col in intraday_display_cols if col in df_intraday_with_indicators.columns]

    if intraday_display_cols:
        st.dataframe(df_intraday_with_indicators[intraday_display_cols].sort_index(ascending=False).round(2), use_container_width=True)
    else:
        st.warning("Could not generate intraday indicator data.")
else:
    st.warning("No intraday data to display technical indicators for.")


# --- Historical Data and Indicators Table ---
st.subheader(f"📋 Last 10 Years Daily Historical Data & Signals for {selected_company}")
st.info("This table includes daily price data along with Supertrend, MACD, RSI, and Aroon indicators. 'BUY'/'SELL' signals are generated based on crossovers and threshold breaches.")

# Select columns to display for clarity
display_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SUPERT_7_3.0', 'MACD_12_26_9', 'RSI_14', 'AROONU_14', 'AROOND_14',
    'Final_Signal'
]
# Filter out columns that might not exist if data is insufficient
display_cols = [col for col in display_cols if col in df_daily_with_indicators.columns]

# Displaying the dataframe with the new indicator and signal columns
if display_cols:
    st.dataframe(df_daily_with_indicators[display_cols].sort_index(ascending=False).round(2), use_container_width=True)
else:
    st.warning("Could not generate indicator data.")


# --- Detailed View of Most Recent Signals ---
st.subheader("🔍 Most Recent Daily Indicator Signals")
recent_daily_data = df_daily_with_indicators.tail(10).sort_index(ascending=False)

# Columns containing the raw signals (1, -1, 0) and the final text signal
signal_cols = ['ST_Signal', 'MACD_Signal', 'RSI_Signal', 'Aroon_Signal', 'Final_Signal']
# Filter for columns that actually exist in the dataframe
display_signal_cols = [col for col in signal_cols if col in recent_daily_data.columns]

if display_signal_cols:
    # Create a new DataFrame for display to avoid modifying the original data
    display_df = recent_daily_data[display_signal_cols].copy()

    # Define the mapping from numeric signal to text
    signal_map = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}

    # Explicitly map the numeric signal columns to their text representation
    for col in ['ST_Signal', 'MACD_Signal', 'RSI_Signal', 'Aroon_Signal']:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(signal_map).fillna('HOLD')

    # Rename columns for a more user-friendly presentation
    display_df.rename(columns={
        'ST_Signal': 'SuperTrend',
        'MACD_Signal': 'MACD Crossover',
        'RSI_Signal': 'RSI Signal',
        'Aroon_Signal': 'Aroon Crossover',
        'Final_Signal': 'Consolidated Signal'
    }, inplace=True)
    
    st.dataframe(display_df, use_container_width=True)
else:
    st.warning("Could not generate recent signal data.")

# --- Detailed View of Most Recent Intraday Signals ---
st.subheader("🔍 Most Recent Intraday Indicator Signals")
recent_intraday_data = df_intraday_with_indicators.tail(10).sort_index(ascending=False)

# Columns containing the raw signals (1, -1, 0) and the final text signal
intraday_signal_cols = ['ST_Signal', 'MACD_Signal', 'RSI_Signal', 'Aroon_Signal', 'Final_Signal']
# Filter for columns that actually exist in the dataframe
display_intraday_signal_cols = [col for col in intraday_signal_cols if col in recent_intraday_data.columns]

if display_intraday_signal_cols:
    # Create a new DataFrame for display to avoid modifying the original data
    display_intraday_df = recent_intraday_data[display_intraday_signal_cols].copy()

    # Define the mapping from numeric signal to text
    signal_map = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}

    # Explicitly map the numeric signal columns to their text representation
    for col in ['ST_Signal', 'MACD_Signal', 'RSI_Signal', 'Aroon_Signal']:
        if col in display_intraday_df.columns:
            display_intraday_df[col] = display_intraday_df[col].map(signal_map).fillna('HOLD')

    # Rename columns for a more user-friendly presentation
    display_intraday_df.rename(columns={
        'ST_Signal': 'SuperTrend',
        'MACD_Signal': 'MACD Crossover',
        'RSI_Signal': 'RSI Signal',
        'Aroon_Signal': 'Aroon Crossover',
        'Final_Signal': 'Consolidated Signal'
    }, inplace=True)
    
    st.dataframe(display_intraday_df, use_container_width=True)
else:
    st.warning("Could not generate recent intraday signal data.")
