import streamlit as st
import pandas as pd
import glob
import altair as alt
import os
import time

# --------- COLUMN CLEANING FUNCTION ---------
def clean_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_idx):
            if i == 0:
                continue
            cols[idx] = f"{dup}_{i}"
    df.columns = cols
    return df

# --------- SIGNAL GENERATION FUNCTION ---------
def generate_signals(df):
    df = df.copy()

    if "MACD" in df.columns and "Signal" in df.columns:
        df["MACD_buy"] = (df["MACD"] > df["Signal"]) & (df["MACD"].shift(1) <= df["Signal"].shift(1))
        df["MACD_sell"] = (df["MACD"] < df["Signal"]) & (df["MACD"].shift(1) >= df["Signal"].shift(1))

    if "RSI" in df.columns:
        df["RSI_buy"] = df["RSI"] < 30
        df["RSI_sell"] = df["RSI"] > 70

    if "Aroon Up" in df.columns and "Aroon Down" in df.columns:
        df["AROON_buy"] = (df["Aroon Up"] > df["Aroon Down"]) & (df["Aroon Up"].shift(1) <= df["Aroon Down"].shift(1))
        df["AROON_sell"] = (df["Aroon Up"] < df["Aroon Down"]) & (df["Aroon Up"].shift(1) >= df["Aroon Down"].shift(1))

    if "Supertrend" in df.columns and "Close" in df.columns:
        df["SUPER_buy"] = (df["Close"] > df["Supertrend"]) & (df["Close"].shift(1) <= df["Supertrend"].shift(1))
        df["SUPER_sell"] = (df["Close"] < df["Supertrend"]) & (df["Close"].shift(1) >= df["Supertrend"].shift(1))

    return df





# --- AUTO-REFRESH EVERY 30 SECONDS ---
import time

REFRESH_INTERVAL = 10  # seconds

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh >= REFRESH_INTERVAL:
    st.session_state.last_refresh = time.time()
    st.experimental_rerun()

# --------- LOAD DATA FUNCTION ---------
@st.cache_data
def load_data(folder_path):
    all_dfs = []
    for file in glob.glob(os.path.join(folder_path, "*.csv")):
        df = pd.read_csv(file)
        df = clean_columns(df)
        df = generate_signals(df)
        all_dfs.append(df)
    if not all_dfs:
        st.error("No CSV files loaded. Please check your data folder and files.")
        st.stop()
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

DATA_FOLDER = "./technical_analysis_results"

# --------- PAGE CONFIG ---------
st.set_page_config(
    page_title="Company Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# --------- CUSTOM CSS ---------
st.markdown("""
<style>
h1 { color: #2E86C1; font-weight: 700; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
[data-testid="stSidebar"] { background-color: #1F618D; color: white; padding: 20px; }
.css-1lcbmhc .css-1v0mbdj { color: white; background-color: #2874A6; border-radius: 6px; padding: 8px; }
.stDataFrame { border-radius: 8px; box-shadow: 0 0 12px rgb(46 134 193 / 0.4); padding: 15px; background: #F0F8FF; }
h2 { color: #2874A6; }
.stAlert > div { background-color: #F9E79F !important; color: #7D6608 !important; border-radius: 8px; padding: 10px; }
.stError > div { background-color: #F1948A !important; color: #641E16 !important; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Company Dashboard")

# --------- BLINKING UPDATE INDICATOR (every 30 seconds) ---------
BLINK_INTERVAL = 10  # seconds
if "last_blink" not in st.session_state:
    st.session_state.last_blink = time.time()
    st.session_state.blink_state = True

if time.time() - st.session_state.last_blink >= BLINK_INTERVAL:
    st.session_state.last_blink = time.time()
    st.session_state.blink_state = not st.session_state.blink_state

if st.session_state.blink_state:
    st.markdown("""
    <div style='
        animation: blinker 1s linear infinite;
        font-size: 18px;
        color: #FF5733;
        font-weight: bold;
        margin-bottom: 10px;
    '>🔄 Updating signals...</div>
    <style>
    @keyframes blinker {
      50% { opacity: 0; }
    }
    </style>
    """, unsafe_allow_html=True)

# --------- LOAD MAIN DATA ---------
data = load_data(DATA_FOLDER)

# --------- COMPANY DROPDOWN ---------
company_col = "Symbol"
if company_col not in data.columns:
    st.error(f"No '{company_col}' column found in your data.")
    st.stop()

companies = data[company_col].dropna().unique()
selected_company = st.sidebar.selectbox("Select Company", options=sorted(companies))

# --------- DISPLAY EXTERNAL SUMMARY CSV IN SIDEBAR ---------
external_summary_path = "./stock_signal_summary.csv"
st.sidebar.subheader("📁 Buy/Sell Summary")
try:
    external_summary_df = pd.read_csv(external_summary_path)
    st.sidebar.dataframe(external_summary_df.style.applymap(
        lambda x: 'color: green; font-weight: bold' if x == 'BUY' else
                  'color: red; font-weight: bold' if x == 'SELL' else
                  'color: gray; font-style: italic'
    ), use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning(f"File not found at `{external_summary_path}`")

# --------- MAIN CONTENT ---------
filtered_data = data[data[company_col] == selected_company]

st.subheader(f"Data for {selected_company}")
st.write(f"Total records: **{len(filtered_data)}**")

if filtered_data.empty:
    st.warning("No data available for the selected company.")
else:
    st.dataframe(filtered_data)

    # Automatically find and visualize any _buy/_sell pair
    buy_cols = [col for col in filtered_data.columns if '_buy' in col.lower()]
    sell_cols = [col for col in filtered_data.columns if '_sell' in col.lower()]

    for buy_col in buy_cols:
        indicator = buy_col.split('_')[0]
        sell_col = f"{indicator}_sell"
        if sell_col not in filtered_data.columns:
            continue

        base = alt.Chart(filtered_data.reset_index()).encode(
            x=alt.X('index', title='Record Index')
        )

        buy_line = base.mark_line(point=True, color='green').encode(
            y=alt.Y(buy_col, title="Buy Signal"),
            tooltip=[alt.Tooltip(buy_col, title="Buy")]
        )

        sell_line = base.mark_line(point=True, color='red').encode(
            y=alt.Y(sell_col, title="Sell Signal"),
            tooltip=[alt.Tooltip(sell_col, title="Sell")]
        )

        chart = alt.layer(buy_line, sell_line).resolve_scale(
            y='independent'
        ).properties(
            width=800,
            height=400,
            title=f"Buy/Sell Signals for {indicator} - {selected_company}"
        )

        st.altair_chart(chart, use_container_width=True)



# --------- MANUAL INDICATOR INPUT ---------
st.sidebar.subheader("Manual Indicator Input")
manual_macd = st.sidebar.number_input("MACD", value=0.0)
manual_macd_signal = st.sidebar.number_input("MACD Signal", value=0.0)
manual_aroon_up = st.sidebar.number_input("Aroon Up", min_value=0, max_value=100, value=50)
manual_aroon_down = st.sidebar.number_input("Aroon Down", min_value=0, max_value=100, value=50)
manual_psar_down = st.sidebar.checkbox("PSAR Down Indicator", value=False)
manual_ema_9 = st.sidebar.number_input("EMA 9", value=0.0)
manual_ema_21 = st.sidebar.number_input("EMA 21", value=0.0)
manual_ema_50 = st.sidebar.number_input("EMA 50", value=0.0)
manual_close = st.sidebar.number_input("Close", value=0.0)
manual_bb_bbl = st.sidebar.number_input("BB Lower Band", value=0.0)

# --------- GENERATE MANUAL SIGNAL SUMMARY ---------
def calculate_signal(condition):
    return "BUY" if condition else "SELL"

manual_summary = {
    "MACD": calculate_signal(manual_macd > manual_macd_signal),
    "Aroon": calculate_signal((manual_aroon_up > 70) and (manual_aroon_down < 30)),
    "PSAR": calculate_signal(manual_psar_down),
    "EMA": calculate_signal((manual_ema_9 > manual_ema_21) and (manual_ema_21 > manual_ema_50)),
    "BB": calculate_signal(manual_close < manual_bb_bbl)
}

st.sidebar.subheader("📌 Manual Signal Summary")
st.sidebar.dataframe(pd.DataFrame([manual_summary]), use_container_width=True)

