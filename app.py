import streamlit as st
import pandas as pd
import glob
import altair as alt
import os
import time
import numpy as np

# --------- RISK MANAGEMENT FUNCTIONS ---------
def calculate_risk_management(entry_price, position_type):
    """Calculate take profit and stop loss based on position type"""
    if pd.isna(entry_price) or entry_price <= 0:
        return None, None, None, None
    
    if position_type == 'BUY':
        take_profit = entry_price * 1.05 
        stop_loss = entry_price * 0.98   
        profit_pct = 5.0
        loss_pct = 2.0
        risk_reward_ratio = profit_pct / loss_pct  
    elif position_type == 'SELL':
        take_profit = entry_price * 0.99 
        stop_loss = entry_price * 1.07   
        profit_pct = 1.0
        loss_pct = 7.0
        risk_reward_ratio = profit_pct / loss_pct  
    else:
        return None, None, None, None
    
    return take_profit, stop_loss, risk_reward_ratio, {'profit_pct': profit_pct, 'loss_pct': loss_pct}

def calculate_position_size(account_balance, risk_per_trade, entry_price, stop_loss):
    """Calculate position size based on risk management"""
    if pd.isna(entry_price) or pd.isna(stop_loss) or entry_price <= 0 or stop_loss <= 0:
        return None
    
    risk_amount = account_balance * (risk_per_trade / 100)
    price_diff = abs(entry_price - stop_loss)
    
    if price_diff == 0:
        return None
    
    position_size = risk_amount / price_diff
    return round(position_size, 2)

# --------- COLUMN CLEANING FUNCTION ---------
def clean_columns(df):
    """Clean duplicate column names"""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_idx):
            if i == 0:
                continue
            cols[idx] = f"{dup}_{i}"
    df.columns = cols
    return df

# --------- ENHANCED SIGNAL GENERATION FUNCTION ---------
def generate_signals_with_risk_management(df, account_balance=100000, risk_per_trade=2):
    """Generate buy/sell signals with risk management calculations"""
    df = df.copy()
    
    # Map column names to standard format
    close_col = None
    for col in ['close', 'Close', 'LTP', 'ltp']:
        if col in df.columns:
            close_col = col
            break
    
    if close_col is None:
        st.error("No price column found. Expected 'close', 'Close', 'LTP', or 'ltp'")
        return df
    
    # Generate technical signals
    # MACD Signals
    macd_cols = [col for col in df.columns if 'macd' in col.lower() and 'signal' not in col.lower()]
    signal_cols = [col for col in df.columns if 'macd_signal' in col.lower() or 'signal' in col.lower()]
    
    if macd_cols and signal_cols:
        macd_col = macd_cols[0]
        signal_col = signal_cols[0]
        df["MACD_buy"] = (df[macd_col] > df[signal_col]) & (df[macd_col].shift(1) <= df[signal_col].shift(1))
        df["MACD_sell"] = (df[macd_col] < df[signal_col]) & (df[macd_col].shift(1) >= df[signal_col].shift(1))

    # Aroon Signals
    aroon_up_cols = [col for col in df.columns if 'aroon_up' in col.lower()]
    aroon_down_cols = [col for col in df.columns if 'aroon_down' in col.lower()]
    
    if aroon_up_cols and aroon_down_cols:
        aroon_up_col = aroon_up_cols[0]
        aroon_down_col = aroon_down_cols[0]
        df["AROON_buy"] = (df[aroon_up_col] > 70) & (df[aroon_down_col] < 30)
        df["AROON_sell"] = (df[aroon_up_col] < 30) & (df[aroon_down_col] > 70)

    # EMA Crossover Signals
    ema_9_cols = [col for col in df.columns if 'ema_9' in col.lower()]
    ema_21_cols = [col for col in df.columns if 'ema_21' in col.lower()]
    
    if ema_9_cols and ema_21_cols:
        ema_9_col = ema_9_cols[0]
        ema_21_col = ema_21_cols[0]
        df["EMA_buy"] = (df[ema_9_col] > df[ema_21_col]) & (df[ema_9_col].shift(1) <= df[ema_21_col].shift(1))
        df["EMA_sell"] = (df[ema_9_col] < df[ema_21_col]) & (df[ema_9_col].shift(1) >= df[ema_21_col].shift(1))

    # Bollinger Bands Signals
    bb_upper_cols = [col for col in df.columns if 'bb_bbh' in col.lower() or 'bb_upper' in col.lower()]
    bb_lower_cols = [col for col in df.columns if 'bb_bbl' in col.lower() or 'bb_lower' in col.lower()]
    
    if bb_upper_cols and bb_lower_cols:
        bb_upper_col = bb_upper_cols[0]
        bb_lower_col = bb_lower_cols[0]
        df["BB_buy"] = df[close_col] < df[bb_lower_col]  # Oversold
        df["BB_sell"] = df[close_col] > df[bb_upper_col]  # Overbought

    # Parabolic SAR Signals
    psar_cols = [col for col in df.columns if 'psar' in col.lower() and 'up' not in col.lower() and 'down' not in col.lower()]
    
    if psar_cols:
        psar_col = psar_cols[0]
        df["PSAR_buy"] = (df[close_col] > df[psar_col]) & (df[close_col].shift(1) <= df[psar_col].shift(1))
        df["PSAR_sell"] = (df[close_col] < df[psar_col]) & (df[close_col].shift(1) >= df[psar_col].shift(1))

    # Determine overall position type for each row
    def determine_position_type(row):
        buy_signals = []
        sell_signals = []
        
        # Collect all buy/sell signals
        for col in df.columns:
            if '_buy' in col.lower() and row.get(col, False):
                buy_signals.append(col)
            elif '_sell' in col.lower() and row.get(col, False):
                sell_signals.append(col)
        
        if len(buy_signals) > len(sell_signals):
            return 'BUY'
        elif len(sell_signals) > len(buy_signals):
            return 'SELL'
        else:
            return 'HOLD'

    # Apply position type determination
    df['Position_Type'] = df.apply(determine_position_type, axis=1)
    
    # Calculate risk management for each row
    def calc_risk_management(row):
        entry_price = row[close_col]
        position_type = row['Position_Type']
        
        if position_type == 'HOLD':
            return pd.Series([None, None, None, None, None, None])
        
        take_profit, stop_loss, risk_reward, pct_info = calculate_risk_management(entry_price, position_type)
        position_size = calculate_position_size(account_balance, risk_per_trade, entry_price, stop_loss)
        
        return pd.Series([
            take_profit, stop_loss, risk_reward, 
            pct_info['profit_pct'] if pct_info else None,
            pct_info['loss_pct'] if pct_info else None,
            position_size
        ])

    df[['Take_Profit', 'Stop_Loss', 'Risk_Reward_Ratio', 'Profit_Pct', 'Loss_Pct', 'Position_Size']] = df.apply(calc_risk_management, axis=1)
    
    return df

# --------- LOAD DATA FUNCTION ---------
@st.cache_data
def load_data(folder_path):
    """Load and process all CSV files from the technical analysis results"""
    all_dfs = []
    for file in glob.glob(os.path.join(folder_path, "*.csv")):
        try:
            df = pd.read_csv(file)
            df = clean_columns(df)
            
            # Extract symbol from filename if not in data
            if 'Symbol' not in df.columns:
                symbol = os.path.basename(file).replace('_technical_analysis.csv', '').replace('.csv', '')
                df['Symbol'] = symbol
            
            all_dfs.append(df)
        except Exception as e:
            st.warning(f"Error loading {file}: {e}")
    
    if len(all_dfs) == 0:
        st.error("No CSV files loaded. Please check your data folder and files.")
        st.stop()
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

# --------- RISK MANAGEMENT SUMMARY FUNCTION ---------
def create_risk_summary(df):
    """Create a summary of risk management metrics for all stocks"""
    summary_data = []
    
    for symbol in df['Symbol'].unique():
        symbol_data = df[df['Symbol'] == symbol]
        
        # Get latest data
        latest_row = symbol_data.iloc[-1] if len(symbol_data) > 0 else None
        
        if latest_row is not None:
            close_col = None
            for col in ['close', 'Close', 'LTP', 'ltp']:
                if col in latest_row.index:
                    close_col = col
                    break
            
            summary_data.append({
                'Symbol': symbol,
                'Current_Price': latest_row[close_col] if close_col else 'N/A',
                'Position_Type': latest_row.get('Position_Type', 'HOLD'),
                'Take_Profit': latest_row.get('Take_Profit', 'N/A'),
                'Stop_Loss': latest_row.get('Stop_Loss', 'N/A'),
                'Risk_Reward': latest_row.get('Risk_Reward_Ratio', 'N/A'),
                'Position_Size': latest_row.get('Position_Size', 'N/A')
            })
    
    return pd.DataFrame(summary_data)

DATA_FOLDER = "./technical_analysis_results"

# --------- PAGE CONFIG ---------
st.set_page_config(
    page_title="NIFTY 100 Risk Management Dashboard",
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
.risk-metric { background-color: #E8F6F3; padding: 10px; border-radius: 5px; margin: 5px 0; }
.buy-signal { background-color: #D5F4E6; color: #27AE60; font-weight: bold; padding: 5px; border-radius: 3px; }
.sell-signal { background-color: #FADBD8; color: #E74C3C; font-weight: bold; padding: 5px; border-radius: 3px; }
.hold-signal { background-color: #F8F9FA; color: #6C757D; font-weight: bold; padding: 5px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 NIFTY 100 Risk Management Dashboard")

# --------- SIDEBAR RISK PARAMETERS ---------
st.sidebar.header("🎯 Risk Management Settings")
account_balance = st.sidebar.number_input("Account Balance (₹)", min_value=1000, value=100000, step=1000)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)

# --------- BLINKING UPDATE INDICATOR ---------
BLINK_INTERVAL = 10
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
    '>🔄 Live Risk Management Updates...</div>
    <style>
    @keyframes blinker {
      50% { opacity: 0; }
    }
    </style>
    """, unsafe_allow_html=True)

# --------- LOAD AND PROCESS DATA ---------
with st.spinner("Loading NIFTY 100 technical analysis data..."):
    data = load_data(DATA_FOLDER)
    data = generate_signals_with_risk_management(data, account_balance, risk_per_trade)

# --------- COMPANY SELECTION ---------
companies = sorted(data['Symbol'].dropna().unique())
selected_company = st.sidebar.selectbox("Select Company", options=companies)

# --------- RISK MANAGEMENT SUMMARY IN SIDEBAR ---------
st.sidebar.subheader("📊 Risk Management Summary")
risk_summary = create_risk_summary(data)

# Filter for active positions only
active_positions = risk_summary[risk_summary['Position_Type'].isin(['BUY', 'SELL'])]

if not active_positions.empty:
    st.sidebar.dataframe(
        active_positions.style.map(
            lambda x: 'background-color: #D5F4E6; color: #27AE60; font-weight: bold' if x == 'BUY' else
                      'background-color: #FADBD8; color: #E74C3C; font-weight: bold' if x == 'SELL' else
                      'color: gray'
        ), 
        use_container_width=True,
        height=300
    )
else:
    st.sidebar.info("No active trading positions currently")

# --------- MAIN CONTENT ---------
filtered_data = data[data['Symbol'] == selected_company]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(filtered_data))

with col2:
    if not filtered_data.empty:
        latest_position = filtered_data.iloc[-1]['Position_Type']
        if latest_position == 'BUY':
            st.markdown('<div class="buy-signal">BUY SIGNAL</div>', unsafe_allow_html=True)
        elif latest_position == 'SELL':
            st.markdown('<div class="sell-signal">SELL SIGNAL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="hold-signal">HOLD</div>', unsafe_allow_html=True)

with col3:
    if not filtered_data.empty:
        latest_risk_reward = filtered_data.iloc[-1].get('Risk_Reward_Ratio', 'N/A')
        st.metric("Risk:Reward Ratio", f"1:{latest_risk_reward:.2f}" if pd.notna(latest_risk_reward) else "N/A")

st.subheader(f"📈 {selected_company} - Technical Analysis & Risk Management")

if filtered_data.empty:
    st.warning("No data available for the selected company.")
else:
    # --------- RISK MANAGEMENT METRICS ---------
    latest_row = filtered_data.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        close_col = None
        for col in ['close', 'Close', 'LTP', 'ltp']:
            if col in latest_row.index:
                close_col = col
                break
        current_price = latest_row[close_col] if close_col else 'N/A'
        st.metric("Current Price", f"₹{current_price:.2f}" if pd.notna(current_price) else "N/A")
    
    with col2:
        take_profit = latest_row.get('Take_Profit', None)
        st.metric("Take Profit", f"₹{take_profit:.2f}" if pd.notna(take_profit) else "N/A")
    
    with col3:
        stop_loss = latest_row.get('Stop_Loss', None)
        st.metric("Stop Loss", f"₹{stop_loss:.2f}" if pd.notna(stop_loss) else "N/A")
    
    with col4:
        position_size = latest_row.get('Position_Size', None)
        st.metric("Position Size", f"{position_size:.0f} shares" if pd.notna(position_size) else "N/A")

    # --------- DATA TABLE ---------
    st.subheader("📋 Detailed Analysis")
    
    # Display key columns
    display_columns = ['Symbol', 'Position_Type', 'Take_Profit', 'Stop_Loss', 'Risk_Reward_Ratio', 'Position_Size']
    
    # Add close price column
    for col in ['close', 'Close', 'LTP', 'ltp']:
        if col in filtered_data.columns:
            display_columns.insert(1, col)
            break
    
    # Add technical indicator columns
    tech_columns = [col for col in filtered_data.columns if any(indicator in col.lower() for indicator in ['macd', 'aroon', 'ema', 'bb_', 'psar'])]
    display_columns.extend(tech_columns[:5])  # Limit to first 5 technical columns
    
    available_columns = [col for col in display_columns if col in filtered_data.columns]
    
    st.dataframe(
        filtered_data[available_columns].style.map(
            lambda x: 'background-color: #D5F4E6; color: #27AE60; font-weight: bold' if x == 'BUY' else
                      'background-color: #FADBD8; color: #E74C3C; font-weight: bold' if x == 'SELL' else
                      'background-color: #F8F9FA; color: #6C757D' if x == 'HOLD' else ''
        ),
        use_container_width=True
    )

    # --------- PRICE CHART WITH RISK LEVELS ---------
    st.subheader("📊 Price Chart with Risk Management Levels")
    
    chart_data = filtered_data.reset_index()
    
    # Base price chart
    base = alt.Chart(chart_data).add_params(
        alt.selection_interval()
    ).properties(width=800, height=400)
    
    # Price line
    price_line = base.mark_line(strokeWidth=2, color='blue').encode(
        x=alt.X('index:O', title='Time Period'),
        y=alt.Y(f'{close_col}:Q', title='Price (₹)'),
        tooltip=[alt.Tooltip('Symbol'), alt.Tooltip(f'{close_col}:Q', title='Price')]
    )
    
    # Take profit line
    tp_line = base.mark_line(strokeWidth=1, color='green', strokeDash=[5,5]).encode(
        x='index:O',
        y=alt.Y('Take_Profit:Q'),
        tooltip=[alt.Tooltip('Take_Profit:Q', title='Take Profit')]
    )
    
    # Stop loss line
    sl_line = base.mark_line(strokeWidth=1, color='red', strokeDash=[5,5]).encode(
        x='index:O',
        y=alt.Y('Stop_Loss:Q'),
        tooltip=[alt.Tooltip('Stop_Loss:Q', title='Stop Loss')]
    )
    
    # Buy/Sell signals
    buy_points = base.mark_circle(size=100, color='green', opacity=0.8).encode(
        x='index:O',
        y=f'{close_col}:Q',
        tooltip=['Symbol', 'Position_Type']
    ).transform_filter(
        alt.datum.Position_Type == 'BUY'
    )
    
    sell_points = base.mark_circle(size=100, color='red', opacity=0.8).encode(
        x='index:O',
        y=f'{close_col}:Q',
        tooltip=['Symbol', 'Position_Type']
    ).transform_filter(
        alt.datum.Position_Type == 'SELL'
    )
    
    # Combine all chart elements
    final_chart = alt.layer(price_line, tp_line, sl_line, buy_points, sell_points).resolve_scale(
        y='independent'
    ).properties(
        title=f"{selected_company} - Price with Risk Management Levels"
    )
    
    st.altair_chart(final_chart, use_container_width=True)

# --------- PORTFOLIO RISK SUMMARY ---------
st.subheader("💼 Portfolio Risk Summary")

portfolio_summary = risk_summary.copy()

# Calculate total portfolio metrics
total_positions = len(portfolio_summary[portfolio_summary['Position_Type'].isin(['BUY', 'SELL'])])
buy_positions = len(portfolio_summary[portfolio_summary['Position_Type'] == 'BUY'])
sell_positions = len(portfolio_summary[portfolio_summary['Position_Type'] == 'SELL'])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Active Positions", total_positions)

with col2:
    st.metric("BUY Positions", buy_positions)

with col3:
    st.metric("SELL Positions", sell_positions)

with col4:
    avg_risk_reward = portfolio_summary['Risk_Reward'].replace('N/A', np.nan).astype(float).mean()
    st.metric("Avg Risk:Reward", f"1:{avg_risk_reward:.2f}" if not np.isnan(avg_risk_reward) else "N/A")

# Display full portfolio summary
st.dataframe(
    portfolio_summary.style.map(
        lambda x: 'background-color: #D5F4E6; color: #27AE60; font-weight: bold' if x == 'BUY' else
                  'background-color: #FADBD8; color: #E74C3C; font-weight: bold' if x == 'SELL' else
                  'background-color: #F8F9FA; color: #6C757D' if x == 'HOLD' else ''
    ),
    use_container_width=True,
    height=400
)

# --------- MANUAL RISK CALCULATOR ---------
st.sidebar.subheader("🧮 Manual Risk Calculator")
manual_entry_price = st.sidebar.number_input("Entry Price (₹)", min_value=0.01, value=100.0, step=0.01)
manual_position_type = st.sidebar.selectbox("Position Type", ['BUY', 'SELL'])

manual_tp, manual_sl, manual_rr, manual_pct = calculate_risk_management(manual_entry_price, manual_position_type)
manual_pos_size = calculate_position_size(account_balance, risk_per_trade, manual_entry_price, manual_sl)

if manual_tp and manual_sl:
    st.sidebar.markdown("### 📊 Risk Calculation Results")
    st.sidebar.metric("Take Profit", f"₹{manual_tp:.2f}")
    st.sidebar.metric("Stop Loss", f"₹{manual_sl:.2f}")
    st.sidebar.metric("Risk:Reward", f"1:{manual_rr:.2f}")
    st.sidebar.metric("Position Size", f"{manual_pos_size:.0f} shares" if manual_pos_size else "N/A")

# --------- FOOTER ---------
st.markdown("---")
st.markdown("💡 **Risk Management Logic**: BUY positions target 5% profit with 2% stop loss. SELL positions target 1% profit with 7% stop loss.")
st.markdown("⚠️ **Disclaimer**: This is for educational purposes only. Always consult with a financial advisor before making investment decisions.")
