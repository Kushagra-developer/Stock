import os
import pandas as pd
import glob

FOLDER_PATH = './technical_analysis_results'  # Change this to your actual folder

def calculate_signal(condition):
    return "BUY" if condition else "SELL"

def generate_summary(file_path):
    stock_name = os.path.basename(file_path).split('_')[0]  # Assuming format is "RELIANCE_technical_analysis.csv"
    df = pd.read_csv(file_path)
    latest = df.iloc[-1]  # Use the most recent row

    summary = {
        "Name of the Stocks": stock_name
    }

    if {'macd', 'macd_signal'}.issubset(df.columns):
        summary["MACD"] = calculate_signal(latest['macd'] > latest['macd_signal'])

    if {'aroon_up', 'aroon_down'}.issubset(df.columns):
        summary["Aroon"] = calculate_signal((latest['aroon_up'] > 70) and (latest['aroon_down'] < 30))

    if {'psar_up_indicator', 'psar_down_indicator'}.issubset(df.columns):
        summary["PSAR"] = calculate_signal(latest.get('psar_down_indicator', 0) == 1)

    if {'ema_9', 'ema_21', 'ema_50'}.issubset(df.columns):
        summary["EMA"] = calculate_signal((latest['ema_9'] > latest['ema_21']) and (latest['ema_21'] > latest['ema_50']))

    if {'Close', 'bb_bbl', 'bb_bbh'}.issubset(df.columns):
        summary["BB"] = calculate_signal(latest['Close'] < latest['bb_bbl'])

    return summary

def generate_all_summaries(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    summaries = []

    for file_path in csv_files:
        try:
            summary = generate_summary(file_path)
            summaries.append(summary)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return pd.DataFrame(summaries)

# Generate and save
summary_df = generate_all_summaries(FOLDER_PATH)
summary_df.to_csv("stock_signal_summary.csv", index=False)
print("✔ Summary CSV saved as 'stock_signal_summary.csv'")