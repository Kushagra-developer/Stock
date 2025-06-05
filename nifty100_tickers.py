import requests
import pandas as pd
from nsetools import Nse
import time
from io import StringIO

def get_nifty100_comprehensive():
    print("🔍 Attempting to fetch NIFTY 100 stock tickers...")

    # Method 1: Try NSETools
    try:
        print("📊 Trying NSETools...")
        nse = Nse()
        stocks = nse.get_stock_quote_in_index("NIFTY 100")
        if stocks:
            tickers = [stock['symbol'] for stock in stocks]
            df = pd.DataFrame(stocks)
            print(f"✅ NSETools: Found {len(tickers)} stocks")
            return tickers, df, "NSETools"
    except Exception as e:
        print(f"❌ NSETools failed: {e}")

    # Method 2: Try official CSV download
    try:
        print("📄 Trying official CSV download...")
        csv_urls = [
            "https://www.nseindia.com/content/indices/ind_nifty100list.csv",
            "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
        ]
        headers = {'User-Agent': 'Mozilla/5.0'}
        for url in csv_urls:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                tickers = df['Symbol'].tolist() if 'Symbol' in df.columns else df.iloc[:, 0].tolist()
                print(f"✅ CSV Download: Found {len(tickers)} stocks")
                return tickers, df, "CSV_Download"
    except Exception as e:
        print(f"❌ CSV download failed: {e}")

    # Method 3: Try NSE API
    try:
        print("🌐 Trying NSE API...")
        api_url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100"
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
            'Referer': 'https://www.nseindia.com/'
        }
        response = requests.get(api_url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            stocks = data.get('data', [])
            tickers = [stock['symbol'] for stock in stocks]
            df = pd.DataFrame(stocks)
            print(f"✅ NSE API: Found {len(tickers)} stocks")
            return tickers, df, "NSE_API"
    except Exception as e:
        print(f"❌ NSE API failed: {e}")

    # Fallback
    print("⚠️ All methods failed, using backup list...")
    backup_tickers = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK",
        "KOTAKBANK", "BHARTIARTL", "ITC", "SBIN", "LT", "ASIANPAINT",
        "AXISBANK", "MARUTI", "TITAN", "SUNPHARMA", "ULTRACEMCO", "NESTLEIND"
    ]
    df = pd.DataFrame({'Symbol': backup_tickers})
    return backup_tickers, df, "Backup_List"

def save_tickers_multiple_formats(tickers, df, method):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename_csv = f'nifty100_tickers_{method}_{timestamp}.csv'
    filename_txt = f'nifty100_tickers_{method}_{timestamp}.txt'
    filename_json = f'nifty100_data_{method}_{timestamp}.json'

    df.to_csv(filename_csv, index=False)
    with open(filename_txt, 'w') as f:
        f.write('\n'.join(tickers))
    df.to_json(filename_json, orient='records', indent=2)

    print(f"💾 Data saved as:")
    print(f"   - CSV: {filename_csv}")
    print(f"   - TXT: {filename_txt}")
    print(f"   - JSON: {filename_json}")

# 🔁 Run every 5 minutes
if __name__ == "__main__":
    while True:
        tickers, data, method = get_nifty100_comprehensive()
        if tickers:
            print(f"\n📋 NIFTY 100 Stock Tickers ({len(tickers)} total):")
            print(f"Method used: {method}")
            print(f"Sample: {tickers[:10]}")
            save_tickers_multiple_formats(tickers, data, method)
        else:
            print("❌ Failed to fetch NIFTY 100 tickers")
        
        print("⏳ Waiting 5 minutes for next refresh...\n")
        time.sleep(300)  # 5 minutes