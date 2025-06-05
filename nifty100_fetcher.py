import requests
import pandas as pd
import time
from io import StringIO
from datetime import datetime, timedelta
import json
import warnings
from bs4 import BeautifulSoup
import yfinance as yf
import os
warnings.filterwarnings('ignore')

class EnhancedNIFTY100Fetcher:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session.headers.update(self.headers)
        
        # Hardcoded NIFTY 100 list as ultimate fallback
        self.nifty100_symbols = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK", 
            "KOTAKBANK", "BHARTIARTL", "ITC", "SBIN", "LT", "ASIANPAINT",
            "AXISBANK", "MARUTI", "TITAN", "SUNPHARMA", "ULTRACEMCO", "NESTLEIND",
            "HCLTECH", "BAJFINANCE", "WIPRO", "POWERGRID", "NTPC", "TECHM",
            "ONGC", "TATAMOTORS", "COALINDIA", "BAJAJFINSV", "HDFCLIFE",
            "SBILIFE", "BRITANNIA", "SHREECEM", "GRASIM", "CIPLA", "DRREDDY",
            "EICHERMOT", "BPCL", "ADANIENT", "TATACONSUM", "APOLLOHOSP",
            "DIVISLAB", "HINDALCO", "JSWSTEEL", "HEROMOTOCO", "INDUSINDBK",
            "BAJAJ-AUTO", "GODREJCP", "PIDILITIND", "BERGEPAINT", "DABUR",
            "MARICO", "COLPAL", "MCDOWELL-N", "HAVELLS", "TORNTPHARM",
            "BIOCON", "LUPIN", "GLAND", "DMART", "NAUKRI", "ZEEL", "SAIL",
            "VEDL", "HINDZINC", "JINDALSTEL", "TATASTEEL", "NMDC", "MOIL",
            "CONCOR", "IRCTC", "RAILTEL", "RVNL", "IRFC", "RECLTD",
            "PFC", "NHPC", "SJVN", "NTPC", "POWERGRID", "TATAPOWER",
            "ADANIPOWER", "JSWENERGY", "TORNTPOWER", "CESC", "PEL",
            "CROMPTON", "VOLTAS", "BLUESTARCO", "WHIRLPOOL", "AMBER",
            "DIXON", "ROUTE", "LALPATHLAB", "METROPOLIS", "THYROCARE",
            "FORTIS", "MAXHEALTH", "NARAYANA", "ASTER", "RAINBOW"
        ]
    
    def get_nifty100_symbols_method1(self):
        """Method 1: Try NSE official CSV download"""
        try:
            print("📄 Trying NSE official CSV download...")
            csv_urls = [
                "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
                "https://www1.nseindia.com/content/indices/ind_nifty100list.csv",
                "https://www.nseindia.com/content/indices/ind_nifty100list.csv"
            ]
            
            for url in csv_urls:
                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        df = pd.read_csv(StringIO(response.text))
                        if 'Symbol' in df.columns and len(df) > 50:
                            symbols = df['Symbol'].tolist()
                            print(f"✅ CSV Method: Found {len(symbols)} symbols")
                            return symbols, df
                except:
                    continue
                    
        except Exception as e:
            print(f"❌ CSV download failed: {e}")
        return [], pd.DataFrame()
    
    def get_nifty100_symbols_method2(self):
        """Method 2: Try NSE API with session management"""
        try:
            print("🌐 Trying NSE API with session...")
            
            # First, get NSE session
            self.session.get("https://www.nseindia.com", timeout=10)
            time.sleep(1)
            
            # Try multiple API endpoints
            api_endpoints = [
                "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100",
                "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY100",
                "https://www.nseindia.com/api/allIndices"
            ]
            
            for endpoint in api_endpoints:
                try:
                    response = self.session.get(endpoint, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'data' in data:
                            stocks = data['data']
                            if len(stocks) > 50:  # Reasonable check for NIFTY 100
                                symbols = [stock['symbol'] for stock in stocks]
                                df = pd.DataFrame(stocks)
                                print(f"✅ NSE API: Found {len(symbols)} symbols")
                                return symbols, df
                        
                        elif isinstance(data, list) and len(data) > 0:
                            # Handle different response format
                            for item in data:
                                if 'indexName' in item and 'NIFTY 100' in item['indexName']:
                                    # This might contain the index data
                                    continue
                                    
                except Exception as e:
                    print(f"API endpoint {endpoint} failed: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ NSE API method failed: {e}")
        return [], pd.DataFrame()
    
    def get_nifty100_symbols_method3(self):
        """Method 3: Web scraping from NSE website"""
        try:
            print("🕷️ Trying web scraping...")
            
            urls_to_try = [
                "https://www.nseindia.com/market-data/live-equity-market?symbol=NIFTY%20100",
                "https://www.nseindia.com/products-services/indices-nifty100-index",
                "https://www1.nseindia.com/live_market/dynaContent/live_watch/get_quote/GetQuote.jsp?symbol=NIFTY%20100"
            ]
            
            for url in urls_to_try:
                try:
                    response = self.session.get(url, timeout=15)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Look for tables with stock data
                        tables = soup.find_all('table')
                        for table in tables:
                            rows = table.find_all('tr')
                            if len(rows) > 50:  # Likely contains stock list
                                symbols = []
                                for row in rows[1:]:  # Skip header
                                    cells = row.find_all(['td', 'th'])
                                    if cells:
                                        symbol = cells[0].get_text(strip=True)
                                        if symbol and len(symbol) < 20:  # Basic validation
                                            symbols.append(symbol)
                                
                                if len(symbols) > 50:
                                    print(f"✅ Web scraping: Found {len(symbols)} symbols")
                                    df = pd.DataFrame({'Symbol': symbols})
                                    return symbols, df
                                    
                except Exception as e:
                    print(f"Scraping URL {url} failed: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Web scraping failed: {e}")
        return [], pd.DataFrame()
    
    def get_nifty100_symbols_comprehensive(self):
        """Try all methods to get NIFTY 100 symbols"""
        print("🔍 Attempting to fetch NIFTY 100 symbols using multiple methods...")
        
        # Method 1: CSV Download
        symbols, df = self.get_nifty100_symbols_method1()
        if symbols:
            return symbols, df, "CSV_Download"
        
        # Method 2: NSE API
        symbols, df = self.get_nifty100_symbols_method2()
        if symbols:
            return symbols, df, "NSE_API"
        
        # Method 3: Web Scraping
        symbols, df = self.get_nifty100_symbols_method3()
        if symbols:
            return symbols, df, "Web_Scraping"
        
        # Fallback: Use hardcoded list
        print("⚠️ All methods failed, using comprehensive hardcoded NIFTY 100 list...")
        df = pd.DataFrame({'Symbol': self.nifty100_symbols})
        return self.nifty100_symbols, df, "Hardcoded_List"
    
    def get_historical_data_yfinance(self, symbol, years_back=10):
        """Get historical data using Yahoo Finance (more reliable)"""
        try:
            # Convert NSE symbol to Yahoo Finance format
            yf_symbol = f"{symbol}.NS"
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years_back * 365)
            
            # Download data from Yahoo Finance
            ticker = yf.Ticker(yf_symbol)
            hist_data = ticker.history(start=start_date, end=end_date)
            
            if not hist_data.empty:
                # Rename columns to match our format
                hist_data = hist_data.reset_index()
                hist_data['Symbol'] = symbol
                hist_data = hist_data.rename(columns={
                    'Date': 'Date',
                    'Open': 'DAYOPEN',
                    'High': 'DAYHIGH',
                    'Low': 'DAYLOW',
                    'Close': 'LTP',
                    'Volume': 'VOLUME(QUANTITY MN)'
                })
                
                # Calculate additional fields
                hist_data['PREVCLOSE'] = hist_data['LTP'].shift(1)
                hist_data['CHNG'] = hist_data['LTP'] - hist_data['PREVCLOSE']
                hist_data['%CHNG'] = (hist_data['CHNG'] / hist_data['PREVCLOSE']) * 100
                
                # Calculate turnover (approximate)
                hist_data['TURNOVER(INR MN)'] = (hist_data['VOLUME(QUANTITY MN)'] * hist_data['LTP']) / 1000000
                
                # Get 52-week high/low
                hist_data['52W HIGH'] = hist_data['LTP'].rolling(window=252, min_periods=1).max()
                hist_data['52W LOW'] = hist_data['LTP'].rolling(window=252, min_periods=1).min()
                
                return hist_data
                
        except Exception as e:
            print(f"❌ Yahoo Finance failed for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def get_historical_data_bulk(self, symbols, years_back=10):
        """Get historical data for all symbols"""
        print(f"📈 Fetching {years_back} years of historical data for {len(symbols)} stocks...")
        
        all_data = []
        failed_symbols = []
        
        for i, symbol in enumerate(symbols):
            print(f"Processing {symbol} ({i+1}/{len(symbols)})...")
            
            hist_data = self.get_historical_data_yfinance(symbol, years_back)
            
            if not hist_data.empty:
                all_data.append(hist_data)
                print(f"✅ {symbol}: {len(hist_data)} records")
            else:
                failed_symbols.append(symbol)
                print(f"❌ {symbol}: No data")
            
            # Rate limiting
            time.sleep(0.1)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n✅ Successfully fetched data for {len(symbols) - len(failed_symbols)} symbols")
            print(f"📊 Total records: {len(combined_df)}")
            
            if failed_symbols:
                print(f"⚠️ Failed symbols: {failed_symbols}")
            
            return combined_df
        
        return pd.DataFrame()
    
    def save_data(self, df, filename_prefix):
        """Save data with enhanced formatting"""
        if df.empty:
            print("❌ No data to save")
            return None, None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean and format data
        numeric_columns = ['LTP', 'CHNG', '%CHNG', 'DAYOPEN', 'DAYHIGH', 'DAYLOW', 
                          'PREVCLOSE', 'VOLUME(QUANTITY MN)', 'TURNOVER(INR MN)', 
                          '52W HIGH', '52W LOW']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Save as CSV
        csv_filename = f"{filename_prefix}_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        
        # Save as Excel with formatting
        excel_filename = f"{filename_prefix}_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All_Data', index=False)
            
            # Create summary sheet
            if 'Symbol' in df.columns:
                summary = df.groupby('Symbol').agg({
                    'LTP': 'last',
                    'VOLUME(QUANTITY MN)': 'sum',
                    'TURNOVER(INR MN)': 'sum'
                }).reset_index()
                summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Create yearly sheets if date column exists
            date_columns = ['Date', 'Timestamp']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df['Year'] = df[date_col].dt.year
                
                for year in sorted(df['Year'].unique()):
                    year_data = df[df['Year'] == year].copy()
                    year_data = year_data.drop('Year', axis=1)
                    year_data.to_excel(writer, sheet_name=f'Year_{year}', index=False)
        
        print(f"💾 Data saved successfully:")
        print(f"   📄 CSV: {csv_filename}")
        print(f"   📊 Excel: {excel_filename}")
        print(f"   📈 Records: {len(df):,}")
        
        return csv_filename, excel_filename

def main():
    """Enhanced main function"""
    fetcher = EnhancedNIFTY100Fetcher()
    
    print("🚀 Enhanced NIFTY 100 Data Fetcher")
    print("=" * 50)
    
    print("\nSelect operation:")
    print("1. Get NIFTY 100 symbols only")
    print("2. Get historical data (Yahoo Finance - Reliable)")
    print("3. Get both symbols and historical data")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Get symbols only
        symbols, df, method = fetcher.get_nifty100_symbols_comprehensive()
        print(f"\n📋 Found {len(symbols)} NIFTY 100 symbols using {method}")
        print(f"Sample symbols: {symbols[:10]}")
        
        fetcher.save_data(df, "nifty100_symbols")
    
    elif choice == "2":
        # Get historical data
        years = input("Enter number of years for historical data (default 10): ").strip()
        years = int(years) if years.isdigit() else 10
        
        # First get symbols
        symbols, _, method = fetcher.get_nifty100_symbols_comprehensive()
        print(f"📋 Using {len(symbols)} symbols from {method}")
        
        # Get historical data
        historical_data = fetcher.get_historical_data_bulk(symbols, years_back=years)
        
        if not historical_data.empty:
            fetcher.save_data(historical_data, f"nifty100_historical_{years}years")
            print(f"\n📊 Sample historical data:")
            print(historical_data[['Symbol', 'Date', 'LTP', 'CHNG', '%CHNG']].head(10))
        else:
            print("❌ No historical data fetched")
    
    elif choice == "3":
        # Get both
        years = input("Enter number of years for historical data (default 10): ").strip()
        years = int(years) if years.isdigit() else 10
        
        # Get symbols
        symbols, symbols_df, method = fetcher.get_nifty100_symbols_comprehensive()
        fetcher.save_data(symbols_df, "nifty100_symbols")
        
        # Get historical data
        historical_data = fetcher.get_historical_data_bulk(symbols, years_back=years)
        if not historical_data.empty:
            fetcher.save_data(historical_data, f"nifty100_historical_{years}years")
        
        print("✅ Both symbols and historical data processing completed!")
    
    else:
        print("❌ Invalid choice")
if __name__ == "__main__":
    fetcher = EnhancedNIFTY100Fetcher()
    
    while True:
        try:
            # Get the latest NIFTY 100 symbols
            symbols, df_symbols, source = fetcher.get_nifty100_symbols_comprehensive()
            
            # Save symbols list
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol_filename = f"nifty100_symbols_{source}_{timestamp}.csv"
            df_symbols.to_csv(symbol_filename, index=False)
            print(f"✅ Saved symbol list to {symbol_filename}")
            
            # Fetch historical data
            data_df = fetcher.get_historical_data_bulk(symbols, years_back=1)  # 1 year for faster refresh
            
            if not data_df.empty:
                data_filename = f"nifty100_historical_data_{timestamp}.csv"
                data_df.to_csv(data_filename, index=False)
                print(f"✅ Saved historical data to {data_filename}")
            else:
                print("⚠️ No data fetched.")
            
        except Exception as e:
            print(f"❌ Error during execution: {e}")
        
        # Wait for 10 minutes before next update
        print("⏳ Waiting 5 minutes for next update...\n")
        time.sleep(300)


if __name__ == "__main__":
    main()



