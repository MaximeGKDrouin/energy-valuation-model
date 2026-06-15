import sqlite3
import pandas as pd
import numpy as np
import logging
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AdvancedBacktestEngine")

class VectorizedBacktester:
    def __init__(self, db_path: str = "energy_asset_management.db"):
        self.conn = sqlite3.connect(db_path)
        self.prices = pd.DataFrame()
        self.price_matrix = pd.DataFrame() # NEW: Optimized matrix for quick momentum lookups
        self.fundamentals = pd.DataFrame()
        self.sp500_prices = pd.Series(dtype=float)

    def load_data(self):
        """Loads all historical data and applies Point-in-Time lag."""
        logger.info("Loading 5 years of historical pricing...")
        self.prices = pd.read_sql_query("SELECT * FROM daily_prices", self.conn)
        self.prices['date'] = pd.to_datetime(self.prices['date'])
        self.prices.set_index(['date', 'ticker'], inplace=True)

        logger.info("Fetching S&P 500 (SPY) benchmark data from Yahoo Finance...")
        spy_data = yf.download('SPY', start='2021-01-01', end='2026-12-31', progress=False)
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = spy_data.columns.get_level_values(0)
        spy_data.index = spy_data.index.tz_localize(None)
        
        if 'Adj Close' in spy_data.columns:
            self.sp500_prices = spy_data['Adj Close']
        else:
            self.sp500_prices = spy_data['Close']

        logger.info("Loading fundamentals and applying 45-day Look-Ahead Bias lag...")
        query = """
            SELECT f.*, t.region, t.energy_type 
            FROM quarterly_fundamentals f
            JOIN tickers t ON f.ticker = t.ticker
        """
        self.fundamentals = pd.read_sql_query(query, self.conn)
        self.fundamentals['fiscal_date_ending'] = pd.to_datetime(self.fundamentals['fiscal_date_ending'])
        
        self.fundamentals['effective_date'] = self.fundamentals['fiscal_date_ending'] + pd.Timedelta(days=45)
        self.fundamentals = self.fundamentals.sort_values('effective_date')

        # --- NEW LOGIC: Fetch static shares outstanding for Market Cap calculation ---
        logger.info("Fetching current shares outstanding to calculate historical Market Cap...")
        unique_tickers = self.prices.index.get_level_values('ticker').unique()
        shares_dict = {}
        for ticker in unique_tickers:
            try:
                shares = yf.Ticker(ticker).fast_info.shares
                shares_dict[ticker] = shares if shares else np.nan
            except Exception:
                shares_dict[ticker] = np.nan
                
        self.shares_series = pd.Series(shares_dict)

    def run_backtest(self, start_year: str = '2022', end_year: str = '2026'):
        """Simulates the monthly rebalancing of the portfolios."""
        logger.info(f"Initiating historical simulation from {start_year} to {end_year}...")
        
        # 1. Get dates and explicitly force them into a DatetimeIndex
        raw_dates = self.prices.index.get_level_values('date').unique()
        dates = pd.to_datetime(raw_dates)
        
        # 2. Safely filter using the now-guaranteed .year attribute
        dates = dates[(dates.year >= int(start_year)) & (dates.year <= int(end_year))]
        
        # FIX 1: Convert month_ends explicitly back to pandas Timestamps
        month_ends = pd.to_datetime(dates.to_series().groupby([dates.year, dates.month]).max().values)
        
        # FIX 2: Sort the MultiIndex safely to prevent pandas lookup KeyErrors
        self.prices = self.prices.sort_index()
        
        portfolio_returns = []

        for i in range(len(month_ends) - 1):
            current_date = month_ends[i]
            next_date = month_ends[i+1]
            
            # 1. Filter fundamentals available EXACTLY on this date
            available_funds = self.fundamentals[self.fundamentals['effective_date'] <= current_date].copy()
            latest_funds = available_funds.drop_duplicates(subset=['ticker'], keep='last').copy()
            
            # --- Calculate Point-in-Time Market Cap ---
            try:
                current_prices = self.prices.loc[current_date]['close']
            except KeyError:
                continue # Skip if pricing dates don't align
                
            latest_funds['price'] = latest_funds['ticker'].map(current_prices)
            latest_funds['shares'] = latest_funds['ticker'].map(self.shares_series)
            latest_funds['market_cap'] = latest_funds['price'] * latest_funds['shares']
            latest_funds['market_cap'] = latest_funds['market_cap'].replace(0, np.nan)
            
            # --- Calculate Factors (Quality = ROA, Value = FCF Yield) ---
            latest_funds['quality_metric'] = latest_funds['ebit'] / latest_funds['total_assets'].replace(0, np.nan)
            
            latest_funds['free_cash_flow'] = latest_funds['operating_cash_flow'] - latest_funds['capital_expenditures']
            latest_funds['value_metric'] = (latest_funds['free_cash_flow'] * 4) / latest_funds['market_cap']
            
            latest_funds['q_rank'] = latest_funds['quality_metric'].rank(pct=True, ascending=True)
            latest_funds['v_rank'] = latest_funds['value_metric'].rank(pct=True, ascending=True)
            latest_funds['score'] = (latest_funds['q_rank'] * 0.5) + (latest_funds['v_rank'] * 0.5)
            
            # 3. Split into Fossil vs Clean Energy
            fossil_universe = latest_funds[latest_funds['energy_type'] == 'Fossil']
            clean_universe = latest_funds[latest_funds['energy_type'].isin(['Renewable', 'Nuclear'])]
            
            top_fossil = fossil_universe.nlargest(10, 'score')['ticker'].tolist()
            top_clean = clean_universe.nlargest(10, 'score')['ticker'].tolist()
            
            # 4. Calculate Forward 1-Month Return for these stocks
            try:
                current_prices_adj = self.prices.loc[current_date]['adj_close']
                next_prices_adj = self.prices.loc[next_date]['adj_close']
                returns = (next_prices_adj - current_prices_adj) / current_prices_adj
                
                fossil_ret = returns.reindex(top_fossil).mean()
                clean_ret = returns.reindex(top_clean).mean()
                
                current_spy = self.sp500_prices.loc[:current_date].iloc[-1]
                next_spy = self.sp500_prices.loc[:next_date].iloc[-1]
                benchmark_ret = (next_spy - current_spy) / current_spy
                
                portfolio_returns.append({
                    'date': next_date,
                    'Fossil_Top10': fossil_ret if pd.notna(fossil_ret) else 0,
                    'Clean_Top10': clean_ret if pd.notna(clean_ret) else 0,
                    'S&P_500_Benchmark': benchmark_ret if pd.notna(benchmark_ret) else 0
                })
            except Exception as e:
                logger.warning(f"Skipping {current_date.strftime('%Y-%m-%d')} due to error: {e}")
                continue 

        # FIX 4: Graceful exit if data is completely empty
        if not portfolio_returns:
            raise ValueError("Backtest failed: No returns calculated. Check logs for missing price data.")
            
        return pd.DataFrame(portfolio_returns).set_index('date')

    def generate_tearsheet(self, returns_df: pd.DataFrame):
        """Calculates institutional risk metrics."""
        logger.info("Calculating Risk Analytics...")
        returns_df = returns_df.fillna(0)
        cum_returns = (1 + returns_df).cumprod()
        
        metrics = {}
        for col in returns_df.columns:
            ann_ret = (cum_returns[col].iloc[-1]) ** (12 / len(returns_df)) - 1
            volatility = returns_df[col].std() * np.sqrt(12)
            sharpe = (ann_ret - 0.02) / volatility if volatility > 0 else 0 
            
            rolling_max = cum_returns[col].cummax()
            drawdown = (cum_returns[col] / rolling_max) - 1
            max_dd = drawdown.min()
            
            metrics[col] = {
                'Total Return': f"{(cum_returns[col].iloc[-1] - 1)*100:.1f}%",
                'Annualized Return': f"{ann_ret*100:.1f}%",
                'Annualized Volatility': f"{volatility*100:.1f}%",
                'Sharpe Ratio': f"{sharpe:.2f}",
                'Max Drawdown': f"{max_dd*100:.1f}%"
            }
            
        print("\n" + "="*75)
        print(" 📊 3-FACTOR BACKTEST (VALUE + QUALITY + MOMENTUM) 📊")
        print("="*75)
        print(pd.DataFrame(metrics).T.to_string())

if __name__ == "__main__":
    backtester = VectorizedBacktester()
    backtester.load_data()
    historical_returns = backtester.run_backtest(start_year='2022', end_year='2026')
    backtester.generate_tearsheet(historical_returns)