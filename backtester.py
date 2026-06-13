import sqlite3
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BacktestEngine")

class VectorizedBacktester:
    def __init__(self, db_path: str = "energy_asset_management.db"):
        self.conn = sqlite3.connect(db_path)
        self.prices = pd.DataFrame()
        self.fundamentals = pd.DataFrame()

    def load_data(self):
        """Loads all historical data and applies Point-in-Time lag."""
        logger.info("Loading 5 years of historical pricing...")
        self.prices = pd.read_sql_query("SELECT * FROM daily_prices", self.conn)
        self.prices['date'] = pd.to_datetime(self.prices['date'])
        self.prices.set_index(['date', 'ticker'], inplace=True)

        logger.info("Loading fundamentals and applying 45-day Look-Ahead Bias lag...")
        query = """
            SELECT f.*, t.region, t.energy_type 
            FROM quarterly_fundamentals f
            JOIN tickers t ON f.ticker = t.ticker
        """
        self.fundamentals = pd.read_sql_query(query, self.conn)
        self.fundamentals['fiscal_date_ending'] = pd.to_datetime(self.fundamentals['fiscal_date_ending'])
        
        # STRICT POINT-IN-TIME LOGIC: The market doesn't see earnings until ~45 days later
        self.fundamentals['effective_date'] = self.fundamentals['fiscal_date_ending'] + pd.Timedelta(days=45)
        self.fundamentals = self.fundamentals.sort_values('effective_date')

    def run_backtest(self, start_year: str = '2022', end_year: str = '2026'):
        """Simulates the monthly rebalancing of the portfolios."""
        logger.info(f"Initiating historical simulation from {start_year} to {end_year}...")
        
        # Get all end-of-month dates in our price history
        dates = self.prices.index.get_level_values('date').unique()
        dates = dates[(dates >= start_year) & (dates <= end_year)]
        month_ends = dates.to_series().groupby([dates.year, dates.month]).max().values
        
        portfolio_returns = []

        for i in range(len(month_ends) - 1):
            current_date = month_ends[i]
            next_date = month_ends[i+1]
            
            # 1. Filter fundamentals available EXACTLY on this date
            available_funds = self.fundamentals[self.fundamentals['effective_date'] <= current_date].copy()
            # Keep only the most recent report for each ticker
            latest_funds = available_funds.drop_duplicates(subset=['ticker'], keep='last').copy()
            
            # 2. Calculate Factors (Quality = ROA, Value = FCF Margin)
            latest_funds['quality_metric'] = latest_funds['ebit'] / latest_funds['total_assets'].replace(0, np.nan)
            latest_funds['fcf'] = latest_funds['operating_cash_flow'] - latest_funds['capital_expenditures']
            latest_funds['value_metric'] = latest_funds['fcf'] / latest_funds['total_revenue'].replace(0, np.nan)
            
            # Rank and combine (50/50 Equal Weight)
            latest_funds['q_rank'] = latest_funds['quality_metric'].rank(pct=True)
            latest_funds['v_rank'] = latest_funds['value_metric'].rank(pct=True)
            latest_funds['score'] = (latest_funds['q_rank'] * 0.5) + (latest_funds['v_rank'] * 0.5)
            
            # 3. Split into Fossil vs Clean Energy
            fossil_universe = latest_funds[latest_funds['energy_type'] == 'Fossil']
            clean_universe = latest_funds[latest_funds['energy_type'].isin(['Renewable', 'Nuclear'])]
            
            # Select Top 10 for each
            top_fossil = fossil_universe.nlargest(10, 'score')['ticker'].tolist()
            top_clean = clean_universe.nlargest(10, 'score')['ticker'].tolist()
            
            # 4. Calculate Forward 1-Month Return for these stocks
            try:
                current_prices = self.prices.loc[current_date]['adj_close']
                next_prices = self.prices.loc[next_date]['adj_close']
                returns = (next_prices - current_prices) / current_prices
                
                # Equal weight portfolio returns
                fossil_ret = returns.reindex(top_fossil).mean()
                clean_ret = returns.reindex(top_clean).mean()
                benchmark_ret = returns.mean() # Simple universe average as benchmark
                
                portfolio_returns.append({
                    'date': next_date,
                    'Fossil_Top10': fossil_ret if pd.notna(fossil_ret) else 0,
                    'Clean_Top10': clean_ret if pd.notna(clean_ret) else 0,
                    'Benchmark': benchmark_ret if pd.notna(benchmark_ret) else 0
                })
            except KeyError:
                continue # Skip if dates don't align perfectly in database

        return pd.DataFrame(portfolio_returns).set_index('date')

    def generate_tearsheet(self, returns_df: pd.DataFrame):
        """Calculates institutional risk metrics."""
        logger.info("Calculating Risk Analytics...")
        
        # Fill missing returns with 0 and calculate cumulative growth of $1
        returns_df = returns_df.fillna(0)
        cum_returns = (1 + returns_df).cumprod()
        
        metrics = {}
        for col in returns_df.columns:
            ann_ret = (cum_returns[col].iloc[-1]) ** (12 / len(returns_df)) - 1
            volatility = returns_df[col].std() * np.sqrt(12)
            sharpe = (ann_ret - 0.02) / volatility if volatility > 0 else 0 # Assume 2% Risk-Free Rate
            
            # Max Drawdown calculation
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
            
        print("\n" + "="*70)
        print(" 📊 GSAM QUANTITATIVE BACKTEST: 5-YEAR TEARSHEET 📊")
        print("="*70)
        print(pd.DataFrame(metrics).T.to_string())

if __name__ == "__main__":
    backtester = VectorizedBacktester()
    backtester.load_data()
    
    # Run from 2022 to the current year
    historical_returns = backtester.run_backtest(start_year='2022', end_year='2026')
    backtester.generate_tearsheet(historical_returns)