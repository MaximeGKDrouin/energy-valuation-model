import sqlite3
import pandas as pd
import numpy as np
import logging
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QuantScreener")

class EnergyScreener:
    def __init__(self, db_path: str = "energy_asset_management.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)

    def load_latest_fundamentals(self) -> pd.DataFrame:
        """Queries the database for the most recent quarterly fundamentals for each ticker."""
        logger.info("Loading fundamental data from SQLite...")
        query = """
            SELECT f.*, t.company_name, t.region, t.energy_type 
            FROM quarterly_fundamentals f
            JOIN tickers t ON f.ticker = t.ticker
            INNER JOIN (
                SELECT ticker, MAX(fiscal_date_ending) as latest_date
                FROM quarterly_fundamentals
                GROUP BY ticker
            ) latest ON f.ticker = latest.ticker AND f.fiscal_date_ending = latest.latest_date
        """
        df = pd.read_sql_query(query, self.conn)
        return df

    def calculate_quality_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Capital Efficiency (ROA proxy) and scores the universe."""
        logger.info("Calculating Quality Factor (EBIT / Total Assets)...")
        df['total_assets'] = df['total_assets'].replace(0, np.nan)
        df['quality_metric'] = df['ebit'] / df['total_assets']
        
        # Rank: Higher efficiency is better (1.0 is best)
        df['quality_rank'] = df['quality_metric'].rank(pct=True, ascending=True)
        return df

    def calculate_value_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetches Market Cap dynamically and calculates Free Cash Flow Yield."""
        logger.info("Fetching live Market Caps to calculate FCF Yield (Value Factor)...")
        
        market_caps = []
        tickers = df['ticker'].tolist()
        
        # Fetch current market caps dynamically
        for ticker in tickers:
            try:
                # Suppress the yfinance output to keep logs clean
                info = yf.Ticker(ticker).fast_info
                mcap = info.market_cap
                market_caps.append(mcap if mcap else np.nan)
            except Exception:
                market_caps.append(np.nan)
                
        df['market_cap'] = market_caps
        
        logger.info("Calculating Free Cash Flow Yield...")
        # FCF = Operating Cash Flow - Capital Expenditures
        df['free_cash_flow'] = df['operating_cash_flow'] - df['capital_expenditures']
        
        # FCF Yield = FCF / Market Cap
        df['market_cap'] = df['market_cap'].replace(0, np.nan)
        # Multiply by 4 to annualize the quarterly FCF
        df['fcf_yield'] = (df['free_cash_flow'] * 4) / df['market_cap'] 
        
        # Rank: Higher FCF yield is cheaper/better (1.0 is best)
        df['value_rank'] = df['fcf_yield'].rank(pct=True, ascending=True)
        return df

    def generate_conviction_list(self, df: pd.DataFrame, quality_weight: float = 0.5, value_weight: float = 0.5) -> pd.DataFrame:
        """Blends the factors into a final institutional conviction score."""
        logger.info(f"Generating Composite Score (Quality: {quality_weight*100}%, Value: {value_weight*100}%)...")
        
        # Drop rows missing crucial data
        df = df.dropna(subset=['quality_rank', 'value_rank']).copy()
        
        # Calculate the weighted composite score
        df['composite_score'] = (df['quality_rank'] * quality_weight) + (df['value_rank'] * value_weight)
        
        # Final Sort: Best overall companies at the top
        df = df.sort_values(by='composite_score', ascending=False).reset_index(drop=True)
        return df

if __name__ == "__main__":
    screener = EnergyScreener()
    
    # 1. Load Data
    fundamentals_df = screener.load_latest_fundamentals()
    
    # 2. Calculate Factors
    df_quality = screener.calculate_quality_factor(fundamentals_df)
    df_value = screener.calculate_value_factor(df_quality)
    
    # 3. Generate Final Conviction List (50% Value, 50% Quality)
    master_list = screener.generate_conviction_list(df_value, quality_weight=0.5, value_weight=0.5)
    
    # 4. Display the Top 15 "Buy" Candidates
    print("\n" + "="*80)
    print(" 🏆 GSAM QUANTITATIVE SCREENER: TOP 15 CONVICTION BUYS 🏆")
    print("="*80)
    
    # Format the display cleanly
    display_cols = ['ticker', 'company_name', 'energy_type', 'fcf_yield', 'quality_metric', 'composite_score']
    
    # Convert decimals to percentages for readability
    master_list_display = master_list[display_cols].head(15).copy()
    master_list_display['fcf_yield'] = (master_list_display['fcf_yield'] * 100).round(2).astype(str) + '%'
    master_list_display['quality_metric'] = (master_list_display['quality_metric'] * 100).round(2).astype(str) + '%'
    master_list_display['composite_score'] = (master_list_display['composite_score'] * 100).round(1)
    
    # Rename columns for the final printout
    master_list_display.columns = ['Ticker', 'Company', 'Sector', 'Est. FCF Yield (Value)', 'ROA (Quality)', 'Conviction Score (0-100)']
    
    print(master_list_display.to_string(index=False))
    print("\n*Note: FCF Yield is annualized based on the most recent quarter.")