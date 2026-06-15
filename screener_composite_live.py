import sqlite3
import pandas as pd
import numpy as np
import logging
import yfinance as yf

# --- THE GAG ORDER ---
# Force yfinance to only report CRITICAL system crashes, silencing all missing data warnings
yf_logger = logging.getLogger('yfinance')
yf_logger.setLevel(logging.CRITICAL)

# Configure your own custom logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QuantScreener")



class EnergyScreener:
    def __init__(self, db_path: str = "energy_asset_management.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)

    def load_latest_fundamentals(self) -> pd.DataFrame:
        """Queries the database for the most recent quarterly fundamentals."""
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
        
        # NEW LOGIC: Drop companies missing required data before ranking
        df = df.dropna(subset=['ebit', 'total_assets']).copy()
        
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
                info = yf.Ticker(ticker).fast_info
                mcap = info.market_cap
                market_caps.append(mcap if mcap else np.nan)
            except Exception:
                market_caps.append(np.nan)
                
        df['market_cap'] = market_caps
        
        # NEW LOGIC: Drop companies missing cash flow or market cap data before ranking
        df = df.dropna(subset=['operating_cash_flow', 'capital_expenditures', 'market_cap']).copy()
        
        logger.info("Calculating Free Cash Flow Yield...")
        # FCF = Operating Cash Flow - Capital Expenditures
        df['free_cash_flow'] = df['operating_cash_flow'] - df['capital_expenditures']
        
        # FCF Yield = FCF / Market Cap
        df['market_cap'] = df['market_cap'].replace(0, np.nan)
        df['fcf_yield'] = (df['free_cash_flow'] * 4) / df['market_cap'] 
        
        # Rank: Higher FCF yield is cheaper/better (1.0 is best)
        df['value_rank'] = df['fcf_yield'].rank(pct=True, ascending=True)
        return df

    def fetch_live_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetches Market Cap dynamically and applies FX normalization."""
        logger.info("Fetching live Market Caps from Yahoo Finance...")
        market_caps = []
        tickers = df['ticker'].tolist()
        
        for ticker in tickers:
            try:
                # Attempt 1: The ultra-fast scraper
                info = yf.Ticker(ticker).fast_info
                mcap = info.market_cap
                market_caps.append(mcap if mcap else np.nan)
            except Exception:
                try:
                    # Attempt 2: The deeper, slower fallback scraper
                    deeper_info = yf.Ticker(ticker).info
                    mcap = deeper_info.get('marketCap', np.nan)
                    market_caps.append(mcap)
                except Exception:
                    # Total failure: gracefully assign NaN and move on
                    market_caps.append(np.nan)
                
        df['market_cap'] = market_caps
        df['market_cap'] = df['market_cap'].replace(0, np.nan)

        # --- THE FX NORMALIZATION FIX ---
        logger.info("Normalizing Emerging Market Currencies to USD equivalents...")
        
        # 1. Argentina (ARS): TGS and YPF report FCF in Pesos, but trade in USD. 
        # We multiply the USD Market Cap by the exchange rate (~900 ARS to 1 USD) so the currencies match.
        df.loc[df['ticker'].isin(['TGS', 'YPF']), 'market_cap'] *= 900.0 
        
        # 2. Brazil (BRL): Petrobras reports FCF in Reals, but ADR trades in USD.
        # Exchange rate is roughly 5.4 BRL to 1 USD.
        df.loc[df['ticker'] == 'PBR', 'market_cap'] *= 5.4 
        
        # 3. Eurozone (EUR): Neste reports in Euros, but we want a unified baseline.
        df.loc[df['ticker'] == 'NESTE.HE', 'market_cap'] *= 0.93

        return df


    def calculate_relative_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the Free Cash Flow Yield (Relative Value)."""
        logger.info("Calculating Free Cash Flow Yield...")
        df['free_cash_flow'] = df['operating_cash_flow'] - df['capital_expenditures']
        df['fcf_yield'] = (df['free_cash_flow'] * 4) / df['market_cap'] 
        df['value_rank'] = df['fcf_yield'].rank(pct=True, ascending=True)
        return df

    def calculate_absolute_value_dcf(self, df: pd.DataFrame, wacc: float = 0.10, growth_rate: float = 0.05, terminal_growth: float = 0.02) -> pd.DataFrame:
        """Runs a 5-Year DCF Model to find the Intrinsic Margin of Safety."""
        logger.info("Running 5-Year Discounted Cash Flow (DCF) Engine...")
        
        # 1. Baseline Cash Flow (Annualized from latest quarter)
        df['annual_fcf'] = df['free_cash_flow'] * 4
        
        # We drop companies burning cash (negative FCF) as they cannot be valued via traditional DCF
        df.loc[df['annual_fcf'] <= 0, 'intrinsic_value'] = np.nan
        
        # 2. Project 5 Years of Cash Flows and Discount to Present Value (PV)
        discount_factors = [(1 + wacc) ** i for i in range(1, 6)]
        
        pv_fcf = 0
        for i, dfactor in enumerate(discount_factors, start=1):
            projected_fcf = df['annual_fcf'] * ((1 + growth_rate) ** i)
            pv_fcf += projected_fcf / dfactor
            
        # 3. Calculate Terminal Value (Value of the company forever after Year 5)
        year_5_fcf = df['annual_fcf'] * ((1 + growth_rate) ** 5)
        terminal_value = (year_5_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
        pv_terminal_value = terminal_value / ((1 + wacc) ** 5)
        
        # 4. Total Intrinsic Value & Margin of Safety
        df['intrinsic_value'] = pv_fcf + pv_terminal_value
        df['margin_of_safety'] = (df['intrinsic_value'] - df['market_cap']) / df['market_cap']
        
        # Rank: Higher Margin of Safety is better
        df['dcf_rank'] = df['margin_of_safety'].rank(pct=True, ascending=True)
        
        return df

    def generate_conviction_list(self, df: pd.DataFrame) -> pd.DataFrame:
        """Blends Relative Value, Absolute Value (DCF), and Quality."""
        logger.info("Generating Final 3-Factor Fundamental Composite Score...")
        
        df = df.dropna(subset=['quality_rank', 'value_rank', 'dcf_rank']).copy()
        
        # Master Formula: 33% Quality, 33% FCF Yield, 34% DCF Margin of Safety
        df['composite_score'] = (df['quality_rank'] * 0.33) + (df['value_rank'] * 0.33) + (df['dcf_rank'] * 0.34)
        df = df.sort_values(by='composite_score', ascending=False).reset_index(drop=True)
        return df

if __name__ == "__main__":
    screener = EnergyScreener()
    
    # Run the pipeline
    df_funds = screener.load_latest_fundamentals()
    df_qual = screener.calculate_quality_factor(df_funds)
    df_live = screener.fetch_live_market_data(df_qual)
    df_val = screener.calculate_relative_value(df_live)
    df_dcf = screener.calculate_absolute_value_dcf(df_val, wacc=0.10, growth_rate=0.05, terminal_growth=0.02)
    
    master_list = screener.generate_conviction_list(df_dcf)
    
    # Split the Portfolios
    fossil_df = master_list[master_list['energy_type'] == 'Fossil'].copy()
    clean_df = master_list[master_list['energy_type'].isin(['Renewable', 'Nuclear'])].copy()
    
    # Format and Display
    display_cols = ['ticker', 'company_name', 'energy_type', 'fcf_yield', 'margin_of_safety', 'composite_score']
    
    def format_output(df: pd.DataFrame) -> pd.DataFrame:
        df_disp = df[display_cols].head(10).copy()
        df_disp['fcf_yield'] = (df_disp['fcf_yield'] * 100).round(1).astype(str) + '%'
        df_disp['margin_of_safety'] = (df_disp['margin_of_safety'] * 100).round(1).astype(str) + '%'
        df_disp['composite_score'] = (df_disp['composite_score'] * 100).round(1)
        df_disp.columns = ['Ticker', 'Company', 'Type', 'FCF Yield', 'DCF Margin of Safety', 'Conviction Score']
        return df_disp

    print("\n" + "="*95)
    print(" 🛢️  TRADITIONAL FOSSIL PORTFOLIO: TOP 10 CONVICTION BUYS (DCF INTEGRATED)")
    print("="*95)
    print(format_output(fossil_df).to_string(index=False))

    print("\n" + "="*95)
    print(" ⚡  CLEAN ENERGY TRANSITION PORTFOLIO: TOP 10 CONVICTION BUYS (DCF INTEGRATED)")
    print("="*95)
    print(format_output(clean_df).to_string(index=False))