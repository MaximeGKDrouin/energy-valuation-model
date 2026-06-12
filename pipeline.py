import logging
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import yfinance as yf

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("EnergyPipeline")

class EnergyDataPipeline:
    """
    A production-grade pipeline to ingest, clean, and store historical pricing
    and quarterly fundamental data for global energy equities into SQLite.
    """
    def __init__(self, db_path: str = "energy_asset_management.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a standard sqlite3 connection object."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")  # Enforce relational integrity
        return conn

    def _init_db(self) -> None:
        """Initializes the database schema with global categorizations."""
        logger.info("Initializing SQLite database schema...")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Master Ticker Table (UPGRADED for Global Strategy)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tickers (
                    ticker TEXT PRIMARY KEY,
                    company_name TEXT,
                    region TEXT,
                    energy_type TEXT,
                    currency TEXT
                );
            """)

            # 2. Daily Time-Series Pricing Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    PRIMARY KEY (ticker, date),
                    FOREIGN KEY (ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
                );
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON daily_prices (date);")

            # 3. Quarterly Fundamentals Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quarterly_fundamentals (
                    ticker TEXT,
                    fiscal_date_ending TEXT,
                    ebit REAL,
                    total_assets REAL,
                    cash_and_equivalents REAL,
                    operating_cash_flow REAL,
                    capital_expenditures REAL,
                    net_income REAL,
                    total_revenue REAL,
                    PRIMARY KEY (ticker, fiscal_date_ending),
                    FOREIGN KEY (ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
                );
            """)
            conn.commit()
        logger.info("Schema initialization complete.")

    def register_tickers(self, ticker_metadata: List[Dict[str, str]]) -> None:
        """Registers asset metadata into the master ticker directory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for meta in ticker_metadata:
                cursor.execute("""
                    INSERT OR REPLACE INTO tickers (ticker, company_name, region, energy_type, currency)
                    VALUES (?, ?, ?, ?, ?);
                """, (meta['ticker'], meta['company_name'], meta['region'], meta['energy_type'], meta['currency']))
            conn.commit()
        logger.info(f"Registered {len(ticker_metadata)} global equities into master directory.")

    def fetch_and_store_prices(self, tickers: List[str], years_back: int = 5) -> None:
        """Pulls daily pricing metrics and updates the local timeseries database."""
        end_date = datetime.today()
        start_date = end_date - timedelta(days=years_back * 365)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching historical prices from {start_str} to {end_str}...")

        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_str, end=end_str, progress=False)
                if df.empty:
                    logger.warning(f"No pricing data retrieved for {ticker}")
                    continue
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.reset_index()
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                df['Ticker'] = ticker

                # THE FIX: Graceful fallback if 'Adj Close' is missing from the yfinance API
                if 'Adj Close' not in df.columns:
                    df['Adj Close'] = df['Close']

                pricing_data = df[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values.tolist()

                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.executemany("""
                        INSERT OR REPLACE INTO daily_prices (ticker, date, open, high, low, close, adj_close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """, pricing_data)
                    conn.commit()
                
                logger.info(f"Stored {len(pricing_data)} price records for {ticker}.")

            except Exception as e:
                logger.error(f"Failed to process pricing for {ticker}: {str(e)}")

    def fetch_and_store_fundamentals(self, tickers: List[str]) -> None:
        """Pulls and parses financial statements to store custom factor raw inputs."""
        logger.info("Fetching quarterly financial statements...")
        
        for ticker in tickers:
            try:
                yf_ticker = yf.Ticker(ticker)
                
                is_q = yf_ticker.quarterly_financials
                bs_q = yf_ticker.quarterly_balance_sheet
                cf_q = yf_ticker.quarterly_cashflow
                
                if is_q.empty or bs_q.empty or cf_q.empty:
                    logger.warning(f"Missing fundamental statement structures for {ticker}")
                    continue

                dates = is_q.columns
                
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    for date_obj in dates:
                        date_str = date_obj.strftime('%Y-%m-%d')
                        
                        def get_metric(df: pd.DataFrame, label: str) -> float:
                            # THE FIX: Ensure both the label AND the date exist in the dataframe
                            if label in df.index and date_obj in df.columns:
                                try:
                                    val = df.loc[label, date_obj]
                                    if isinstance(val, pd.Series):
                                        val = val.iloc[0]
                                    return float(val) if pd.notna(val) else 0.0
                                except KeyError:
                                    return 0.0
                            return 0.0

                        ebit = get_metric(is_q, 'EBIT')
                        total_assets = get_metric(bs_q, 'Total Assets')
                        cash = get_metric(bs_q, 'Cash And Cash Equivalents')
                        ocf = get_metric(cf_q, 'Operating Cash Flow')
                        capex = abs(get_metric(cf_q, 'Capital Expenditure')) 
                        net_income = get_metric(is_q, 'Net Income')
                        revenue = get_metric(is_q, 'Total Revenue')

                        cursor.execute("""
                            INSERT OR REPLACE INTO quarterly_fundamentals 
                            (ticker, fiscal_date_ending, ebit, total_assets, cash_and_equivalents, 
                             operating_cash_flow, capital_expenditures, net_income, total_revenue)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, (ticker, date_str, ebit, total_assets, cash, ocf, capex, net_income, revenue))
                    
                    conn.commit()
                logger.info(f"Updated quarterly fundamentals for {ticker}.")
                
            except Exception as e:
                logger.error(f"Failed to process fundamentals for {ticker}: {str(e)}")
# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    
    # 1. The Master Curated Global Universe
    # 1. The Global 300 Master Universe
    global_energy_universe = [
        # --- US MEGA & LARGE CAP E&P ---
        {"ticker": "XOM", "company_name": "Exxon Mobil", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "CVX", "company_name": "Chevron Corp", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "COP", "company_name": "ConocoPhillips", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "OXY", "company_name": "Occidental Petroleum", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "EOG", "company_name": "EOG Resources", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "HES", "company_name": "Hess Corp", "region": "US", "energy_type": "Fossil", "currency": "USD"},

        # --- US MID/SMALL CAP E&P ---
        {"ticker": "DVN", "company_name": "Devon Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "FANG", "company_name": "Diamondback Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "MRO", "company_name": "Marathon Oil", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "APA", "company_name": "APA Corporation", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "EQT", "company_name": "EQT Corp", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "CTRA", "company_name": "Coterra Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "AR", "company_name": "Antero Resources", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "RRC", "company_name": "Range Resources", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "MTDR", "company_name": "Matador Resources", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "PR", "company_name": "Permian Resources", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "MUR", "company_name": "Murphy Oil", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "SM", "company_name": "SM Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "CIVI", "company_name": "Civitas Resources", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "CHRD", "company_name": "Chord Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "VNOM", "company_name": "Viper Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "MGY", "company_name": "Magnolia Oil & Gas", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "CRK", "company_name": "Comstock Resources", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "CNX", "company_name": "CNX Resources", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "WTI", "company_name": "W&T Offshore", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "LPI", "company_name": "Laredo Petroleum", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "SBOW", "company_name": "SilverBow Res", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "NOG", "company_name": "Northern Oil", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "ESTE", "company_name": "Earthstone Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "CRC", "company_name": "California Res", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "VTLE", "company_name": "Vital Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},

        # --- US REFINING ---
        {"ticker": "VLO", "company_name": "Valero Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "MPC", "company_name": "Marathon Petroleum", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "PSX", "company_name": "Phillips 66", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "DINO", "company_name": "HF Sinclair", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "PBF", "company_name": "PBF Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "CVI", "company_name": "CVR Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "DK", "company_name": "Delek US", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "CLMT", "company_name": "Calumet", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "PARR", "company_name": "Par Pacific", "region": "US", "energy_type": "Fossil", "currency": "USD"},

        # --- US MIDSTREAM & INFRASTRUCTURE ---
        {"ticker": "KMI", "company_name": "Kinder Morgan", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "WMB", "company_name": "Williams Companies", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "EPD", "company_name": "Enterprise Products", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "ET", "company_name": "Energy Transfer", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "PAA", "company_name": "Plains All American", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "MPLX", "company_name": "MPLX LP", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "TRGP", "company_name": "Targa Resources", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "OKE", "company_name": "ONEOK Inc", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "LNG", "company_name": "Cheniere Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "CQP", "company_name": "Cheniere Partners", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "ENLC", "company_name": "EnLink Midstream", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "AM", "company_name": "Antero Midstream", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "CEQP", "company_name": "Crestwood Equity", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "NS", "company_name": "NuStar Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "WES", "company_name": "Western Midstream", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "KNTK", "company_name": "Kinetik Holdings", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "DTM", "company_name": "DT Midstream", "region": "US", "energy_type": "Fossil", "currency": "USD"},

        # --- US OILFIELD SERVICES (OFS) ---
        {"ticker": "SLB", "company_name": "Schlumberger", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "HAL", "company_name": "Halliburton", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "BKR", "company_name": "Baker Hughes", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "NOV", "company_name": "NOV Inc", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "FTI", "company_name": "TechnipFMC", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "WHD", "company_name": "Cactus Inc", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "PTEN", "company_name": "Patterson-UTI", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "HP", "company_name": "Helmerich & Payne", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "NBR", "company_name": "Nabors Industries", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "LBRT", "company_name": "Liberty Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "CHX", "company_name": "ChampionX", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "RES", "company_name": "RPC Inc", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "RNGR", "company_name": "Ranger Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "HLX", "company_name": "Helix Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "OII", "company_name": "Oceaneering Int", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "RIG", "company_name": "Transocean", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "VAL", "company_name": "Valaris", "region": "US", "energy_type": "Fossil", "currency": "USD"},

        # --- US COAL ---
        {"ticker": "BTU", "company_name": "Peabody Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "ARCH", "company_name": "Arch Resources", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "CEIX", "company_name": "CONSOL Energy", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "AMR", "company_name": "Alpha Metallurgical", "region": "US", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "HCC", "company_name": "Warrior Met Coal", "region": "US", "energy_type": "Fossil", "currency": "USD"},

        # --- US NUCLEAR & URANIUM ---
        {"ticker": "CEG", "company_name": "Constellation Energy", "region": "US", "energy_type": "Nuclear", "currency": "USD"},
        {"ticker": "EXC", "company_name": "Exelon Corp", "region": "US", "energy_type": "Nuclear", "currency": "USD"},
        {"ticker": "VST", "company_name": "Vistra Corp", "region": "US", "energy_type": "Nuclear", "currency": "USD"},
        {"ticker": "UUUU", "company_name": "Energy Fuels", "region": "US", "energy_type": "Nuclear", "currency": "USD"},
        {"ticker": "UEC", "company_name": "Uranium Energy", "region": "US", "energy_type": "Nuclear", "currency": "USD"},
        {"ticker": "LEU", "company_name": "Centrus Energy", "region": "US", "energy_type": "Nuclear", "currency": "USD"},
        {"ticker": "SMR", "company_name": "NuScale Power", "region": "US", "energy_type": "Nuclear", "currency": "USD"},

        # --- US RENEWABLES & UTILITIES ---
        {"ticker": "NEE", "company_name": "NextEra Energy", "region": "US", "energy_type": "Renewable", "currency": "USD"},
        {"ticker": "ENPH", "company_name": "Enphase Energy", "region": "US", "energy_type": "Renewable", "currency": "USD"},
        {"ticker": "FSLR", "company_name": "First Solar", "region": "US", "energy_type": "Renewable", "currency": "USD"},
        {"ticker": "SEDG", "company_name": "SolarEdge", "region": "US", "energy_type": "Renewable", "currency": "USD"},
        {"ticker": "RUN", "company_name": "Sunrun Inc", "region": "US", "energy_type": "Renewable", "currency": "USD"},
        {"ticker": "PLUG", "company_name": "Plug Power", "region": "US", "energy_type": "Renewable", "currency": "USD"},
        {"ticker": "BEP", "company_name": "Brookfield Renewable", "region": "US", "energy_type": "Renewable", "currency": "USD"},
        {"ticker": "CWEN", "company_name": "Clearway Energy", "region": "US", "energy_type": "Renewable", "currency": "USD"},
        {"ticker": "HASI", "company_name": "Hannon Armstrong", "region": "US", "energy_type": "Renewable", "currency": "USD"},
        {"ticker": "NOVA", "company_name": "Sunnova", "region": "US", "energy_type": "Renewable", "currency": "USD"},
        {"ticker": "ARRY", "company_name": "Array Technologies", "region": "US", "energy_type": "Renewable", "currency": "USD"},
        {"ticker": "SHLS", "company_name": "Shoals Tech", "region": "US", "energy_type": "Renewable", "currency": "USD"},
        {"ticker": "FCEL", "company_name": "FuelCell Energy", "region": "US", "energy_type": "Renewable", "currency": "USD"},
        {"ticker": "SO", "company_name": "Southern Company", "region": "US", "energy_type": "Renewable", "currency": "USD"},
        {"ticker": "DUK", "company_name": "Duke Energy", "region": "US", "energy_type": "Renewable", "currency": "USD"},
        {"ticker": "D", "company_name": "Dominion Energy", "region": "US", "energy_type": "Renewable", "currency": "USD"},

        # --- CANADA FOSSIL (.TO) ---
        {"ticker": "SU.TO", "company_name": "Suncor Energy", "region": "Canada", "energy_type": "Fossil", "currency": "CAD"},
        {"ticker": "CNQ.TO", "company_name": "Canadian Natural", "region": "Canada", "energy_type": "Fossil", "currency": "CAD"},
        {"ticker": "CVE.TO", "company_name": "Cenovus Energy", "region": "Canada", "energy_type": "Fossil", "currency": "CAD"},
        {"ticker": "ENB.TO", "company_name": "Enbridge", "region": "Canada", "energy_type": "Fossil", "currency": "CAD"},
        {"ticker": "TRP.TO", "company_name": "TC Energy", "region": "Canada", "energy_type": "Fossil", "currency": "CAD"},
        {"ticker": "TOU.TO", "company_name": "Tourmaline Oil", "region": "Canada", "energy_type": "Fossil", "currency": "CAD"},
        {"ticker": "ARX.TO", "company_name": "ARC Resources", "region": "Canada", "energy_type": "Fossil", "currency": "CAD"},
        {"ticker": "WCP.TO", "company_name": "Whitecap Res", "region": "Canada", "energy_type": "Fossil", "currency": "CAD"},
        {"ticker": "MEG.TO", "company_name": "MEG Energy", "region": "Canada", "energy_type": "Fossil", "currency": "CAD"},
        {"ticker": "BTE.TO", "company_name": "Baytex Energy", "region": "Canada", "energy_type": "Fossil", "currency": "CAD"},
        {"ticker": "POU.TO", "company_name": "Paramount Res", "region": "Canada", "energy_type": "Fossil", "currency": "CAD"},
        {"ticker": "CPG.TO", "company_name": "Crescent Point", "region": "Canada", "energy_type": "Fossil", "currency": "CAD"},
        {"ticker": "VET.TO", "company_name": "Vermilion Energy", "region": "Canada", "energy_type": "Fossil", "currency": "CAD"},
        {"ticker": "ERF.TO", "company_name": "Enerplus", "region": "Canada", "energy_type": "Fossil", "currency": "CAD"},
        {"ticker": "PBA", "company_name": "Pembina Pipeline", "region": "Canada", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "KEY.TO", "company_name": "Keyera", "region": "Canada", "energy_type": "Fossil", "currency": "CAD"},

        # --- CANADA NUCLEAR & RENEWABLE (.TO) ---
        {"ticker": "CCO.TO", "company_name": "Cameco Corp", "region": "Canada", "energy_type": "Nuclear", "currency": "CAD"},
        {"ticker": "NXE.TO", "company_name": "NexGen Energy", "region": "Canada", "energy_type": "Nuclear", "currency": "CAD"},
        {"ticker": "DML.TO", "company_name": "Denison Mines", "region": "Canada", "energy_type": "Nuclear", "currency": "CAD"},
        {"ticker": "NPI.TO", "company_name": "Northland Power", "region": "Canada", "energy_type": "Renewable", "currency": "CAD"},
        {"ticker": "INE.TO", "company_name": "Innergex", "region": "Canada", "energy_type": "Renewable", "currency": "CAD"},
        {"ticker": "BLX.TO", "company_name": "Boralex", "region": "Canada", "energy_type": "Renewable", "currency": "CAD"},
        {"ticker": "RNW.TO", "company_name": "TransAlta Renew", "region": "Canada", "energy_type": "Renewable", "currency": "CAD"},
        {"ticker": "AQN.TO", "company_name": "Algonquin Power", "region": "Canada", "energy_type": "Renewable", "currency": "CAD"},

        # --- UNITED KINGDOM (.L) ---
        {"ticker": "BP.L", "company_name": "BP plc", "region": "UK", "energy_type": "Fossil", "currency": "GBP"},
        {"ticker": "SHEL.L", "company_name": "Shell plc", "region": "UK", "energy_type": "Fossil", "currency": "GBP"},
        {"ticker": "CNA.L", "company_name": "Centrica", "region": "UK", "energy_type": "Nuclear", "currency": "GBP"},
        {"ticker": "SSE.L", "company_name": "SSE plc", "region": "UK", "energy_type": "Renewable", "currency": "GBP"},
        {"ticker": "ITM.L", "company_name": "ITM Power", "region": "UK", "energy_type": "Renewable", "currency": "GBP"},
        {"ticker": "HBR.L", "company_name": "Harbour Energy", "region": "UK", "energy_type": "Fossil", "currency": "GBP"},
        {"ticker": "ENQ.L", "company_name": "EnQuest", "region": "UK", "energy_type": "Fossil", "currency": "GBP"},
        {"ticker": "TLW.L", "company_name": "Tullow Oil", "region": "UK", "energy_type": "Fossil", "currency": "GBP"},
        {"ticker": "CNE.L", "company_name": "Capricorn Energy", "region": "UK", "energy_type": "Fossil", "currency": "GBP"},
        {"ticker": "DEC.L", "company_name": "Diversified Energy", "region": "UK", "energy_type": "Fossil", "currency": "GBP"},

        # --- EUROPEAN UNION & NORWAY FOSSIL ---
        {"ticker": "TTE.PA", "company_name": "TotalEnergies", "region": "EU", "energy_type": "Fossil", "currency": "EUR"},
        {"ticker": "ENI.MI", "company_name": "Eni S.p.A.", "region": "EU", "energy_type": "Fossil", "currency": "EUR"},
        {"ticker": "REP.MC", "company_name": "Repsol", "region": "EU", "energy_type": "Fossil", "currency": "EUR"},
        {"ticker": "GALP.LS", "company_name": "Galp Energia", "region": "EU", "energy_type": "Fossil", "currency": "EUR"},
        {"ticker": "OMV.VI", "company_name": "OMV AG", "region": "EU", "energy_type": "Fossil", "currency": "EUR"},
        {"ticker": "EQNR.OL", "company_name": "Equinor", "region": "Norway", "energy_type": "Fossil", "currency": "NOK"},
        {"ticker": "AKRBP.OL", "company_name": "Aker BP", "region": "Norway", "energy_type": "Fossil", "currency": "NOK"},
        {"ticker": "VAR.OL", "company_name": "Var Energi", "region": "Norway", "energy_type": "Fossil", "currency": "NOK"},
        {"ticker": "SUBC.OL", "company_name": "Subsea 7", "region": "Norway", "energy_type": "Fossil", "currency": "NOK"},
        {"ticker": "TGS.OL", "company_name": "TGS", "region": "Norway", "energy_type": "Fossil", "currency": "NOK"},
        {"ticker": "TEN.MI", "company_name": "Tenaris", "region": "EU", "energy_type": "Fossil", "currency": "EUR"},
        
        # --- EUROPEAN UNION & NORWAY RENEWABLE/NUCLEAR ---
        {"ticker": "NESTE.HE", "company_name": "Neste", "region": "EU", "energy_type": "Renewable", "currency": "EUR"},
        {"ticker": "ENEL.MI", "company_name": "Enel", "region": "EU", "energy_type": "Nuclear", "currency": "EUR"},
        {"ticker": "IBE.MC", "company_name": "Iberdrola", "region": "EU", "energy_type": "Renewable", "currency": "EUR"},
        {"ticker": "VWS.CO", "company_name": "Vestas Wind", "region": "EU", "energy_type": "Renewable", "currency": "DKK"},
        {"ticker": "ORSTED.CO", "company_name": "Orsted", "region": "EU", "energy_type": "Renewable", "currency": "DKK"},
        {"ticker": "SCATC.OL", "company_name": "Scatec", "region": "Norway", "energy_type": "Renewable", "currency": "NOK"},
        {"ticker": "NEL.OL", "company_name": "Nel ASA", "region": "Norway", "energy_type": "Renewable", "currency": "NOK"},
        {"ticker": "EDPR.LS", "company_name": "EDP Renovaveis", "region": "EU", "energy_type": "Renewable", "currency": "EUR"},
        {"ticker": "ENGIE.PA", "company_name": "Engie", "region": "EU", "energy_type": "Renewable", "currency": "EUR"},
        {"ticker": "RWE.DE", "company_name": "RWE AG", "region": "EU", "energy_type": "Renewable", "currency": "EUR"},

        # --- AUSTRALIA (.AX) ---
        {"ticker": "WDS.AX", "company_name": "Woodside Energy", "region": "Australia", "energy_type": "Fossil", "currency": "AUD"},
        {"ticker": "STO.AX", "company_name": "Santos Ltd", "region": "Australia", "energy_type": "Fossil", "currency": "AUD"},
        {"ticker": "ORG.AX", "company_name": "Origin Energy", "region": "Australia", "energy_type": "Fossil", "currency": "AUD"},
        {"ticker": "WHC.AX", "company_name": "Whitehaven Coal", "region": "Australia", "energy_type": "Fossil", "currency": "AUD"},
        {"ticker": "NHC.AX", "company_name": "New Hope Corp", "region": "Australia", "energy_type": "Fossil", "currency": "AUD"},
        {"ticker": "BPT.AX", "company_name": "Beach Energy", "region": "Australia", "energy_type": "Fossil", "currency": "AUD"},
        {"ticker": "KAR.AX", "company_name": "Karoon Energy", "region": "Australia", "energy_type": "Fossil", "currency": "AUD"},
        {"ticker": "PDN.AX", "company_name": "Paladin Energy", "region": "Australia", "energy_type": "Nuclear", "currency": "AUD"},
        {"ticker": "BOE.AX", "company_name": "Boss Energy", "region": "Australia", "energy_type": "Nuclear", "currency": "AUD"},
        {"ticker": "DYL.AX", "company_name": "Deep Yellow", "region": "Australia", "energy_type": "Nuclear", "currency": "AUD"},

        # --- JAPAN (.T) & SOUTH KOREA (.KS) ---
        {"ticker": "5020.T", "company_name": "ENEOS", "region": "Japan", "energy_type": "Fossil", "currency": "JPY"},
        {"ticker": "1605.T", "company_name": "INPEX", "region": "Japan", "energy_type": "Fossil", "currency": "JPY"},
        {"ticker": "9503.T", "company_name": "Kansai Electric", "region": "Japan", "energy_type": "Nuclear", "currency": "JPY"},
        {"ticker": "9501.T", "company_name": "TEPCO", "region": "Japan", "energy_type": "Nuclear", "currency": "JPY"},
        {"ticker": "9519.T", "company_name": "Renova", "region": "Japan", "energy_type": "Renewable", "currency": "JPY"},
        {"ticker": "5019.T", "company_name": "Idemitsu Kosan", "region": "Japan", "energy_type": "Fossil", "currency": "JPY"},
        {"ticker": "096770.KS", "company_name": "SK Innovation", "region": "South Korea", "energy_type": "Fossil", "currency": "KRW"},
        {"ticker": "015760.KS", "company_name": "KEPCO", "region": "South Korea", "energy_type": "Nuclear", "currency": "KRW"},
        {"ticker": "009830.KS", "company_name": "Hanwha Solutions", "region": "South Korea", "energy_type": "Renewable", "currency": "KRW"},

        # --- CHINA (HK) & INDIA (.NS) ---
        {"ticker": "0883.HK", "company_name": "CNOOC", "region": "China", "energy_type": "Fossil", "currency": "HKD"},
        {"ticker": "0857.HK", "company_name": "PetroChina", "region": "China", "energy_type": "Fossil", "currency": "HKD"},
        {"ticker": "0386.HK", "company_name": "Sinopec", "region": "China", "energy_type": "Fossil", "currency": "HKD"},
        {"ticker": "1816.HK", "company_name": "CGN Power", "region": "China", "energy_type": "Nuclear", "currency": "HKD"},
        {"ticker": "0916.HK", "company_name": "China Longyuan", "region": "China", "energy_type": "Renewable", "currency": "HKD"},
        {"ticker": "2208.HK", "company_name": "Goldwind", "region": "China", "energy_type": "Renewable", "currency": "HKD"},
        {"ticker": "0968.HK", "company_name": "Xinyi Solar", "region": "China", "energy_type": "Renewable", "currency": "HKD"},
        {"ticker": "RELIANCE.NS", "company_name": "Reliance Ind.", "region": "India", "energy_type": "Fossil", "currency": "INR"},
        {"ticker": "ONGC.NS", "company_name": "ONGC", "region": "India", "energy_type": "Fossil", "currency": "INR"},
        {"ticker": "ADANIGREEN.NS", "company_name": "Adani Green", "region": "India", "energy_type": "Renewable", "currency": "INR"},
        {"ticker": "NTPC.NS", "company_name": "NTPC", "region": "India", "energy_type": "Fossil", "currency": "INR"},
        {"ticker": "COALINDIA.NS", "company_name": "Coal India", "region": "India", "energy_type": "Fossil", "currency": "INR"},
        {"ticker": "SUZLON.NS", "company_name": "Suzlon Energy", "region": "India", "energy_type": "Renewable", "currency": "INR"},

        # --- EMERGING MARKETS (LATAM, Africa, Middle East) ---
        {"ticker": "PETR4.SA", "company_name": "Petrobras", "region": "Brazil", "energy_type": "Fossil", "currency": "BRL"},
        {"ticker": "PRIO3.SA", "company_name": "PRIO SA", "region": "Brazil", "energy_type": "Fossil", "currency": "BRL"},
        {"ticker": "EC", "company_name": "Ecopetrol", "region": "Colombia", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "YPF", "company_name": "YPF SA", "region": "Argentina", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "PAM", "company_name": "Pampa Energia", "region": "Argentina", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "TGS", "company_name": "Transportadora Gas", "region": "Argentina", "energy_type": "Fossil", "currency": "USD"},
        {"ticker": "ELET3.SA", "company_name": "Eletrobras", "region": "Brazil", "energy_type": "Nuclear", "currency": "BRL"},
        {"ticker": "2222.SR", "company_name": "Saudi Aramco", "region": "Saudi Arabia", "energy_type": "Fossil", "currency": "SAR"},
        {"ticker": "SOL.JO", "company_name": "Sasol", "region": "South Africa", "energy_type": "Fossil", "currency": "ZAR"}
    ]
    # Extract just the tickers for the yfinance download functions
    ticker_list = [item["ticker"] for item in global_energy_universe]

    # 2. Run the complete pipeline
    pipeline = EnergyDataPipeline(db_path="energy_asset_management.db")
    
    # Step A: Setup the master directory
    pipeline.register_tickers(global_energy_universe)
    
    # Step B: Download and store 5 years of pricing
    pipeline.fetch_and_store_prices(ticker_list, years_back=5)
    
    # Step C: Download and store quarterly fundamentals
    pipeline.fetch_and_store_fundamentals(ticker_list)
    
    logger.info("Phase 1 Master Data Pipeline executed successfully.")