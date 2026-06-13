# Quantitative Energy Asset Management Engine

A Python-based, quantitative screening and backtesting engine focused on the global energy sector (198 equities from countries accross the world). This project models the end-to-end workflow of a quantitative asset management desk, from raw data engineering to factor-based alpha generation and historical risk analysis.

## 🎯 Project Goal
The primary objective of this project is to systematically identify mispriced, high-quality companies within the global energy market (Traditional Fossil Fuels vs. Clean/Renewable Energy) and rigorously backtest these factor models against macroeconomic benchmarks to evaluate risk-adjusted returns (Alpha).

## 🏗️ Architecture & Methodology (Phases 1 - 3)

### Phase 1: The Data Engineering Pipeline (`pipeline.py`)
This phase built an automated pipeline to scrape, clean, and structure financial data into a local SQLite database (`energy_asset_management.db`).
* **Breadth:** Tracks a curated universe of 198 global energy equities across the US, Canada, Europe, Asia, and other Emerging Markets.
* **Depth:** Downloads 5 years of daily pricing timeseries and quarterly (semi-annual when not available) fundamental accounting statements.
* **Resilience:** Engineered with strict `try/except` safety nets to gracefully handle API rate limits, missing data (e.g., semi-annual European and Australian reporting), and delisted tickers without crashing the pipeline.

### Phase 2: The Quantitative Factor Screener (`02b_screener_composite_live.py`)
This engine connects to the local database, loads the data into Pandas DataFrames, and evaluates companies on two distinct fundamental axes using cross-sectional percentile ranking:
1. **Quality Factor (ROA):** Calculated as `EBIT / Total Assets`. Screens for management teams that highly efficiently deploy capital intensive assets.
2. **Value Factor (FCF Yield):** Calculated as `(Operating Cash Flow - CapEx) / Live Market Cap`. Dynamically fetches live market caps via API to find companies generating massive cash relative to their current trading price.
* **Output:** Generates a blended "Conviction Score" (50% Value / 50% Quality) and splits the Top 10 recommendations into a **Fossil Portfolio** and a **Clean Energy Transition Portfolio**.

### Phase 3: The Point-in-Time Backtester (`03_backtester.py`)
A custom-built historical simulation engine to evaluate the factor model's performance from 2022 to 2026.
* **Bias Prevention:** Enforces a strict 45-day Point-in-Time lag on all fundamental data to perfectly simulate the delay in SEC 10-Q filings and eliminate "Look-Ahead Bias."
* **Mechanics:** Equal-weights the Top 10 stocks in each portfolio, holds for 30 days, and rebalances monthly.
* **Analytics:** Generates an institutional tearsheet comparing the strategies against the **S&P 500 (SPY)** benchmark, calculating Total Return, Annualized Volatility, Max Drawdown, and the Sharpe Ratio.

---

## 📊 Key Findings (2022 - 2026 Backtest)
* **Risk Mitigation:** While the pure Value/Quality factor model underperformed the massive tech-driven S&P 500 rally in total return, it successfully engineered risk out of the portfolio. The Fossil portfolio reduced Maximum Drawdown by roughly 800 basis points compared to the broader market (-18.9% vs -26.9%).
* **Regime Analysis:** The backtest revealed that in the recent macroeconomic environment, pure fundamental "Value" in energy acted as a defensive, low-volatility factor rather than an aggressive Alpha generator. 
* **ESG Outperformance:** The mathematically identical factor model performed significantly better on the Clean Energy universe (+8.2% return) than the traditional Fossil universe (-10.4% return) over the 5-year period.

---

## ⚠️ Current Limitations
As an active quantitative research project, the current iteration has known limitations:
1. **Zero Transaction Costs:** The current backtester does not penalize the portfolio for bid-ask spread slippage or broker commissions during the monthly rebalancing.
2. **Benchmark Mismatch:** Comparing a 10-stock global energy portfolio to the S&P 500 (which is heavily weighted toward US Technology) creates a distorted relative performance metric. A sector-specific benchmark (like XLE or IXC) would provide a cleaner Alpha measurement.
3. **Data Quality:** The model relies on the free `yfinance` API, which occasionally struggles with standardized fundamental mapping for Emerging Market ADRs and international semi-annual filers.

---

## 🚀 Next Steps (Phase 4)
* **Factor Expansion (Momentum):** Update the screener and backtester to include a 6-month Price Momentum factor. Blending Momentum with Value historically helps avoid "Value Traps" and improves the Sharpe Ratio.
* **Risk Parity Weighting:** Replace the current Equal-Weight allocation with an Inverse-Volatility weighting scheme to further minimize portfolio drawdowns.
* **Fundamental DCF Integration:** Build an automated Discounted Cash Flow (DCF) calculator to supplement the relative quantitative multiples with absolute intrinsic valuation metrics.

---

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/MaximeGKDrouin/energy-valuation-model.git](https://github.com/MaximeGKDrouin/energy-valuation-model.git)
   cd energy-valuation-model
