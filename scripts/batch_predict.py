import os
import argparse
import pandas as pd
import logging
from datetime import datetime, timezone
from src.market_data import get_market_data
from src.predict import predict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(limit=None, refresh=False, news=False):
    master_path = os.path.join("data", "master", "companies_master.csv")
    if not os.path.exists(master_path):
        logger.error(f"Master file not found: {master_path}")
        return

    df_master = pd.read_csv(master_path)
    if limit:
        df_master = df_master.head(limit)

    results = []
    total = len(df_master)

    for idx, row in df_master.iterrows():
        company_name = str(row.get('company_name', '')).strip() if pd.notna(row.get('company_name')) else ""
        exchange = str(row.get('exchange', '')).strip() if pd.notna(row.get('exchange')) else ""
        ticker = str(row.get('ticker', '')).strip() if pd.notna(row.get('ticker')) else ""
        provider_symbol = str(row.get('provider_symbol', '')).strip() if pd.notna(row.get('provider_symbol')) else ""
        
        if not provider_symbol or provider_symbol == 'nan':
            provider_symbol = f"{ticker}.NS" if exchange == "NSE" else f"{ticker}.BO"

        logger.info(f"[{idx+1}/{total}] Processing {company_name} ({exchange}:{ticker})...")

        # Skip Rights-Entitlement Securities
        if any(suffix in ticker for suffix in ['-RE', '-RE1']) or ('RIGHTS' in company_name.upper() and 'ENTITLEMENT' in company_name.upper()):
            res = {
                "company_name": company_name,
                "exchange": exchange,
                "ticker": ticker,
                "provider_symbol": provider_symbol,
                "status": "SKIPPED_NON_ORDINARY_SECURITY",
                "message": "Rights-entitlement security; ordinary-share prediction skipped."
            }
            results.append(res)
            continue

        try:
            df_raw = get_market_data(provider_symbol)
            if df_raw is None or df_raw.empty:
                res = {
                    "company_name": company_name,
                    "exchange": exchange,
                    "ticker": ticker,
                    "provider_symbol": provider_symbol,
                    "status": "NO_PRICE_HISTORY",
                    "message": "No historical price data available."
                }
            else:
                res = predict(
                    symbol=provider_symbol,
                    company_name=company_name,
                    exchange=exchange,
                    ticker=ticker,
                    df_raw=df_raw
                )
        except Exception as e:
            res = {
                "company_name": company_name,
                "exchange": exchange,
                "ticker": ticker,
                "provider_symbol": provider_symbol,
                "status": "FAILED",
                "message": f"PIPELINE_ERROR: {str(e)}"
            }

        results.append(res)

    df_res = pd.DataFrame(results)
    
    os.makedirs(os.path.join("data", "predictions"), exist_ok=True)
    latest_path = os.path.join("data", "predictions", "predictions_latest.csv")
    export_path = os.path.join("data", "predictions", "stockai_it_predictions_export.csv")
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ts_path = os.path.join("data", "predictions", f"predictions_{timestamp}.csv")
    
    df_res.to_csv(latest_path, index=False)
    df_res.to_csv(ts_path, index=False)
    df_res.to_csv(export_path, index=False)
    
    print("\n" + "=" * 70)
    print("                      STATUS SUMMARY                           ")
    print("=" * 70)
    print(df_res['status'].value_counts().to_string())
    print("\nSaved output predictions to:")
    print(f"  - {latest_path}")
    print(f"  - {export_path}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run StockAI India Batch Predictions")
    parser.add_argument("--limit", type=int, help="Limit number of companies to process")
    parser.add_argument("--refresh", action="store_true", help="Refresh market data")
    parser.add_argument("--news", action="store_true", help="Include news sentiment")
    args = parser.parse_args()
    
    main(limit=args.limit, refresh=args.refresh, news=args.news)
