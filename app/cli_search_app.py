import os
import sys
import pandas as pd
import numpy as np
from src.market_data import get_market_data
from src.predict import predict

def render_ascii_chart(df: pd.DataFrame, title: str = "30-Day Price Trend"):
    if df.empty or 'Close' not in df.columns:
        return
    recent = df['Close'].tail(30).values
    if len(recent) < 5:
        return
        
    min_val, max_val = np.min(recent), np.max(recent)
    val_range = max(1e-5, max_val - min_val)
    height = 5
    width = len(recent)
    
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for col, val in enumerate(recent):
        norm = (val - min_val) / val_range
        row = int((1 - norm) * (height - 1))
        grid[row][col] = "*"
        
    print(f"\n  [ASCII PRICE CHART - {title}]")
    print(f"  ₹{max_val:.1f} | " + "".join(grid[0]))
    for r in range(1, height - 1):
        print("         | " + "".join(grid[r]))
    print(f"  ₹{min_val:.1f} | " + "".join(grid[height - 1]))
    print("         +" + "-" * width)

def run_search_app():
    master_path = os.path.join("data", "master", "companies_master.csv")
    if not os.path.exists(master_path):
        print(f"Error: Master file not found at {master_path}")
        return
        
    df_master = pd.read_csv(master_path)
    print("=" * 75)
    print("      STOCKAI INDIA — EXPLAINABLE QUANTITATIVE FORECASTING APP        ")
    print("=" * 75)
    
    while True:
        query = input("\nEnter 1-2+ letters of IT Company Name or Ticker (or 'q' to quit): ").strip()
        if query.lower() == 'q':
            print("Exiting StockAI India Dashboard. Goodbye!")
            break
            
        if len(query) < 1:
            print("Please enter at least 1 character to search.")
            continue
            
        mask = (
            df_master['company_name'].astype(str).str.contains(query, case=False, na=False) |
            df_master['ticker'].astype(str).str.contains(query, case=False, na=False)
        )
        matches = df_master[mask]
        
        if matches.empty:
            print(f"No IT companies found matching '{query}'. Try another search.")
            continue
            
        print(f"\nFound {len(matches)} matching IT company records:")
        matches_list = matches.to_dict('records')
        for idx, row in enumerate(matches_list[:10], 1):
            comp = row.get('company_name', 'Unknown')
            exch = row.get('exchange', '')
            tck = row.get('ticker', '')
            psym = row.get('provider_symbol', '')
            print(f"  [{idx}] {comp} | Exchange: {exch} | Ticker: {tck} | Symbol: {psym}")
            
        if len(matches_list) > 10:
            print(f"  ... and {len(matches_list) - 10} more matches. Refine query for exact company.")
            
        choice = input("\nSelect company number [1-10] (or press Enter to cancel): ").strip()
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(matches_list[:10]):
            continue
            
        selected = matches_list[int(choice) - 1]
        print(f"\n---> Selected: {selected.get('company_name')} ({selected.get('exchange')}:{selected.get('ticker')})")
        
        hz_input = input("Enter number of days for forecast (e.g. 10, 13, 30, 100, 252, 2000) [Default = 10]: ").strip()
        horizon = 10
        if hz_input.isdigit() and int(hz_input) > 0:
            horizon = int(hz_input)
            
        psym = str(selected.get('provider_symbol', ''))
        if not psym or psym == 'nan':
            psym = f"{selected.get('ticker')}.NS" if selected.get('exchange') == 'NSE' else f"{selected.get('ticker')}.BO"
            
        print(f"\nFetching data and generating {horizon}-Day quantitative analysis for {psym}...")
        
        try:
            df_raw = get_market_data(psym, refresh=True)
            res = predict(
                symbol=psym,
                company_name=selected.get('company_name', ''),
                exchange=selected.get('exchange', ''),
                df_raw=df_raw,
                horizon=horizon
            )
            
            if not df_raw.empty:
                render_ascii_chart(df_raw, f"{selected.get('company_name')} Price Action")
            
            print("\n" + "=" * 75)
            print(f" PREDICTION REPORT: {res.get('company_name')} ({res.get('exchange')}:{res.get('ticker')})")
            print("=" * 75)
            print(f"  Status                     : {res.get('status')}")
            if res.get('status') == 'OK':
                print(f"  Latest Market Close Date   : {res.get('latest_market_date')} (Close: ₹{res.get('latest_actual_close')})")
                print(f"  TARGET FORECAST DATE       : {res.get('target_forecast_date')} (After {res.get('horizon_days')} Days)")
                print(f"  Predicted Price on Date    : ₹{res.get('predicted_next_close')} (Change: ₹{res.get('predicted_change')})")
                print(f"  Expected Return on Date    : {res.get('model_expected_return_pct')}%")
                if res.get('annualized_cagr_pct', 0) > 0:
                    print(f"  Annualized CAGR Return     : {res.get('annualized_cagr_pct')}% p.a.")
                print(f"  FINAL SIGNAL DECISION      : *** [{res.get('decision')}] ***")
                print("-" * 75)
                print("  [PARAMETER RATIONALE & TRADING LOGIC]")
                for r_idx, rat in enumerate(res.get('rationales', []), 1):
                    print(f"   {r_idx}. {rat}")
                print("-" * 75)
                print("  [TECHNICAL INDICATOR SUMMARY]")
                print(f"   - 9 EMA / 21 EMA           : ₹{res.get('ema_9')} / ₹{res.get('ema_21')}")
                print(f"   - 14-Day RSI               : {res.get('rsi_14')}")
                print(f"   - MACD Momentum            : {res.get('macd')}")
                print("-" * 75)
                print("  [MODEL VALIDATION METRICS]")
                print(f"   - Model Holdout MAE        : ₹{res.get('mae_holdout')} (Baseline MAE: ₹{res.get('baseline_mae_holdout')})")
                print(f"   - Beats Baseline MAE       : {res.get('model_beats_baseline_mae')}")
                print(f"   - Directional Accuracy     : {res.get('directional_accuracy_pct')}%")
            else:
                print(f"  Message                    : {res.get('message')}")
            print("=" * 75)
        except Exception as e:
            print(f"Error processing prediction: {e}")

if __name__ == "__main__":
    run_search_app()
