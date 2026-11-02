import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone, timedelta
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.features import generate_features, SCHEMA_VERSION
from src.sentiment import get_sentiment_for_company

logger = logging.getLogger(__name__)

def evaluate_model(y_true_price, y_pred_price, y_baseline_price):
    mae = mean_absolute_error(y_true_price, y_pred_price)
    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    base_mae = mean_absolute_error(y_true_price, y_baseline_price)
    base_rmse = np.sqrt(mean_squared_error(y_true_price, y_baseline_price))
    
    actual_dir = np.sign(y_true_price.values - y_baseline_price.values)
    pred_dir = np.sign(y_pred_price - y_baseline_price.values)
    dir_acc = np.mean(actual_dir == pred_dir) * 100
    mean_actual = np.mean(y_true_price)
    
    return {
        "mae_holdout": round(float(mae), 4),
        "rmse_holdout": round(float(rmse), 4),
        "mae_holdout_pct_of_mean_actual": round(float((mae / mean_actual) * 100), 2) if mean_actual > 0 else 0.0,
        "directional_accuracy_pct": round(float(dir_acc), 2),
        "baseline_name": "LAST_CLOSE",
        "baseline_mae_holdout": round(float(base_mae), 4),
        "baseline_rmse_holdout": round(float(base_rmse), 4),
        "baseline_mae_pct_of_mean_actual": round(float((base_mae / mean_actual) * 100), 2) if mean_actual > 0 else 0.0,
        "model_beats_baseline_mae": bool(mae < base_mae),
        "model_beats_baseline_rmse": bool(rmse < base_rmse)
    }

def build_parameter_rationales(row: pd.Series, decision: str, expected_return: float, horizon: int, sentiment_data: dict, target_date_str: str) -> list:
    rationales = []
    
    ema9 = row.get('ema_9', 0)
    ema21 = row.get('ema_21', 0)
    rsi = row.get('rsi_14', 50)
    dbl_top = row.get('double_top_bearish_breakout', False)
    inv_hs = row.get('inverse_hs_bullish_reversal', False)
    
    if ema9 < ema21:
        rationales.append("BEARISH TREND: 9 EMA is below 21 EMA, indicating downward momentum.")
    else:
        rationales.append("BULLISH TREND: 9 EMA is above 21 EMA, signaling upward momentum.")
        
    if dbl_top:
        rationales.append("PATTERN WARNING: Double Top resistance breakdown detected near 20-day high.")
    elif inv_hs:
        rationales.append("PATTERN BREAKOUT: Inverse Head & Shoulders bullish pattern detected.")
        
    if rsi > 70:
        rationales.append("MOMENTUM WARNING: RSI (14) Overbought (>70), pullback risk present.")
    elif rsi < 30:
        rationales.append("MOMENTUM OPPORTUNITY: RSI (14) Oversold (<30), bounce potential.")
        
    sent_score = sentiment_data.get('sentiment_score', 0.0) if isinstance(sentiment_data, dict) else 0.0
    sent_label = sentiment_data.get('sentiment_label', 'unavailable') if isinstance(sentiment_data, dict) else 'unavailable'
    
    if sent_score > 0.2:
        rationales.append(f"FUNDAMENTAL CATALYST: Positive news sentiment detected ({sent_label}).")
    elif sent_score < -0.2:
        rationales.append(f"HEADWIND EVENT: Negative news or institutional selling pressure ({sent_label}).")
    else:
        rationales.append("FUNDAMENTAL STATUS: Inline performance with neutral news overlay.")

    rationales.append(f"TARGET DATE FORECAST: Forecast calculated for {target_date_str} ({horizon} calendar/trading days from latest close). Projected return: {expected_return:+.2f}%.")
    return rationales

def get_decision_label(expected_return: float, horizon_days: int) -> str:
    hurdle = min(0.50, max(0.015, 0.015 * np.sqrt(horizon_days)))
    if expected_return > hurdle:
        return "BUY"
    elif expected_return < -hurdle:
        return "SELL"
    return "HOLD"

def generate_prediction(symbol: str, company_name: str, exchange: str, df_raw: pd.DataFrame, horizon: int = 1, sentiment_data: dict = None, ticker: str = None):
    try:
        symbol = str(symbol).strip() if symbol and pd.notna(symbol) else ""
        company_name = str(company_name).strip() if company_name and pd.notna(company_name) else ""
        exchange = str(exchange).strip() if exchange and pd.notna(exchange) else ""
        ticker = str(ticker).strip() if ticker and pd.notna(ticker) else ""
        
        if not ticker and symbol:
            ticker = symbol.split('.')[0] if '.' in symbol else symbol
        if not symbol and ticker and exchange:
            symbol = f"{ticker}.NS" if exchange == "NSE" else f"{ticker}.BO"

        if df_raw is None or not isinstance(df_raw, pd.DataFrame) or len(df_raw) < 10:
            return {"provider_symbol": symbol, "company_name": company_name, "exchange": exchange, "ticker": ticker, "status": "FAILED", "message": "INSUFFICIENT_HISTORY"}
            
        df_features = generate_features(df_raw, horizon=horizon)
        
        if df_features.empty:
            return {"provider_symbol": symbol, "company_name": company_name, "exchange": exchange, "ticker": ticker, "status": "FAILED", "message": "FEATURE_GENERATION_FAILED"}

        train_df = df_features.dropna(subset=['target_return']).copy()
        inference_row = df_features.iloc[-1:].copy()
        
        min_required = max(5, min(20, len(df_raw) // 2))
        if len(train_df) < min_required:
            return {"provider_symbol": symbol, "company_name": company_name, "exchange": exchange, "ticker": ticker, "status": "FAILED", "message": f"INSUFFICIENT_USABLE_TRAINING_ROWS (found {len(train_df)}, required {min_required} for horizon {horizon})"}

        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'target_return', 'double_top_bearish_breakout', 'inverse_hs_bullish_reversal']
        features = [c for c in train_df.columns if c not in exclude_cols]
        
        split_idx = max(1, int(len(train_df) * 0.8))
        train_set = train_df.iloc[:split_idx]
        test_set = train_df.iloc[split_idx:]
        
        X_train, y_train = train_set[features], train_set['target_return']
        X_test, y_test_return = test_set[features], test_set['target_return']
        
        test_close_t = test_set['Close']
        y_test_price = test_close_t * (1 + y_test_return)
        baseline_test_price = test_close_t
        
        min_samples = max(1, min(5, len(train_set) // 2))
        model = HistGradientBoostingRegressor(random_state=42, max_iter=100, min_samples_leaf=min_samples)
        model.fit(X_train, y_train)
        
        pred_returns = model.predict(X_test)
        pred_prices = test_close_t * (1 + pred_returns)
        metrics = evaluate_model(y_test_price, pred_prices, baseline_test_price)
        
        X_latest = inference_row[features]
        predicted_return = float(model.predict(X_latest)[0])
        
        latest_close = float(inference_row['Close'].iloc[0])
        predicted_price = latest_close * (1 + predicted_return)
        predicted_change = predicted_price - latest_close
        
        latest_idx_val = inference_row.index[0]
        latest_market_date_obj = pd.to_datetime(latest_idx_val)
        latest_market_date_str = str(latest_market_date_obj.date())
        
        # Calculate EXACT Target Forecast Date (Latest Date + Horizon Days)
        target_forecast_date_obj = latest_market_date_obj + timedelta(days=horizon)
        target_forecast_date_str = str(target_forecast_date_obj.date())
        
        years_equiv = horizon / 252.0
        cagr_pct = (((1 + predicted_return) ** (1.0 / max(0.01, years_equiv))) - 1) * 100 if years_equiv >= 0.5 and (1 + predicted_return) > 0 else 0.0

        if not sentiment_data:
            sentiment_data = get_sentiment_for_company(ticker, company_name)
            
        sent_score = sentiment_data.get('sentiment_score', 0.0) if isinstance(sentiment_data, dict) else 0.0
        sent_status = sentiment_data.get('sentiment_status', 'not requested') if isinstance(sentiment_data, dict) else 'not requested'
        sent_adj_return = predicted_return + (sent_score * 0.01)
        
        decision = get_decision_label(sent_adj_return, horizon)
        horizon_label = f"{horizon}_DAY_CUSTOM_HORIZON" if horizon > 1 else "NEXT_OBSERVATION"

        rationales = build_parameter_rationales(
            row=inference_row.iloc[0],
            decision=decision,
            expected_return=predicted_return * 100,
            horizon=horizon,
            sentiment_data=sentiment_data,
            target_date_str=target_forecast_date_str
        )

        ema9_val = round(float(inference_row['ema_9'].iloc[0]), 2)
        ema21_val = round(float(inference_row['ema_21'].iloc[0]), 2)
        rsi14_val = round(float(inference_row['rsi_14'].iloc[0]), 2)
        macd_val = round(float(inference_row['macd'].iloc[0]), 2)

        return {
            "company_name": company_name,
            "exchange": exchange,
            "ticker": ticker,
            "provider_symbol": symbol,
            "status": "OK",
            "message": "",
            "sentiment_score": sent_score,
            "sentiment_label": sentiment_data.get('sentiment_label', 'neutral'),
            "news_count": sentiment_data.get('news_count', 0),
            "sentiment_status": sent_status,
            "latest_actual_close": round(latest_close, 2),
            "latest_market_date": latest_market_date_str,
            "target_forecast_date": target_forecast_date_str,
            "horizon_days": horizon,
            "predicted_next_close": round(predicted_price, 2),
            "predicted_change": round(predicted_change, 2),
            "model_expected_return_pct": round(predicted_return * 100, 2),
            "annualized_cagr_pct": round(cagr_pct, 2),
            "sentiment_adjusted_return_pct": round(sent_adj_return * 100, 2),
            "decision": decision,
            "rationales": rationales,
            "ema_9": ema9_val,
            "ema_21": ema21_val,
            "rsi_14": rsi14_val,
            "macd": macd_val,
            **metrics,
            "holdout_rows": len(test_set),
            "training_rows": len(train_set),
            "model": "HistGradientBoostingRegressor",
            "forecast_horizon": horizon_label,
            "feature_schema_version": SCHEMA_VERSION,
            "prediction_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "warning": f"Research estimate for {horizon_label} only. Not guaranteed investment advice.",
            "raw_history_rows": len(df_raw),
            "history_start_date": str(df_raw.index[0].date()) if hasattr(df_raw.index[0], 'date') else str(df_raw.index[0]),
            "raw_history_latest_date": str(df_raw.index[-1].date()) if hasattr(df_raw.index[-1], 'date') else str(df_raw.index[-1]),
            "minimum_training_rows": min_required
        }
    except Exception as e:
        logger.error(f"Error predicting {symbol}: {e}")
        return {
            "company_name": str(company_name),
            "exchange": str(exchange),
            "ticker": str(ticker),
            "provider_symbol": str(symbol),
            "status": "FAILED",
            "message": f"PIPELINE_ERROR: {str(e)}"
        }

def _extract_meta_from_obj(obj, meta_dict):
    if obj is None:
        return
    if isinstance(obj, (dict, pd.Series)) or hasattr(obj, 'to_dict'):
        try:
            d = obj.to_dict() if hasattr(obj, 'to_dict') else dict(obj)
            for k, v in d.items():
                if pd.notna(v) and v is not None:
                    k_str = str(k).strip().lower()
                    v_str = str(v).strip()
                    if k_str in ['company_name', 'company', 'name']:
                        meta_dict.setdefault('company_name', v_str)
                    elif k_str in ['exchange']:
                        meta_dict.setdefault('exchange', v_str)
                    elif k_str in ['ticker', 'security_code']:
                        meta_dict.setdefault('ticker', v_str)
                    elif k_str in ['provider_symbol', 'providersymbol', 'symbol']:
                        meta_dict.setdefault('provider_symbol', v_str)
        except Exception:
            pass

def predict(*args, **kwargs):
    df_raw = None
    meta = {}
    horizon = kwargs.get('horizon', 1)
    sentiment_data = kwargs.get('sentiment_data')

    for key in ['company_name', 'exchange', 'ticker', 'provider_symbol', 'symbol']:
        if kwargs.get(key) and pd.notna(kwargs[key]):
            meta[key] = str(kwargs[key]).strip()

    for kw_val in kwargs.values():
        _extract_meta_from_obj(kw_val, meta)

    for arg in args:
        if isinstance(arg, pd.DataFrame):
            df_raw = arg
        elif isinstance(arg, (dict, pd.Series)) or hasattr(arg, 'to_dict'):
            _extract_meta_from_obj(arg, meta)
        elif isinstance(arg, str):
            arg_str = arg.strip()
            if arg_str in ['NSE', 'BSE']:
                meta.setdefault('exchange', arg_str)
            elif '.' in arg_str or (len(arg_str) <= 12 and arg_str.isupper()):
                meta.setdefault('provider_symbol', arg_str)
            else:
                meta.setdefault('company_name', arg_str)
        elif isinstance(arg, int):
            horizon = arg

    if df_raw is None:
        df_raw = kwargs.get('df_raw') if kwargs.get('df_raw') is not None else kwargs.get('df')

    company_name = meta.get('company_name', '')
    exchange = meta.get('exchange', '')
    ticker = meta.get('ticker', '')
    symbol = meta.get('provider_symbol') or meta.get('symbol') or ticker

    if not symbol and ticker and exchange:
        symbol = f"{ticker}.NS" if exchange == "NSE" else f"{ticker}.BO"

    return generate_prediction(symbol=symbol, company_name=company_name, exchange=exchange, df_raw=df_raw, horizon=horizon, sentiment_data=sentiment_data, ticker=ticker)
