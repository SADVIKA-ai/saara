# Architecture and limits
Data flow: company master -> symbol mapping -> incremental OHLCV cache -> point-in-time technical features -> chronological holdout -> next observed session close estimate -> sentiment-adjusted signal -> dashboard/CSV.
- Company master: supplied workbook or validated CSV.
- Price source: Yahoo Finance/yfinance, not an official exchange feed.
- News source: Google News RSS search; entity matching may be imperfect.
- Baseline model: per-security HistGradientBoostingRegressor, not the old checkpoint model.
- Target: next observed row close, avoiding a next-calendar-day assumption.
- Evaluation: chronological split; MAE/RMSE shown, but compare against naive baselines before production.
- BUY/SELL/HOLD: +/-0.5% default configurable thresholds; illustrative, not validated investment advice.
- Sentiment: simple lexicon with a small heuristic adjustment. It is not jointly trained with price regression; compare with/without sentiment on identical out-of-sample splits before making research claims.
- No history means no time-series forecast. Do not invent IPO prices or promise predictions for every security.
Production checklist: export complete Drive project; inspect legacy model feature_names and training schema; reconcile 38-vs-39 mismatch; validate exchange master and conflicts; verify BSE/NSE provider symbols; evaluate many securities/market regimes and transaction costs; schedule outside transient Colab runtime.
