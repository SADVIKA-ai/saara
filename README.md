# StockAI India — ML Engineer Package

This is a continuation package based on the supplied checkpoint excerpts and `top_100_india_companies.xlsx`. The supplied ZIP is **not** a complete export of the existing Drive project: it contains a checkpoint README and two excerpts, not the complete source tree, datasets, notebook cells, or model binaries. This package does not claim to have audited unseen files or integrated the legacy model.

## Features
- Company search by name/ticker with a Streamlit dashboard.
- Incremental/resumable Yahoo Finance historical-price cache.
- Technical features, next-observed-session close regression, chronological holdout MAE/RMSE.
- Company-news retrieval through Google News RSS and a transparent lexicon sentiment baseline.
- BUY/SELL/HOLD classification using documented configurable return thresholds.
- Batch market update and batch prediction CSV outputs.
- Google Colab notebook for setup and running.

## Run
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
python -m unittest discover -s tests -v
python scripts/update_market_data.py --limit 5
python scripts/batch_predict.py --limit 5 --news
```
Upload `notebooks/colab_setup_and_run.ipynb` to Colab and upload this ZIP to `/content/StockAI_India_ML_Engineer_Package.zip`. The notebook copies the project to `/content/drive/MyDrive/stock_market_ai_india_cloud/stockai_ml_package` without overwriting the original project.

## Important limitations
- No model can guarantee 100% correct predictions or 100% data coverage.
- A company-master row is not proof that a security is currently active/tradable or has a supported price history.
- Prices come from Yahoo Finance, not an official exchange feed; coverage, timing, mapping and licensing may vary.
- New listings without history remain searchable but are marked unavailable for time-series forecasting; prices are never invented.
- Sentiment is a simple lexicon baseline, not a validated financial NLP model. In this baseline it modestly affects the decision score; it is not claimed to be a learned causal price effect.
- Colab runtimes disconnect. The notebook does not establish unattended daily scheduling.
- Forecasts and BUY/SELL/HOLD labels are research outputs, not financial advice or trading orders.

## Full universe
The included workbook has 100 company metadata rows. To expand, place a validated NSE/BSE master at `data/master/companies_master.csv`. Minimum required fields: `company_name`, `exchange`, and one of `ticker`, `symbol`, `provider_symbol`, `yahoo_symbol`. Recommended fields include `isin`, `security_code`, `listing_status`, `sector`, `industry`. Preserve exchange-specific rows and identifiers; do not merge NSE/BSE price histories into one series.

## Legacy model
The checkpoint reports an existing direction model expecting 38 features while the newer feature output had 39 columns. This package intentionally does not load the old model. Resolve schema and training metadata from the full project before integrating it.
