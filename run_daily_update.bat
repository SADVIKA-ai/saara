@echo off
cd /d "C:\Users\lavet\Downloads\Equity flow\stock_market_ai_india_cloud\stockai_it_only_package\StockAI_India_ML_Engineer_Package"
"C:\Users\lavet\Downloads\Equity flow\stock_market_ai_india_cloud\stockai_it_only_package\StockAI_India_ML_Engineer_Package\.venv\Scripts\python.exe" -m scripts.daily_update --limit 0 --delay 0.4 --news >> "C:\Users\lavet\Downloads\Equity flow\stock_market_ai_india_cloud\stockai_it_only_package\StockAI_India_ML_Engineer_Package\outputs\logs\scheduled_daily_update.log" 2>&1
exit /b %ERRORLEVEL%
