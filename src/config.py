from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.getenv("STOCKAI_DATA_DIR",str(ROOT/"data")))
MASTER_CSV=DATA/"master"/"companies_master.csv"
MASTER_XLSX=ROOT/"data"/"master"/"top_100_india_companies.xlsx"
RAW=DATA/"raw"; NEWS=DATA/"news"; PRED=DATA/"predictions"; LOGS=ROOT/"outputs"/"logs"
INITIAL_YEARS=10; MIN_ROWS=120; TEST_FRACTION=0.2
BUY_THRESHOLD=0.005; SELL_THRESHOLD=-0.005
for p in (RAW,NEWS,PRED,LOGS): p.mkdir(parents=True,exist_ok=True)
