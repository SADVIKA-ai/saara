import logging
from scripts.batch_predict import main as run_batch_predict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting StockAI India Automated Daily Pipeline...")
    logging.info("Step 1: Refreshing Market Data & Features for IT Universe...")
    run_batch_predict()
    logging.info("Step 2: Daily Execution Complete. Reports saved to data/predictions/predictions_latest.csv")

if __name__ == "__main__":
    main()
