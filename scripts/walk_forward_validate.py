
import argparse
import json

from src.market_data import history
from src.walk_forward import walk_forward_validate


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward validation for StockAI."
    )
    parser.add_argument("--ticker", default="INFY.NS")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-test-rows", type=int, default=10)
    args = parser.parse_args()

    raw = history(args.ticker)

    result = walk_forward_validate(
        raw,
        folds=args.folds,
        min_test_rows=args.min_test_rows,
    )

    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
