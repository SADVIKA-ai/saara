from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

symbols = [
    "3IINFOLTD.BO",
    "AFFLE.BO",
    "BSOFT.BO",
    "COFORGE.BO",
    "CYBERTECH.BO",
    "DEVIT.BO",
    "ECLERX.BO",
    "HAPPSTMNDS.BO",
    "NETWEB.BO",
    "OMNIAX.BO",
    "SASKEN.BO",
    "SONATSOFTW.BO",
    "STARCOM.BO",
    "TANLA.BO",
]

for symbol in symbols:
    path = RAW_DIR / f"{symbol}.csv"

    print(f"\n{'=' * 65}")
    print(f"Symbol: {symbol}")

    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        continue

    try:
        df = pd.read_csv(path)

        required = ["Date", "Open", "High", "Low", "Close"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            print(f"Missing columns: {missing}")
            print(f"Available columns: {df.columns.tolist()}")
            continue

        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        prices = df[["Open", "High", "Low", "Close"]]

        bad = (
            prices.isna().any(axis=1)
            | (prices <= 0).any(axis=1)
            | (df["High"] < df[["Open", "Close"]].max(axis=1))
            | (df["Low"] > df[["Open", "Close"]].min(axis=1))
            | (df["High"] < df["Low"])
        )

        invalid = df.loc[bad, required]

        if invalid.empty:
            print("No invalid OHLC rows found.")
        else:
            print(f"Invalid rows found: {len(invalid)}")
            print(invalid.to_string(index=False))

    except Exception as exc:
        print(f"Inspection error: {exc}")

print("\nInspection complete. No raw files were modified.")