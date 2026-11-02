
import pandas as pd

from src.market_data import history


df = history("INFY.NS").copy()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print("Rows:", len(df))
print("Invalid dates:", df["Date"].isna().sum())
print("Dates sorted:", df["Date"].is_monotonic_increasing)
print("Duplicate dates:", df["Date"].duplicated().sum())

print("\nMissing values:")
print(df.isna().sum().to_string())

price_columns = ["Open", "High", "Low", "Close", "Adj_Close"]

print("\nNon-positive prices:")
print((df[price_columns] <= 0).sum().to_string())

print("\nOHLC consistency violations:")
violations = {
    "High_below_Open_or_Close": int(
        (df["High"] < df[["Open", "Close"]].max(axis=1)).sum()
    ),
    "Low_above_Open_or_Close": int(
        (df["Low"] > df[["Open", "Close"]].min(axis=1)).sum()
    ),
    "High_below_Low": int((df["High"] < df["Low"]).sum()),
}
print(violations)

print("\nLargest absolute close-to-close changes (%):")
changes = df["Close"].pct_change(fill_method=None).mul(100).abs()
print(changes.nlargest(10).to_string())

print("\nLargest absolute adjusted-close changes (%):")
adjusted_changes = (
    df["Adj_Close"].pct_change(fill_method=None).mul(100).abs()
)
print(adjusted_changes.nlargest(10).to_string())
