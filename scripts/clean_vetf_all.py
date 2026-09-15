"""Clean the combined Vanguard ETF list into an analysis-ready CSV.

Input  (read-only): raw_original_data/vetf_all.csv
Output:             data/vetf_all.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "raw_original_data" / "vetf_all.csv"
DST = ROOT / "data" / "vetf_all.csv"

# Dates shown in the Vanguard list page headers when the data was scraped.
PRICE_AS_OF = "2026-09-11"    # price, change
RETURNS_AS_OF = "2026-08-31"  # YTD, 1/5/10-yr, since-inception returns

RENAME = {
    "Symbol": "symbol",
    "Name": "name",
    "Asset class": "asset_class_raw",
    "Risk (1-5)": "risk_level",
    "Expense ratio (%)": "expense_ratio_pct",
    "SEC yield (%)": "sec_yield_pct",
    "SEC yield footnote": "sec_yield_footnote",
    "SEC yield period (days)": "sec_yield_period_days",
    "SEC yield as of (MM/DD/YYYY)": "sec_yield_as_of",
    "YTD return (%)": "return_ytd_pct",
    "1-yr return (%)": "return_1y_pct",
    "5-yr return (%)": "return_5y_pct",
    "10-yr return (%)": "return_10y_pct",
    "Since inception return (%)": "return_since_inception_pct",
    "Inception date (MM/DD/YYYY)": "inception_date",
    "Investment minimum ($)": "investment_min_usd",
    "Price ($)": "price_usd",
    "Change ($)": "price_change_usd",
    "Change (%)": "price_change_pct",
}

FLOAT_COLS = [
    "expense_ratio_pct", "sec_yield_pct", "return_ytd_pct", "return_1y_pct",
    "return_5y_pct", "return_10y_pct", "return_since_inception_pct",
    "investment_min_usd", "price_usd", "price_change_usd", "price_change_pct",
]


def main() -> None:
    df = pd.read_csv(SRC, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    df = df.rename(columns=RENAME)

    # Trim whitespace; blanks and dash placeholders become missing values.
    df = df.apply(lambda s: s.str.strip())
    df = df.replace({"": pd.NA, "—": pd.NA, "-": pd.NA})

    # "Bond - Short-term Investment" -> asset_class="Bond", category="Short-term Investment"
    parts = df["asset_class_raw"].str.split(" - ", n=1, expand=True)
    df["asset_class"] = parts[0]
    df["category"] = parts[1] if 1 in parts else pd.NA
    df = df.drop(columns="asset_class_raw")

    for col in FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["risk_level"] = pd.to_numeric(df["risk_level"], errors="raise").astype("Int64")
    df["sec_yield_period_days"] = pd.to_numeric(df["sec_yield_period_days"]).astype("Int64")

    for col in ["sec_yield_as_of", "inception_date"]:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")

    df["price_as_of"] = PRICE_AS_OF
    df["returns_as_of"] = RETURNS_AS_OF

    df = df[[
        "symbol", "name", "asset_class", "category", "risk_level", "expense_ratio_pct",
        "sec_yield_pct", "sec_yield_footnote", "sec_yield_period_days", "sec_yield_as_of",
        "return_ytd_pct", "return_1y_pct", "return_5y_pct", "return_10y_pct",
        "return_since_inception_pct", "returns_as_of", "inception_date",
        "investment_min_usd", "price_usd", "price_change_usd", "price_change_pct", "price_as_of",
    ]]

    assert df["symbol"].is_unique, "duplicate symbols"
    assert df["price_usd"].notna().all(), "missing prices"

    df.to_csv(DST, index=False, encoding="utf-8")
    print(f"wrote {DST} ({len(df)} rows, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
