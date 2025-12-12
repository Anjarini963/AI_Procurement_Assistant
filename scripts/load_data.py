import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient


def load_env():
    load_dotenv()


def get_collection():
    load_env()
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB_NAME", "procurement_db")
    collection_name = os.getenv("MONGODB_COLLECTION_NAME", "ca_procurements")

    client = MongoClient(uri)
    db = client[db_name]
    return db[collection_name]


def infer_numeric(df: pd.DataFrame, currency_columns, numeric_columns):
    """Convert columns to numeric, handling currency strings for price fields."""
    # Handle currency columns (Unit Price, Total Price)
    for col in currency_columns:
        if col in df.columns:
            # Convert to string first, then clean currency formatting
            df[col] = df[col].astype(str)
            # Remove $, commas, and spaces
            df[col] = df[col].str.replace('$', '', regex=False)
            df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = df[col].str.replace(' ', '', regex=False)
            # Convert to numeric (NaN for invalid values)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Handle regular numeric columns (Quantity, Fiscal Year)
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def main(csv_path: str, chunksize: int = 5000):
    collection = get_collection()

    # Optional: clear existing data for idempotent runs
    collection.delete_many({})

    total_inserted = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        # Convert numeric fields - currency fields need special cleaning
        chunk = infer_numeric(
            chunk,
            currency_columns=["Unit Price", "Total Price"],  # These have $, commas, spaces
            numeric_columns=["Quantity"],  # Regular numeric fields (Fiscal Year is a string like "2013-2014")
        )

        records = chunk.to_dict(orient="records")
        if records:
            collection.insert_many(records)
            total_inserted += len(records)
            print(f"Inserted {total_inserted} records...", flush=True)

    print(f"Done. Inserted total of {total_inserted} records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load CA procurement CSV into MongoDB.")
    parser.add_argument("--csv-path", required=True, help="Path to the procurement CSV file.")
    args = parser.parse_args()
    main(args.csv_path)


