import os

import pandas as pd


RAW_PATH = "data/raw/supply_chain_data.csv"
PROCESSED_PATH = "data/processed/supply_chain_clean.csv"


def load_data(path=RAW_PATH):

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    return df


def inspect_data(df):

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nSummary:")
    print(df.describe())


def clean_data(df):

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Ensure correct date
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Remove invalid dates
    df = df.dropna(
        subset=["date"]
    )

    # Numerical columns
    numeric_columns = [
        "price",
        "discount",
        "units_sold",
        "inventory_level",
        "revenue",
        "lead_time_days",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove impossible negative values
    for column in [
        "price",
        "discount",
        "units_sold",
        "inventory_level",
        "revenue",
        "lead_time_days",
    ]:

        df.loc[
            df[column] < 0,
            column
        ] = None

    # Fill numerical missing values
    for column in numeric_columns:

        df[column] = df[column].fillna(
            df[column].median()
        )

    # Sort
    df = df.sort_values(
        [
            "product_id",
            "warehouse_id",
            "date"
        ]
    )

    return df


def save_data(df):

    # Create the output directory if it doesn't exist
    os.makedirs(
        os.path.dirname(PROCESSED_PATH),
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_PATH,
        index=False
    )

    print(
        f"Processed dataset saved to {PROCESSED_PATH}"
    )


if __name__ == "__main__":

    data = load_data()

    inspect_data(data)

    cleaned = clean_data(data)

    save_data(cleaned)

    print("\nFinal shape:")
    print(cleaned.shape)