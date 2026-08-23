import pandas as pd


def create_features(df):

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    # Calendar features
    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["week_of_year"] = (
        df["date"].dt.isocalendar().week
        .astype(int)
    )

    df["month"] = (
        df["date"].dt.month
    )

    df["quarter"] = (
        df["date"].dt.quarter
    )

    df["year"] = (
        df["date"].dt.year
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Sort before creating lag features
    df = df.sort_values(
        [
            "product_id",
            "warehouse_id",
            "date"
        ]
    )

    group = [
        "product_id",
        "warehouse_id"
    ]

    # Lag features
    df["lag_1"] = (
        df.groupby(group)["units_sold"]
        .shift(1)
    )

    df["lag_7"] = (
        df.groupby(group)["units_sold"]
        .shift(7)
    )

    df["lag_14"] = (
        df.groupby(group)["units_sold"]
        .shift(14)
    )

    df["lag_30"] = (
        df.groupby(group)["units_sold"]
        .shift(30)
    )

    # Rolling features
    df["rolling_mean_7"] = (
        df.groupby(group)["units_sold"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7)
            .mean()
        )
    )

    df["rolling_mean_30"] = (
        df.groupby(group)["units_sold"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(30)
            .mean()
        )
    )

    df["rolling_std_7"] = (
        df.groupby(group)["units_sold"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7)
            .std()
        )
    )

    return df