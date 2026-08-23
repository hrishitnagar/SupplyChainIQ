import os
import joblib
import pandas as pd

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


MODEL_PATH = "models/xgboost_demand_model.pkl"


def prepare_data(df):

    from src.features import create_features

    df = create_features(df)

    feature_columns = [
        "day_of_week",
        "week_of_year",
        "month",
        "quarter",
        "year",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_30",
        "rolling_std_7",
        "price",
        "discount",
        "promotion",
        "inventory_level",
        "lead_time_days",
    ]

    df = df.dropna(
        subset=feature_columns
    )

    return df, feature_columns


def chronological_split(df):

    dates = sorted(
        df["date"].unique()
    )

    train_end = dates[
        int(len(dates) * 0.70)
    ]

    validation_end = dates[
        int(len(dates) * 0.85)
    ]

    train = df[
        df["date"] <= train_end
    ]

    validation = df[
        (df["date"] > train_end)
        &
        (df["date"] <= validation_end)
    ]

    test = df[
        df["date"] > validation_end
    ]

    return train, validation, test


def train_model(
    train,
    validation,
    features
):

    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(
        train[features],
        train["units_sold"],
        eval_set=[
            (
                validation[features],
                validation["units_sold"]
            )
        ],
        verbose=False,
    )

    return model


def evaluate_model(
    model,
    test,
    features
):

    predictions = model.predict(
        test[features]
    )

    mae = mean_absolute_error(
        test["units_sold"],
        predictions
    )

    rmse = mean_squared_error(
        test["units_sold"],
        predictions
    ) ** 0.5

    actual = test["units_sold"]

    # Avoid division by zero
    mask = actual != 0

    mape = (
        (
            abs(
                (
                    actual[mask]
                    -
                    predictions[mask]
                )
                /
                actual[mask]
            )
        )
        .mean()
        * 100
    )

    print(
        f"MAE: {mae:.2f}"
    )

    print(
        f"RMSE: {rmse:.2f}"
    )

    print(
        f"MAPE: {mape:.2f}%"
    )

    return predictions


def save_model(model):

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"Model saved to {MODEL_PATH}"
    )


if __name__ == "__main__":

    data = pd.read_csv(
        "data/processed/supply_chain_clean.csv"
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data, features = prepare_data(
        data
    )

    train, validation, test = (
        chronological_split(data)
    )

    print(
        "Train:",
        train.shape
    )

    print(
        "Validation:",
        validation.shape
    )

    print(
        "Test:",
        test.shape
    )

    model = train_model(
        train,
        validation,
        features
    )

    predictions = evaluate_model(
        model,
        test,
        features
    )

    save_model(model)