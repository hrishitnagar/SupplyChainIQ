import numpy as np
import pandas as pd


def calculate_inventory_metrics(df):

    df = df.copy()

    # Average daily demand
    average_demand = (
        df.groupby(
            [
                "product_id",
                "warehouse_id"
            ]
        )["units_sold"]
        .mean()
        .reset_index(
            name="average_daily_demand"
        )
    )

    # Demand volatility
    demand_std = (
        df.groupby(
            [
                "product_id",
                "warehouse_id"
            ]
        )["units_sold"]
        .std()
        .reset_index(
            name="demand_std"
        )
    )

    # Latest inventory
    latest_inventory = (
        df.sort_values("date")
        .groupby(
            [
                "product_id",
                "warehouse_id"
            ]
        )
        .tail(1)
        [
            [
                "product_id",
                "warehouse_id",
                "inventory_level",
                "lead_time_days",
            ]
        ]
    )

    result = (
        latest_inventory
        .merge(
            average_demand,
            on=[
                "product_id",
                "warehouse_id"
            ]
        )
        .merge(
            demand_std,
            on=[
                "product_id",
                "warehouse_id"
            ]
        )
    )

    # Safety stock
    service_factor = 1.65

    result["safety_stock"] = (
        service_factor
        *
        result["demand_std"]
        *
        np.sqrt(
            result["lead_time_days"]
        )
    )

    # Reorder point
    result["reorder_point"] = (
        result["average_daily_demand"]
        *
        result["lead_time_days"]
        +
        result["safety_stock"]
    )

    # Days of inventory
    result["days_of_inventory"] = (
        result["inventory_level"]
        /
        result["average_daily_demand"]
    )

    # Risk
    def risk(row):

        if (
            row["inventory_level"]
            <= row["reorder_point"] * 0.5
        ):
            return "CRITICAL"

        if (
            row["inventory_level"]
            <= row["reorder_point"]
        ):
            return "HIGH"

        if (
            row["inventory_level"]
            <= row["reorder_point"] * 1.5
        ):
            return "MEDIUM"

        return "LOW"

    result["stockout_risk"] = (
        result.apply(
            risk,
            axis=1
        )
    )

    # Overstock
    result["overstock"] = (
        result["inventory_level"]
        >
        result["average_daily_demand"]
        * 60
    )

    return result


if __name__ == "__main__":

    data = pd.read_csv(
        "data/processed/supply_chain_clean.csv"
    )

    result = calculate_inventory_metrics(
        data
    )

    print(
        result.head(20)
    )

    result.to_csv(
        "data/processed/inventory_metrics.csv",
        index=False
    )