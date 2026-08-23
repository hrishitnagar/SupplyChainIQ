import pandas as pd


def generate_insights(
    inventory_df,
    demand_df
):

    insights = []

    # Critical stock risks
    critical = inventory_df[
        inventory_df["stockout_risk"]
        == "CRITICAL"
    ]

    if len(critical) > 0:

        insights.append({
            "type": "CRITICAL",
            "title": "Critical stock-out risk",
            "message": (
                f"{len(critical)} "
                "product-warehouse combinations "
                "have critically low inventory."
            )
        })

    # High risk
    high = inventory_df[
        inventory_df["stockout_risk"]
        == "HIGH"
    ]

    if len(high) > 0:

        insights.append({
            "type": "WARNING",
            "title": "High stock-out risk",
            "message": (
                f"{len(high)} "
                "product-warehouse combinations "
                "are below their reorder point."
            )
        })

    # Overstock
    overstock = inventory_df[
        inventory_df["overstock"]
    ]

    if len(overstock) > 0:

        insights.append({
            "type": "OPPORTUNITY",
            "title": "Potential overstock",
            "message": (
                f"{len(overstock)} "
                "product-warehouse combinations "
                "have more than 60 days of inventory."
            )
        })

    # Demand growth
    demand_df = demand_df.copy()

    demand_df["month"] = (
        demand_df["date"]
        .dt.to_period("M")
    )

    monthly = (
        demand_df
        .groupby("month")["units_sold"]
        .sum()
    )

    if len(monthly) >= 2:

        current = monthly.iloc[-1]

        previous = monthly.iloc[-2]

        growth = (
            (current - previous)
            /
            previous
            * 100
        )

        if growth > 10:

            insights.append({
                "type": "OPPORTUNITY",
                "title": "Demand growth",
                "message": (
                    f"Overall demand increased "
                    f"by {growth:.1f}% compared "
                    "with the previous month."
                )
            })

        elif growth < -10:

            insights.append({
                "type": "WARNING",
                "title": "Demand decline",
                "message": (
                    f"Overall demand decreased "
                    f"by {abs(growth):.1f}% compared "
                    "with the previous month."
                )
            })

    return insights


if __name__ == "__main__":

    inventory = pd.read_csv(
        "data/processed/inventory_metrics.csv"
    )

    demand = pd.read_csv(
        "data/processed/supply_chain_clean.csv"
    )

    demand["date"] = pd.to_datetime(
        demand["date"]
    )

    insights = generate_insights(
        inventory,
        demand
    )

    for insight in insights:

        print(
            f"[{insight['type']}] "
            f"{insight['title']}"
        )

        print(
            insight["message"]
        )

        print()