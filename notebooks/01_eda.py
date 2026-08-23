import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = "data/processed/supply_chain_clean.csv"

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])


# -----------------------------
# Basic information
# -----------------------------

print(df.shape)

print(df.head())

print(df.describe())


# -----------------------------
# Revenue over time
# -----------------------------

monthly_revenue = (
    df.groupby(
        df["date"].dt.to_period("M")
    )["revenue"]
    .sum()
)

monthly_revenue.index = (
    monthly_revenue.index
    .astype(str)
)

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_revenue.index,
    monthly_revenue.values
)

plt.title("Monthly Revenue")

plt.xlabel("Month")

plt.ylabel("Revenue")

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()


# -----------------------------
# Monthly demand
# -----------------------------

monthly_demand = (
    df.groupby(
        df["date"].dt.to_period("M")
    )["units_sold"]
    .sum()
)

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_demand.index.astype(str),
    monthly_demand.values
)

plt.title("Monthly Demand")

plt.xlabel("Month")

plt.ylabel("Units Sold")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# -----------------------------
# Category performance
# -----------------------------

category_revenue = (
    df.groupby("category")["revenue"]
    .sum()
    .sort_values(
        ascending=False
    )
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=category_revenue.values,
    y=category_revenue.index
)

plt.title("Revenue by Category")

plt.xlabel("Revenue")

plt.ylabel("Category")

plt.tight_layout()

plt.show()


# -----------------------------
# Warehouse demand
# -----------------------------

warehouse_demand = (
    df.groupby("warehouse")["units_sold"]
    .sum()
    .sort_values(
        ascending=False
    )
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=warehouse_demand.index,
    y=warehouse_demand.values
)

plt.title("Demand by Warehouse")

plt.xlabel("Warehouse")

plt.ylabel("Units Sold")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# -----------------------------
# Promotion vs demand
# -----------------------------

promotion_demand = (
    df.groupby("promotion")["units_sold"]
    .mean()
)

print(
    "\nAverage demand by promotion status:"
)

print(promotion_demand)


# -----------------------------
# Correlation
# -----------------------------

numeric_columns = [
    "price",
    "discount",
    "promotion",
    "units_sold",
    "inventory_level",
    "revenue",
    "lead_time_days",
]

correlation = (
    df[numeric_columns]
    .corr()
)

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title(
    "Feature Correlation Matrix"
)

plt.tight_layout()

plt.show()