import os
import numpy as np
import pandas as pd


np.random.seed(42)

OUTPUT_PATH = "data/raw/supply_chain_data.csv"

os.makedirs("data/raw", exist_ok=True)

# -----------------------------
# Configuration
# -----------------------------

START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

products = [
    ("P001", "Laptop", "Electronics", 65000),
    ("P002", "Smartphone", "Electronics", 30000),
    ("P003", "Tablet", "Electronics", 22000),
    ("P004", "Keyboard", "Accessories", 1800),
    ("P005", "Mouse", "Accessories", 900),
    ("P006", "Monitor", "Electronics", 15000),
    ("P007", "Headphones", "Accessories", 2500),
    ("P008", "Printer", "Office", 12000),
    ("P009", "Office Chair", "Furniture", 8500),
    ("P010", "Desk", "Furniture", 12000),
    ("P011", "USB Hub", "Accessories", 1200),
    ("P012", "Webcam", "Accessories", 3500),
    ("P013", "Router", "Networking", 4500),
    ("P014", "SSD", "Storage", 6000),
    ("P015", "External HDD", "Storage", 5500),
]

warehouses = {
    "W001": "Indore",
    "W002": "Mumbai",
    "W003": "Delhi",
    "W004": "Bangalore",
}

suppliers = {
    "S001": "TechSupply",
    "S002": "GlobalComponents",
    "S003": "PrimeDistributors",
    "S004": "FastLogistics",
}

dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

rows = []

for product_id, product_name, category, base_price in products:

    base_demand = np.random.randint(15, 100)

    for warehouse_id, warehouse_name in warehouses.items():

        supplier_id = np.random.choice(list(suppliers.keys()))

        lead_time = np.random.randint(3, 15)

        inventory = np.random.randint(
            base_demand * 10,
            base_demand * 30
        )

        for date in dates:

            # Weekly seasonality
            weekday_factor = {
                0: 0.95,
                1: 1.00,
                2: 1.02,
                3: 1.05,
                4: 1.10,
                5: 1.20,
                6: 1.15,
            }[date.weekday()]

            # Monthly seasonality
            month_factor = 1.0

            if date.month in [10, 11, 12]:
                month_factor = 1.25

            elif date.month in [4, 5]:
                month_factor = 1.10

            # Random promotion
            promotion = np.random.choice(
                [0, 1],
                p=[0.85, 0.15]
            )

            discount = (
                np.random.uniform(5, 25)
                if promotion
                else 0
            )

            promotion_factor = (
                1 + discount / 100
                if promotion
                else 1
            )

            # Trend
            days_since_start = (
                date - dates[0]
            ).days

            trend_factor = (
                1 + days_since_start / 3000
            )

            demand = (
                base_demand
                * weekday_factor
                * month_factor
                * promotion_factor
                * trend_factor
            )

            demand *= np.random.normal(
                1.0,
                0.15
            )

            units_sold = max(
                0,
                int(round(demand))
            )

            # Price variation
            price = base_price * np.random.uniform(
                0.95,
                1.05
            )

            revenue = units_sold * price

            # Inventory movement
            inventory -= units_sold

            # Restocking
            if inventory < base_demand * lead_time:

                replenishment = int(
                    base_demand * np.random.randint(
                        10,
                        20
                    )
                )

                inventory += replenishment

            inventory = max(
                0,
                int(inventory)
            )

            stockout = (
                1
                if inventory <= 0
                else 0
            )

            rows.append({
                "date": date,
                "product_id": product_id,
                "product_name": product_name,
                "category": category,
                "warehouse_id": warehouse_id,
                "warehouse": warehouse_name,
                "supplier_id": supplier_id,
                "supplier": suppliers[supplier_id],
                "lead_time_days": lead_time,
                "price": round(price, 2),
                "discount": round(discount, 2),
                "promotion": promotion,
                "units_sold": units_sold,
                "inventory_level": inventory,
                "revenue": round(revenue, 2),
                "stockout": stockout,
            })


df = pd.DataFrame(rows)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    [
        "date",
        "product_id",
        "warehouse_id"
    ]
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Dataset generated successfully.")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Saved to: {OUTPUT_PATH}")