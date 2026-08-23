import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import pandas as pd
import streamlit as st
import plotly.express as px

from src.inventory import (
    calculate_inventory_metrics
)

from src.insights import (
    generate_insights
)


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="SupplyChainIQ",
    page_icon="📦",
    layout="wide"
)


# --------------------------------
# Load data
# --------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/processed/supply_chain_clean.csv"
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


@st.cache_data
def load_inventory():

    df = load_data()

    return calculate_inventory_metrics(
        df
    )


df = load_data()

inventory = load_inventory()


# --------------------------------
# Title
# --------------------------------

st.title(
    "📦 SupplyChainIQ"
)

st.caption(
    "ML-Based Supply Chain Demand "
    "Forecasting & Inventory Intelligence"
)


# --------------------------------
# Sidebar
# --------------------------------

st.sidebar.header(
    "Filters"
)

categories = [
    "All"
] + sorted(
    df["category"]
    .unique()
    .tolist()
)

selected_category = (
    st.sidebar.selectbox(
        "Category",
        categories
    )
)

warehouses = [
    "All"
] + sorted(
    df["warehouse"]
    .unique()
    .tolist()
)

selected_warehouse = (
    st.sidebar.selectbox(
        "Warehouse",
        warehouses
    )
)


filtered = df.copy()

if selected_category != "All":

    filtered = filtered[
        filtered["category"]
        == selected_category
    ]

if selected_warehouse != "All":

    filtered = filtered[
        filtered["warehouse"]
        == selected_warehouse
    ]


# --------------------------------
# KPI metrics
# --------------------------------

total_revenue = (
    filtered["revenue"].sum()
)

total_demand = (
    filtered["units_sold"].sum()
)

total_inventory = (
    filtered
    .groupby(
        [
            "product_id",
            "warehouse_id"
        ]
    )
    .tail(1)
    ["inventory_level"]
    .sum()
)

stockout_count = (
    inventory[
        inventory["stockout_risk"]
        .isin(
            ["HIGH", "CRITICAL"]
        )
    ]
    .shape[0]
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Revenue",
    f"₹{total_revenue:,.0f}"
)

col2.metric(
    "Total Demand",
    f"{total_demand:,.0f}"
)

col3.metric(
    "Current Inventory",
    f"{total_inventory:,.0f}"
)

col4.metric(
    "High/Critical Risk",
    stockout_count
)


# --------------------------------
# Revenue trend
# --------------------------------

st.subheader(
    "Revenue Trend"
)

monthly_revenue = (
    filtered
    .set_index("date")
    .resample("ME")["revenue"]
    .sum()
    .reset_index()
)

fig = px.line(
    monthly_revenue,
    x="date",
    y="revenue",
    title="Monthly Revenue"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------
# Demand trend
# --------------------------------

st.subheader(
    "Demand Trend"
)

monthly_demand = (
    filtered
    .set_index("date")
    .resample("ME")["units_sold"]
    .sum()
    .reset_index()
)

fig = px.line(
    monthly_demand,
    x="date",
    y="units_sold",
    title="Monthly Demand"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------
# Category analysis
# --------------------------------

col1, col2 = st.columns(2)

with col1:

    category_revenue = (
        filtered
        .groupby("category")["revenue"]
        .sum()
        .reset_index()
        .sort_values(
            "revenue",
            ascending=False
        )
    )

    fig = px.bar(
        category_revenue,
        x="category",
        y="revenue",
        title="Revenue by Category"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    warehouse_demand = (
        filtered
        .groupby("warehouse")["units_sold"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        warehouse_demand,
        x="warehouse",
        y="units_sold",
        title="Demand by Warehouse"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# --------------------------------
# Inventory risk
# --------------------------------

st.subheader(
    "Inventory Risk"
)

risk_counts = (
    inventory["stockout_risk"]
    .value_counts()
    .reset_index()
)

risk_counts.columns = [
    "risk",
    "count"
]

fig = px.bar(
    risk_counts,
    x="risk",
    y="count",
    title="Stock-out Risk Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------
# AI Insights
# --------------------------------

st.subheader(
    "🤖 Business Insights"
)

insights = generate_insights(
    inventory,
    df
)

for insight in insights:

    if insight["type"] == "CRITICAL":

        st.error(
            f"🔴 {insight['title']}: "
            f"{insight['message']}"
        )

    elif insight["type"] == "WARNING":

        st.warning(
            f"🟡 {insight['title']}: "
            f"{insight['message']}"
        )

    else:

        st.success(
            f"🟢 {insight['title']}: "
            f"{insight['message']}"
        )