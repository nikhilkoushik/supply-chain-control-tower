from copilot_chat import answer_question
from pathlib import Path
import re

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="AI Supply Chain Copilot",
    page_icon="📦",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "supply_chain_copilot.duckdb"


# -----------------------------
# Database helpers
# -----------------------------
def clean_column_name(column_name):
    column_name = str(column_name).strip().lower()
    column_name = re.sub(r"[^a-z0-9]+", "_", column_name)
    return column_name.strip("_")


@st.cache_data
def load_table(table_name):
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        dataframe = con.execute(f"SELECT * FROM {table_name}").df()

    dataframe.columns = [clean_column_name(column) for column in dataframe.columns]
    return dataframe


def first_matching_column(dataframe, candidates):
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


def format_currency(value):
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


# -----------------------------
# Load approved analytics tables
# -----------------------------
shipping = load_table("shipping_mode_performance")
regional = load_table("regional_delivery_risk")
models = load_table("forecast_model_comparison")
inventory = load_table("inventory_recommendations")
daily_forecast = load_table("daily_forecast_predictions")
kpis = load_table("executive_kpis")

shipping_mode_col = first_matching_column(shipping, ["shipping_mode"])
shipping_late_rate_col = first_matching_column(shipping, ["late_delivery_rate"])

region_col = first_matching_column(regional, ["order_region", "region"])
late_shipments_col = first_matching_column(
    regional,
    ["late_shipment_records", "late_delivery_records"]
)
region_late_rate_col = first_matching_column(regional, ["late_delivery_rate"])

model_name_col = first_matching_column(models, ["model"])
wape_col = first_matching_column(models, ["wape_percent", "wape"])

item_col = first_matching_column(inventory, ["item_id"])
inventory_category_col = first_matching_column(inventory, ["cat_id", "category"])
recommended_stock_col = first_matching_column(
    inventory,
    ["recommended_28_day_stock_units", "recommended_28_day_stock"]
)
reorder_point_col = first_matching_column(inventory, ["reorder_point_units", "reorder_point"])
safety_stock_col = first_matching_column(inventory, ["safety_stock_units", "safety_stock"])
revenue_col = first_matching_column(inventory, ["total_revenue"])

daily_item_col = first_matching_column(daily_forecast, ["item_id"])
date_col = first_matching_column(daily_forecast, ["date"])
actual_col = first_matching_column(daily_forecast, ["actual_units_sold", "actual_units"])
prediction_col = first_matching_column(
    daily_forecast,
    ["lightgbm_recursive_prediction", "recursive_lightgbm_prediction"]
)

# -----------------------------
# Header
# -----------------------------
st.title("📦 AI Supply Chain Copilot")
st.caption(
    "Delivery-risk diagnostics, 28-day demand forecasting, and inventory replenishment recommendations."
)

# -----------------------------
# Executive metrics
# -----------------------------
best_model = models.loc[models[wape_col].idxmin()]
best_wape = float(best_model[wape_col])
forecast_accuracy = 100 - best_wape

sales_value = 33_054_402.38
late_delivery_rate = 54.83

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric("Total Sales", format_currency(sales_value))
metric_2.metric("Late-Delivery Rate", f"{late_delivery_rate:.2f}%")
metric_3.metric("Best Forecast WAPE", f"{best_wape:.2f}%")
metric_4.metric("WAPE-Derived Accuracy", f"{forecast_accuracy:.2f}%")

st.divider()

# -----------------------------
# Dashboard tabs
# -----------------------------
delivery_tab, forecast_tab, inventory_tab = st.tabs(
    ["Delivery Risk", "Demand Forecast", "Inventory Recommendations"]
)

with delivery_tab:
    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Late Delivery Rate by Shipping Mode")

        shipping_chart = shipping.sort_values(
            shipping_late_rate_col,
            ascending=False
        )

        figure = px.bar(
            shipping_chart,
            x=shipping_late_rate_col,
            y=shipping_mode_col,
            orientation="h",
            color=shipping_late_rate_col,
            color_continuous_scale="Reds",
            text_auto=".2f",
            labels={
                shipping_late_rate_col: "Late Delivery Rate (%)",
                shipping_mode_col: "Shipping Mode",
            },
        )

        figure.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            height=400,
        )

        st.plotly_chart(figure, use_container_width=True)

    with right_column:
        st.subheader("Top Regional Delivery-Risk Priorities")

        regional_chart = regional.sort_values(
            late_shipments_col,
            ascending=False
        ).head(10)

        figure = px.bar(
            regional_chart,
            x=late_shipments_col,
            y=region_col,
            orientation="h",
            color=region_late_rate_col,
            color_continuous_scale="Oranges",
            text_auto=",",
            labels={
                late_shipments_col: "Late Shipment Records",
                region_col: "Order Region",
                region_late_rate_col: "Late Delivery Rate (%)",
            },
        )

        figure.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=400,
        )

        st.plotly_chart(figure, use_container_width=True)

with forecast_tab:
    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.subheader("Model Comparison")

        model_chart = models.sort_values(wape_col, ascending=True)

        model_colors = {
            "Recursive LightGBM": "#2ca02c",
            "Initial ML Model": "#9e9e9e",
            "Seasonal Naïve Baseline": "#ef476f",
        }

        figure = px.bar(
            model_chart,
            x=wape_col,
            y=model_name_col,
            orientation="h",
            color=model_name_col,
            color_discrete_map=model_colors,
            text_auto=".2f",
            labels={
                wape_col: "WAPE (%)",
                model_name_col: "Model",
            },
        )

        figure.update_layout(
            showlegend=False,
            yaxis={"categoryorder": "total ascending"},
            height=400,
        )

        st.plotly_chart(figure, use_container_width=True)

    with right_column:
        st.subheader("Actual vs Recursive LightGBM Forecast")

        selected_sku = st.selectbox(
            "Select SKU",
            sorted(daily_forecast[daily_item_col].unique())
        )

        sku_data = daily_forecast[
            daily_forecast[daily_item_col] == selected_sku
        ].copy()

        sku_data[date_col] = pd.to_datetime(sku_data[date_col])
        sku_data = sku_data.sort_values(date_col)

        chart_data = sku_data.melt(
            id_vars=[date_col],
            value_vars=[actual_col, prediction_col],
            var_name="Series",
            value_name="Units Sold",
        )

        chart_data["Series"] = chart_data["Series"].replace({
            actual_col: "Actual Units Sold",
            prediction_col: "Recursive LightGBM Forecast",
        })

        figure = px.line(
            chart_data,
            x=date_col,
            y="Units Sold",
            color="Series",
            color_discrete_map={
                "Actual Units Sold": "#1f77b4",
                "Recursive LightGBM Forecast": "#ef476f",
            },
            markers=True,
        )

        figure.update_layout(height=400)
        st.plotly_chart(figure, use_container_width=True)

with inventory_tab:
    st.subheader("Bias-Adjusted Inventory Replenishment Recommendations")

    categories = ["All"] + sorted(inventory[inventory_category_col].dropna().unique())

    selected_category = st.selectbox(
        "Filter by Category",
        categories
    )

    inventory_view = inventory.copy()

    if selected_category != "All":
        inventory_view = inventory_view[
            inventory_view[inventory_category_col] == selected_category
        ]

    inventory_view = inventory_view.sort_values(
        recommended_stock_col,
        ascending=False
    )

    display_columns = [
        item_col,
        inventory_category_col,
        revenue_col,
        recommended_stock_col,
        reorder_point_col,
        safety_stock_col,
    ]

    inventory_view = inventory_view[display_columns].rename(columns={
        item_col: "SKU",
        inventory_category_col: "Category",
        revenue_col: "Total Revenue",
        recommended_stock_col: "Recommended 28-Day Stock",
        reorder_point_col: "Reorder Point",
        safety_stock_col: "Safety Stock",
    })

    st.dataframe(
        inventory_view,
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption(
    "Data source: validated project outputs. Forecast: Recursive LightGBM, "
    "28-day horizon. Inventory recommendations use bias adjustment and safety stock."
)
# ---------------- AI SUPPLY CHAIN COPILOT ----------------

st.divider()
st.subheader("Ask the Supply Chain Copilot")
st.caption(
    "Answers are grounded in the approved DuckDB project tables and forecast outputs. "
    "This historical project updates when the underlying data pipeline is refreshed."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("copilot_question_form", clear_on_submit=True):
    question = st.text_input(
        "Ask a question",
        placeholder="Example: Which shipping mode has the highest late-delivery rate?"
    )
    ask_button = st.form_submit_button("Ask Copilot")

if ask_button:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        api_key = st.secrets.get("GEMINI_API_KEY", "")

        with st.spinner("Checking the approved supply-chain data..."):
            answer, evidence, evidence_title = answer_question(question, api_key)

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer,
                "evidence": evidence,
                "evidence_title": evidence_title,
            }
        )

for interaction in reversed(st.session_state.chat_history[-3:]):
    st.markdown(f"**You:** {interaction['question']}")
    st.markdown(f"**Copilot:** {interaction['answer']}")

    with st.expander(f"View evidence: {interaction['evidence_title']}"):
        st.dataframe(interaction["evidence"], use_container_width=True)

    st.divider()
