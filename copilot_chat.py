
from pathlib import Path
import re
import duckdb
import pandas as pd
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "supply_chain_copilot.duckdb"


def read_table(table_name):
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return con.execute(f"SELECT * FROM {table_name}").df()


def find_sku(question):
    match = re.search(r"(FOODS|HOBBIES|HOUSEHOLD)_\d_\d{3}", question.upper())
    return match.group(0) if match else None


def get_evidence(question):
    q = question.lower()
    sku = find_sku(question)

    # Inventory questions
    if any(word in q for word in ["inventory", "stock", "reorder", "safety stock", "replenish"]):
        df = read_table("inventory_recommendations")

        if sku:
            return df[df["item_id"].str.upper() == sku].copy(), "Inventory recommendation"

        if "food" in q:
            df = df[df["cat_id"].str.upper() == "FOODS"]
        elif "household" in q:
            df = df[df["cat_id"].str.upper() == "HOUSEHOLD"]
        elif "hobbies" in q or "hobby" in q:
            df = df[df["cat_id"].str.upper() == "HOBBIES"]

        sort_column = "recommended_28_day_stock_units"
        return df.sort_values(sort_column, ascending=False).head(10), "Top inventory recommendations"

    # SKU-specific forecast questions
    if sku or any(word in q for word in ["forecast", "demand", "predict", "prediction"]):
        df = read_table("daily_forecast_predictions")

        if sku:
            df = df[df["item_id"].str.upper() == sku].copy()
        else:
            df = (
                df.groupby("item_id", as_index=False)
                .agg(
                    actual_units_sold=("actual_units_sold", "sum"),
                    lightgbm_recursive_prediction=("lightgbm_recursive_prediction", "sum")
                )
                .sort_values("lightgbm_recursive_prediction", ascending=False)
                .head(10)
            )

        return df, "Recursive LightGBM forecast evidence"

    # Shipping-mode questions
    if any(word in q for word in ["shipping mode", "first class", "second class", "same day", "standard class"]):
        df = read_table("shipping_mode_performance")
        return df.sort_values("late_delivery_rate", ascending=False), "Shipping-mode performance"

    # Regional delivery-risk questions
    if any(word in q for word in ["region", "regional", "central america", "western europe", "delivery risk"]):
        df = read_table("regional_delivery_risk")

        regions = df["order_region"].astype(str)
        matches = df[regions.str.lower().apply(lambda x: x in q)]

        if not matches.empty:
            return matches, "Regional delivery-risk evidence"

        return (
            df.sort_values("priority_score", ascending=False).head(10),
            "Top regional delivery-risk priorities"
        )

    # Model comparison questions
    if any(word in q for word in ["model", "wape", "accuracy", "mae", "rmse", "bias", "baseline", "lightgbm"]):
        df = read_table("forecast_model_comparison")
        return df, "Forecast model comparison"

    # Executive KPI default
    return read_table("executive_kpis"), "Executive KPI evidence"


def answer_question(question, api_key):
    evidence, evidence_title = get_evidence(question)

    # Safety: only the selected, approved evidence is sent to the model.
    evidence_text = evidence.head(20).to_csv(index=False)

    if not api_key:
        return (
            "The AI key is missing. Add GEMINI_API_KEY to .streamlit/secrets.toml, "
            "save it, and restart Streamlit."
        ), evidence, evidence_title

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are the AI Supply Chain Copilot for a portfolio project.

Answer the user's question using ONLY the supplied evidence.
Do not invent data, assume live data, or claim causal proof beyond the evidence.
The source is historical project data and forecast outputs. Keep the answer concise,
business-focused, and include key numbers when available.

User question:
{question}

Evidence title:
{evidence_title}

Evidence:
{evidence_text}
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        return response.text, evidence, evidence_title

    except Exception as error:
        return (
            f"The Copilot could not generate an AI response. "
            f"Check the API key and internet connection. Technical detail: {error}"
        ), evidence, evidence_title
