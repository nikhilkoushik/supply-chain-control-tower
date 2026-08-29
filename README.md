# AI Supply Chain Control Tower

An AI-enabled supply-chain decision platform that combines delivery-risk diagnostics, 28-day SKU demand forecasting, inventory replenishment recommendations, interactive dashboards, and a governed AI Copilot.

**Live App:** [AI Supply Chain Copilot](https://supply-chain-control-tower-tjjyr2qwseyzxsyzy2quri.streamlit.app)  
**Tableau Dashboard:** [Supply Chain Control Tower](https://public.tableau.com/app/profile/nikhil.koushik.bettadapura.subramanya/viz/supply_chain_control_tower/Dashboard1)

---

## Business Problem

Supply-chain teams need to identify delivery risk, improve demand-forecast accuracy, and make inventory decisions before stockouts or excess inventory affect operations.

This project transforms historical supply-chain and retail demand data into a centralized decision-support application for:

- Delivery-risk monitoring by shipping mode and region
- SKU-level 28-day demand forecasting
- Forecast-model evaluation and selection
- Bias-adjusted inventory replenishment recommendations
- Natural-language business questions through an AI Copilot

---

## Key Results

| Metric | Result |
|---|---:|
| Total Sales Analyzed | $33.05M |
| Total Orders Analyzed | 65,752 |
| Total Profit Analyzed | $3.97M |
| Overall Late-Delivery Rate | 54.83% |
| Best Forecast Model | Recursive LightGBM |
| Best Forecast WAPE | 25.21% |
| Forecast MAE | 4.43 units |
| Forecast RMSE | 6.56 units |
| WAPE Improvement vs. Seasonal Baseline | 26.49% |
| WAPE-Derived Forecast Accuracy | 74.79% |

---

## Delivery-Risk Insights

The analysis identified material delivery-risk variation across shipping modes:

| Shipping Mode | Late-Delivery Rate |
|---|---:|
| First Class | 95.32% |
| Second Class | 76.63% |
| Same Day | 45.74% |
| Standard Class | 38.07% |

Central America and Western Europe had the highest volumes of late shipments, making them key areas for operational review.

---

## Forecasting Approach

Three forecasting approaches were evaluated using a time-based holdout period:

| Model | WAPE | MAE | RMSE | Forecast Bias |
|---|---:|---:|---:|---:|
| Seasonal Naïve Baseline | 34.29% | 6.03 | 8.79 | -2.55% |
| Initial ML Model | 26.68% | 4.69 | 6.85 | 2.16% |
| Recursive LightGBM | **25.21%** | **4.43** | **6.56** | -6.22% |

Recursive LightGBM was selected as the final model because it produced the lowest WAPE, MAE, and RMSE.

### Forecast Features

- Historical demand lag features, including 28-day lag
- Selling price
- Day of week, month, and week of year
- SNAP eligibility indicators
- Event indicators
- Weekend flags
- Item and category attributes

---

## Inventory Recommendations

The inventory module converts forecast output into actionable recommendations using:

- Bias-adjusted 28-day demand forecasts
- Forecast RMSE
- 95% service-level safety stock assumption
- Seven-day lead-time assumption
- Reorder points
- Recommended 28-day inventory quantities

The application identifies high-priority SKUs based on recommended inventory levels and revenue contribution.

> **Note:** This project provides inventory decision support from historical and forecast data. It does not yet connect to live on-hand inventory, open purchase orders, or real-time supplier/shipment feeds.

---

## AI Supply Chain Copilot

The Streamlit application includes an AI Copilot powered by Gemini and grounded in approved DuckDB project tables.

The Copilot can answer questions such as
