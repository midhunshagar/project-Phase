import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
from app.services.pipeline import run_scenario, train_if_needed

st.set_page_config(page_title="ECONOSIM-AI", page_icon="📈", layout="wide")

st.title("ECONOSIM-AI")
st.caption("AI-driven counterfactual economic policy simulation and impact analysis")

with st.sidebar:
    st.header("Policy Scenario")
    policy = st.selectbox(
        "Policy",
        ["income_tax", "interest_rate", "government_spending", "subsidy"],
        format_func=lambda x: x.replace("_", " ").title()
    )

    defaults = {
        "income_tax": (20.0, 25.0),
        "interest_rate": (6.0, 8.0),
        "government_spending": (4350.0, 4700.0),
        "subsidy": (0.0, 5.0),
    }
    baseline, counterfactual = defaults[policy]

    baseline = st.number_input("Baseline value", value=baseline)
    counterfactual = st.number_input("Counterfactual value", value=counterfactual)
    duration = st.slider("Simulation horizon (years)", 1, 20, 5)

    run = st.button("Run Simulation", type="primary", use_container_width=True)

st.info("The included dataset is synthetic and exists only to make the application runnable before external data is supplied.")

if run:
    with st.spinner("Running simulation and AI analysis..."):
        result = run_scenario(policy, baseline, counterfactual, duration)

    st.subheader("Scenario Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Policy", policy.replace("_", " ").title())
    c2.metric("Baseline", f"{baseline:,.2f}")
    c3.metric("Counterfactual", f"{counterfactual:,.2f}")

    st.subheader("Baseline vs Counterfactual")
    comparison = result["comparison"].copy()
    comparison["indicator_label"] = comparison["indicator"].str.replace("_", " ").str.title()

    st.dataframe(
        comparison[["indicator_label", "baseline", "counterfactual", "absolute_impact", "percentage_impact"]]
        .rename(columns={
            "indicator_label": "Indicator",
            "baseline": "Baseline",
            "counterfactual": "Counterfactual",
            "absolute_impact": "Impact",
            "percentage_impact": "Impact (%)"
        }),
        use_container_width=True,
        hide_index=True
    )

    chart_df = comparison[["indicator_label", "baseline", "counterfactual"]].melt(
        id_vars="indicator_label", var_name="Scenario", value_name="Value"
    )
    fig = px.bar(chart_df, x="indicator_label", y="Value", color="Scenario", barmode="group",
                 title="Economic Indicator Comparison")
    fig.update_layout(xaxis_title="", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Economic Trajectory")
    baseline_hist = pd.DataFrame(result["baseline_history"])
    cf_hist = pd.DataFrame(result["counterfactual_history"])
    trajectory = pd.DataFrame({
        "Year": baseline_hist["year"],
        "Baseline GDP": baseline_hist["gdp"],
        "Counterfactual GDP": cf_hist["gdp"],
    }).set_index("Year")
    st.line_chart(trajectory)

    st.subheader("AI Prediction")
    p1, p2 = st.columns(2)
    p1.metric("Predicted baseline GDP", f"{result['predicted_baseline_gdp']:,.2f}")
    p2.metric("Predicted counterfactual GDP", f"{result['predicted_counterfactual_gdp']:,.2f}")

    st.subheader("Explanation")
    for item in result["explanation"]:
        st.write("• " + item)

    st.subheader("Model Feature Importance")
    st.dataframe(result["feature_importance"], use_container_width=True, hide_index=True)

    st.subheader("Model Evaluation")
    metrics = pd.DataFrame(result["metrics"]).T
    st.dataframe(metrics, use_container_width=True)

else:
    st.subheader("How it works")
    st.markdown("""
    1. Select an economic policy.
    2. Define the baseline and counterfactual values.
    3. Run the multi-year economic simulation.
    4. Compare baseline and policy outcomes.
    5. Review the ML prediction, feature importance, and explanation.
    """)
