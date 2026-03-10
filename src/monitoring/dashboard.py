import streamlit as st
import requests
import json
import random
from datetime import datetime
import pandas as pd
import pathlib

# Configuration
API_URL_PREDICT = "http://127.0.0.1:8000/predict"
API_URL_EXPLAIN = "http://127.0.0.1:8000/explain"
API_URL_METRICS = "http://127.0.0.1:8000/metrics"

st.set_page_config(layout="wide", page_title="DeepPrep ML Platform Insight")

st.title("🍔 DeepPrep FPT Observability & AI Explainer")
st.markdown("Real-time observability into the PyTorch predictions, M/M/c Engine, Component Latency, System Health, and Data Drift.")

st.sidebar.header("Operations")
if st.sidebar.button("Simulate Incoming Order Stream"):
    
    with st.spinner("Fetching Features & Running Inference..."):
        # Generate Fake Event
        restaurant_id = f"R{str(random.randint(1, 20)).zfill(3)}"
        order_id = f"ORD{random.randint(1000, 9999)}"
        items_pool = ["Pizza", "Burger", "Coke", "Pasta", "Salad", "Ice Cream"]
        items = random.sample(items_pool, random.randint(1, 4))
        
        payload = {
            "order_id": order_id,
            "restaurant_id": restaurant_id,
            "items": items,
            "created_at": datetime.utcnow().isoformat()
        }
        
        try:
            # 1. Fetch Explain (which also gives us the prediction)
            res = requests.post(API_URL_EXPLAIN, json=payload, timeout=5)
            res.raise_for_status()
            explain_data = res.json()
            
            # Display Order Details
            st.subheader(f"Order: `{order_id}` @ `{restaurant_id}`")
            st.write(f"**Items:** {', '.join(items)}")
            
            # Display Prediction
            p95 = explain_data['prediction_p95']
            st.success(f"**Target ETA (P95):** {p95} minutes")
            
            # 2. Display Feature Importances (SHAP)
            st.subheader("🤖 SHAP Feature Importances")
            st.markdown("What factors drove this prediction up or down?")
            
            factors = explain_data['key_factors']
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Top Driver 1", factors[0] if len(factors) > 0 else "N/A")
            with col2: st.metric("Top Driver 2", factors[1] if len(factors) > 1 else "N/A")
            with col3: st.metric("Top Driver 3", factors[2] if len(factors) > 2 else "N/A")
            
            st.write("---")
            st.markdown("### Waterfall Importance")
            shap_dict = explain_data['feature_importance_shap']
            df_shap = pd.DataFrame(list(shap_dict.items()), columns=['Feature', 'SHAP Value'])
            st.bar_chart(df_shap.set_index('Feature'))
            
        except requests.exceptions.RequestException as e:
            st.error(f"API Error. Is FastAPI running? Details: {e}")

st.write("---")
st.header("📈 Production Promethus Metrics")
if st.button("Refresh Telemetry"):
    try:
        metrics_req = requests.get(API_URL_METRICS, timeout=2)
        metrics_req.raise_for_status()
        st.code(metrics_req.text, language="text")
    except Exception as e:
        st.warning(f"Could not load prometheus metrics: {e}")

st.write("---")
st.header("🛡️ Evidently Data Drift Report")
report_path = pathlib.Path("src/monitoring/reports/model_health_report.html")

if report_path.exists():
    with st.expander("Reveal Health Report (Daily Data vs Reference)"):
        with open(report_path, "r", encoding="utf-8") as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=800, scrolling=True)
else:
    st.info("No drift report found. Run `python src/monitoring/drift_detector.py` to generate one.")

st.sidebar.markdown("---")
st.sidebar.markdown("*DeepPrep Engine v2.0*")
