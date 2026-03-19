import streamlit as st
import requests
import json
import random
from datetime import datetime
import pandas as pd
import pathlib
import os

# Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_URL_PREDICT = f"{API_URL}/predict"
API_URL_EXPLAIN = f"{API_URL}/explain"
API_URL_METRICS = f"{API_URL}/metrics"

st.set_page_config(layout="wide", page_title="DeepPrep Telemetry", page_icon="📈")

# --- CUSTOM CSS FOR ENTERPRISE UI ---
st.markdown("""
<style>
    /* Global modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Clean, minimal header */
    h1 {
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin-bottom: 0.1em;
    }
    
    /* Sleek Metric Cards */
    div[data-testid="metric-container"] {
        background: #0E1117;
        border: 1px solid #1E2329;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        border-color: #3B82F6;
        transform: translateY(-2px);
    }

    /* ETA Success Box */
    .eta-box {
        background: #0E1117;
        border-left: 4px solid #10B981;
        border-radius: 6px;
        padding: 1.5rem;
        color: #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    
    .eta-box h2 {
        color: #10B981;
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
        letter-spacing: -0.02em;
    }
    
    /* Copilot / Insights Box */
    .copilot-box {
        background: #0E1117;
        border: 1px solid #1E2329;
        border-top: 3px solid #6366F1;
        padding: 1.2rem;
        border-radius: 6px;
        font-size: 0.95rem;
        line-height: 1.5;
        color: #94A3B8;
        margin-top: 1rem;
    }
    
    .copilot-box strong {
        color: #F8FAFC;
    }
    
    /* Surge Warning Box */
    .surge-box {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-left: 4px solid #EF4444;
        border-radius: 6px;
        padding: 1rem;
        color: #EF4444;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    /* Custom Sidebar Button */
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        background: #FFFFFF;
        color: #0F172A;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border: 1px solid #E2E8F0;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #F8FAFC;
        border-color: #CBD5E1;
        color: #0F172A;
    }
    
    /* Subtitle text */
    .subtitle-text {
        color: #64748B;
        font-size: 1.05rem;
        font-weight: 400;
        letter-spacing: -0.01em;
    }
</style>
""", unsafe_allow_html=True)

st.title("Network Telemetry")
st.markdown("<p class='subtitle-text'>DeepPrep Production Forecasting & Logistics Engine</p>", unsafe_allow_html=True)
st.write("---")

# Layout: Split into Sidebar and Main Content
st.sidebar.title("Operations Console")
st.sidebar.markdown("Execute synthetic pipeline injection for performance validation.")

if st.sidebar.button("Execute Pipeline Injection"):
    st.sidebar.success("Payload queued. Awaiting inference...")
    
    # Generate Fake Event
    restaurant_id = f"R{str(random.randint(1, 20)).zfill(3)}"
    order_id = f"ORD{random.randint(1000, 9999)}"
    items_pool = ["Artisan Pizza", "Truffle Burger", "Diet Coke", "Tuscan Pasta", "Caesar Salad", "Gelato"]
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
        
        # --- TOP LEVEL DASHBOARD ---
        col_main, col_side = st.columns([2, 1])
        
        p95 = explain_data['prediction_p95']
        
        with col_main:
            st.markdown(f"### Live Order Details")
            st.markdown(f"**Ticket ID:** `<span style='color: #60A5FA;'>{order_id}</span>` &nbsp;&nbsp;|&nbsp;&nbsp; **Hub:** `<span style='color: #F87171;'>{restaurant_id}</span>`", unsafe_allow_html=True)
            st.markdown(f"**Manifest:** {', '.join(items)}")
            
            st.markdown(f"""
            <div class="eta-box">
                <p style="margin:0; font-size: 0.95rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Target Delivery Forecast (P95)</p>
                <h2>{p95} <span style="font-size: 1.5rem; color: #10B981; font-weight: 500;">MINUTES</span></h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Show Surge Status if active
            if explain_data.get('is_surging'):
                st.markdown("""
                <div class="surge-box">
                    SYSTEM ALERT: Load throttling protocols and dynamic pricing active.
                </div>
                """, unsafe_allow_html=True)

        with col_side:
            # --- AI OPERATIONS COPILOT ---
            st.markdown("### Automated Insights")
            try:
                copilot_res = requests.post(f"{API_URL}/copilot", json=explain_data, timeout=5)
                copilot_res.raise_for_status()
                narrative = copilot_res.json()["narrative"]
                st.markdown(f"""
                <div class="copilot-box">
                    <strong>Event Analysis:</strong><br><br>
                    {narrative}
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Analysis service unavailable: {e}")
        
        st.write("---")
        
        # --- DEEP EXPLAINABILITY ---
        st.markdown("### Interpretability Matrix (SHAP)")
        
        factors = explain_data['key_factors']
        st.caption("Primary vectors driving the current forecast:")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1: st.metric("Primary Driver", factors[0] if len(factors) > 0 else "N/A")
        with m_col2: st.metric("Secondary Driver", factors[1] if len(factors) > 1 else "N/A")
        with m_col3: st.metric("Tertiary Driver", factors[2] if len(factors) > 2 else "N/A")
        
        shap_dict = explain_data['feature_importance_shap']
        df_shap = pd.DataFrame(list(shap_dict.items()), columns=['Tensor', 'Impact Score'])
        df_shap = df_shap.sort_values(by='Impact Score', ascending=False)
        st.bar_chart(df_shap.set_index('Tensor'), use_container_width=True)
        
    except requests.exceptions.RequestException as e:
        st.error(f"API Error. System Offline. Details: {e}")

# --- TABS FOR METRICS ---
st.write("---")
tab1, tab2 = st.tabs(["Telemetry Metrics", "Model Health Drift"])

with tab1:
    st.markdown("### Native Prometheus Exporter")
    if st.button("Poll Live Infrastructure Metrics"):
        try:
            metrics_req = requests.get(API_URL_METRICS, timeout=2)
            metrics_req.raise_for_status()
            st.code(metrics_req.text, language="text")
        except Exception as e:
            st.warning(f"Collector offline: {e}")

with tab2:
    st.markdown("### Statistical Data Drift & Validation")
    report_path = pathlib.Path("src/monitoring/reports/model_health_report.html")
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=800, scrolling=True)
    else:
        st.info("Scanner standby. Run `python src/monitoring/drift_detector.py` to compile latest HTML artifact.")

st.sidebar.markdown("---")
st.sidebar.caption("DeepPrep Engineering | Build 2.3.1")
