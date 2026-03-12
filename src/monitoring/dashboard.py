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

st.set_page_config(layout="wide", page_title="DeepPrep ML Platform Insight", page_icon="🍔")

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
<style>
    /* Global modern font and background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Subtle glowing header */
    h1 {
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #FF8E53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 4px 20px rgba(255, 107, 107, 0.4);
        margin-bottom: 0.2em;
    }
    
    /* Glassmorphism Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(255, 107, 107, 0.3);
    }

    /* Target the ETA success box */
    .eta-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 12px;
        padding: 25px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(56, 239, 125, 0.4);
        margin: 20px 0;
    }
    
    .eta-box h2 {
        color: white;
        margin: 0;
        font-weight: 800;
        font-size: 3rem;
    }
    
    /* Copilot Chat Box */
    .copilot-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 5px solid #6366f1;
        padding: 20px;
        border-radius: 8px;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #e2e8f0;
        margin-top: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Surge Warning Box */
    .surge-box {
        background: linear-gradient(135deg, #b91d73 0%, #f953c6 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        font-weight: bold;
        text-align: center;
        animation: pulse 2s infinite;
        box-shadow: 0 0 20px rgba(249, 83, 198, 0.6);
        margin-bottom: 20px;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Custom Sidebar Button */
    .stButton > button {
        width: 100%;
        border-radius: 30px;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4);
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        transform: scale(1.05);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("DeepPrep Command Center")
st.markdown("<p style='color: #888; font-size: 1.2rem;'>Next-Generation AI Logistics & ETA Forecasting Engine</p>", unsafe_allow_html=True)
st.write("---")

# Layout: Split into Sidebar and Main Content
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2921/2921822.png", width=100)
st.sidebar.title("Ops Console")
st.sidebar.markdown("Inject live streams directly into the Kafka-PyTorch network.")

if st.sidebar.button("🚀 INJECT LIVE ORDER STREAM"):
    st.sidebar.success("Stream active! Waiting for PyTorch inference...")
    
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
            st.markdown(f"**Ticket ID:** `<span style='color: #4facfe;'>{order_id}</span>` &nbsp;&nbsp;|&nbsp;&nbsp; **Hub:** `<span style='color: #FF8E53;'>{restaurant_id}</span>`", unsafe_allow_html=True)
            st.markdown(f"**Manifest:** {', '.join(items)}")
            
            st.markdown(f"""
            <div class="eta-box">
                <p style="margin:0; font-size: 1.2rem; opacity: 0.9;">Target Delivery Prediction (P95)</p>
                <h2>{p95} <span style="font-size: 1.5rem;">MINUTES</span></h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Show Surge Status if active
            if explain_data.get('is_surging'):
                st.markdown("""
                <div class="surge-box">
                    ⚠️ CRITICAL SURGE: AUTOMATED THROTTLING & PRICING ACTIVATED
                </div>
                """, unsafe_allow_html=True)

        with col_side:
            # --- AI OPERATIONS COPILOT ---
            st.markdown("### 🤖 Neural Copilot")
            try:
                copilot_res = requests.post(f"{API_URL}/copilot", json=explain_data, timeout=5)
                copilot_res.raise_for_status()
                narrative = copilot_res.json()["narrative"]
                st.markdown(f"""
                <div class="copilot-box">
                    <strong>AI Operations Insight:</strong><br><br>
                    {narrative}
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Copilot offline: {e}")
        
        st.write("---")
        
        # --- DEEP EXPLAINABILITY ---
        st.markdown("### 🔬 Interpretability Matrix (SHAP Values)")
        
        factors = explain_data['key_factors']
        st.caption("Primary mathematically extracted vectors driving the PyTorch prediction:")
        
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
tab1, tab2 = st.tabs(["🚀 Global System Telemetry", "🛡️ Model Health (Evidently AI)"])

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
st.sidebar.caption("DeepPrep Platform OS v2.3.1 (Aesthetica Build)")
