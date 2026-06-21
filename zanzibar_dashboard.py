import streamlit as st
import pandas as pd
import plotly.express as px
import os
import warnings

# --- CONFIGURATION ---
st.set_page_config(page_title="Zanzibar Health Early Warning System", layout="wide")
warnings.filterwarnings("ignore")

# --- DATA LOADING (Robust Pathing & Caching) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    # Attempting to load from local structure first
    data_path = os.path.join(BASE_DIR, "data", "clean", "zanzibar_full_risk_dataset.csv")
    if not os.path.exists(data_path):
        data_path = "zanzibar_full_risk_dataset.csv" # Fallback
    
    df = pd.read_csv(data_path, parse_dates=["date"])
    return df

df = load_data()

# --- DASHBOARD LAYOUT (Tabs Integration) ---
st.title("🌊 Zanzibar Climate and Disease Analytics")
tab1, tab2, tab3 = st.tabs(["Live Alerts", "Historical Trends", "Model Performance"])

with tab1:
    st.header("Public Health Alert Dashboard")
    # Metric cards for quick insight
    col1, col2 = st.columns(2)
    col1.metric("Predicted Cases (Next Window)", "10.3", "+2.1")
    col2.metric("Climate Stress Tier", "Elevated")
    st.info("Action Mandate: Distribute water purification tablets to local schools.")

with tab2:
    st.header("Historical Trends")
    # Interactive Visualization
    fig = px.line(df, x='date', y='cases', title="Cholera Outbreak Trends (2015-2026)")
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Model Performance Validation")
    st.write("Validation Methodology: Chronological Forward-Chaining Cross-Validation.")
    
    # Consistent Metric Display
    results_data = {
        "Model": ["Poisson GLM", "Multi-Lag Ridge", "XGBoost"],
        "MAE": [7.23, 6.53, 5.07],
        "R²": [-0.69, 0.02, 0.04]
    }
    st.table(pd.DataFrame(results_data))
