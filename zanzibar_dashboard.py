import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Zanzibar Health EWS", layout="wide")

# --- DATA LOADING (Dynamic & Resilient) ---
@st.cache_data
def load_data():
    # 1. Path Management
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "data", "clean", "zanzibar_full_risk_dataset.csv")
    
    # 2. Load and Detect Variable
    df = pd.read_csv(data_path, parse_dates=["date"])
    
    # 3. Resilient Mapping Hook (Scanning for target column)
    target_candidates = ['cases', 'Cholera_Cases', 'Cholera_cases', 'Cases']
    target_col = next((col for col in df.columns if col in target_candidates), None)
    
    if target_col:
        df = df.rename(columns={target_col: 'cases'})
    return df

df = load_data()

# --- UI LAYOUT ---
st.title("🌊 Zanzibar Climate and Disease Analytics")
tab1, tab2, tab3 = st.tabs(["Live Alerts", "Historical Trends", "Model Performance"])

with tab1:
    st.header("Public Health Alert Dashboard")
    col1, col2 = st.columns(2)
    # Using your validated logic for metric display
    col1.metric("Current Incident Rate", f"{df['cases'].iloc[-1]:.0f}")
    col2.metric("Climate Stress Tier", "Elevated")

with tab2:
    st.header("Historical Trends")
    # Interactive Plotting
    fig = px.line(df, x='date', y='cases', title="Outbreak Trends (2015-2026)")
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Model Performance")
    # Displaying metrics from your ablation study
    results_data = {
        "Model": ["Baseline", "Pruned/Quantized", "Coreset (18 samples)"],
        "MAE": [7.23, 6.53, 5.07],
        "R²": [0.173, 0.226, 0.603]
    }
    st.table(pd.DataFrame(results_data))
