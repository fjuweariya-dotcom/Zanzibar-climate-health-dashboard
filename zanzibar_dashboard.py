import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import geopandas as gpd
from shapely.geometry import Polygon
import warnings
import os

# Set academic publication visual standards
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.autolayout': True,
    'legend.frameon': True,
    'legend.fancybox': True
})

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Zanzibar Health Early Warning System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #021B2E 0%, #065A82 100%);
        padding: 20px 30px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .main-header h1 { color: #02C39A; margin: 0; font-size: 28px; }
    .main-header p  { color: #94B8C8; margin: 5px 0 0 0; font-size: 14px; }
    .metric-card {
        background: #021B2E;
        border: 1px solid #065A82;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-value { font-size: 32px; font-weight: bold; color: #02C39A; }
    .metric-label { font-size: 12px; color: #94B8C8; margin-top: 5px; }
    .alert-red    { background:#e74c3c22; border:2px solid #e74c3c; border-radius:8px; padding:15px; }
    .alert-orange { background:#e67e2222; border:2px solid #e67e22; border-radius:8px; padding:15px; }
    .alert-yellow { background:#f1c40f22; border:2px solid #f1c40f; border-radius:8px; padding:15px; }
    .alert-green  { background:#2ecc7122; border:2px solid #2ecc71; border-radius:8px; padding:15px; }
</style>
""", unsafe_allow_html=True)

# Get the absolute path of the directory where this script runs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, "data", "clean", "zanzibar_full_risk_dataset.csv")
    try:
        df = pd.read_csv(data_path, parse_dates=["date"])
    except FileNotFoundError:
        st.error(f"Data file not found at: {data_path}. Please check your GitHub folder structure.")
        st.stop()
    return df
    
@st.cache_data
def load_districts():
    district_data = pd.DataFrame([
        {"district":"Mjini",       "region":"Mjini Magharibi","island":"Unguja", "population":250000,"cholera_risk":0.95,"heat_risk":0.90,"water_access":0.85,"combined_risk":0.672,"risk_category":"High"},
        {"district":"Magharibi",   "region":"Mjini Magharibi","island":"Unguja", "population":180000,"cholera_risk":0.75,"heat_risk":0.70,"water_access":0.70,"combined_risk":0.588,"risk_category":"Medium"},
        {"district":"Kaskazini A", "region":"Unguja Kaskazini","island":"Unguja", "population":95000,"cholera_risk":0.50,"heat_risk":0.52,"water_access":0.65,"combined_risk":0.460,"risk_category":"Low"},
        {"district":"Kaskazini B", "region":"Unguja Kaskazini","island":"Unguja", "population":88000,"cholera_risk":0.48,"heat_risk":0.50,"water_access":0.62,"combined_risk":0.456,"risk_category":"Low"},
        {"district":"Kusini",      "region":"Unguja Kusini","island":"Unguja", "population":120000,"cholera_risk":0.65,"heat_risk":0.60,"water_access":0.55,"combined_risk":0.572,"risk_category":"Medium"},
        {"district":"Kati",        "region":"Unguja Kusini","island":"Unguja", "population":75000,"cholera_risk":0.45,"heat_risk":0.50,"water_access":0.60,"combined_risk":0.448,"risk_category":"Low"},
        {"district":"Wete",        "region":"Pemba Kaskazini","island":"Pemba", "population":110000,"cholera_risk":0.60,"heat_risk":0.55,"water_access":0.50,"combined_risk":0.556,"risk_category":"Medium"},
        {"district":"Micheweni",   "region":"Pemba Kaskazini","island":"Pemba", "population":65000,"cholera_risk":0.40,"heat_risk":0.45,"water_access":0.65,"combined_risk":0.402,"risk_category":"Low"},
        {"district":"Chake Chake", "region":"Pemba Kusini","island":"Pemba", "population":95000,"cholera_risk":0.55,"heat_risk":0.60,"water_access":0.55,"combined_risk":0.532,"risk_category":"Medium"},
        {"district":"Mkoani",      "region":"Pemba Kusini","island":"Pemba", "population":80000,"cholera_risk":0.50,"heat_risk":0.50,"water_access":0.60,"combined_risk":0.473,"risk_category":"Low"},
        {"district":"Chakechake North","region":"Pemba Kusini","island":"Pemba", "population":40000,"cholera_risk":0.45,"heat_risk":0.48,"water_access":0.62,"combined_risk":0.439,"risk_category":"Low"},
    ])
    polygons = {
        "Mjini":            Polygon([(39.15,-6.08),(39.24,-6.08),(39.24,-6.22),(39.15,-6.22)]),
        "Magharibi":      Polygon([(39.08,-6.08),(39.15,-6.08),(39.15,-6.38),(39.08,-6.28)]),
        "Kaskazini A":    Polygon([(39.18,-5.75),(39.32,-5.75),(39.35,-5.95),(39.18,-5.98)]),
        "Kaskazini B":    Polygon([(39.32,-5.75),(39.46,-5.82),(39.46,-6.02),(39.35,-5.95)]),
        "Kusini":         Polygon([(39.18,-6.28),(39.44,-6.18),(39.48,-6.50),(39.18,-6.52)]),
        "Kati":           Polygon([(39.18,-6.08),(39.44,-6.05),(39.44,-6.18),(39.18,-6.28)]),
        "Wete":           Polygon([(39.65,-4.88),(39.80,-4.88),(39.82,-5.08),(39.65,-5.08)]),
        "Micheweni":      Polygon([(39.78,-4.75),(39.92,-4.75),(39.92,-4.92),(39.78,-4.92)]),
        "Chake Chake":    Polygon([(39.68,-5.18),(39.82,-5.08),(39.85,-5.30),(39.68,-5.30)]),
        "Mkoani":         Polygon([(39.62,-5.30),(39.78,-5.30),(39.80,-5.48),(39.62,-5.48)]),
        "Chakechake North":Polygon([(39.65,-5.08),(39.82,-5.08),(39.82,-5.18),(39.65,-5.18)]),
    }
    geoms = [polygons[d] for d in district_data["district"]]
    gdf = gpd.GeoDataFrame(district_data, geometry=geoms, crs="EPSG:4326")
    return gdf

def get_alert_color(score):
    if score >= 70: return "#e74c3c", "🔴 CRITICAL"
    if score >= 50: return "#e67e22", "🟠 HIGH"
    if score >= 30: return "#f1c40f", "🟡 MODERATE"
    return "#2ecc71", "🟢 LOW"

def compute_risk_score(temp_anom, rainfall, heat_idx, cholera, malaria):
    def norm(v, lo, hi): return max(0, min(1, (v-lo)/(hi-lo)))
    s = (norm(temp_anom, -3, 3)    * 20 +
         norm(rainfall,   0, 500)  * 20 +
         norm(heat_idx,  24, 40)   * 20 +
         norm(cholera,    0, 500)  * 20 +
         norm(malaria,    0, 2000) * 20)
    return round(s, 1)

# ── Load datasets ────────────────────────────────────────────
df  = load_data()
gdf = load_districts()

# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🌊 Zanzibar Health Early Warning System</h1>
  <p>Climate-Epidemiological Analysis • Heat Effects & Waterborne Diseases • 2010–2026</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Zanzibar_in_Tanzania_%28special_marker%29.svg/200px-Zanzibar_in_Tanzania_%28special_marker%29.svg.png", width=120)
st.sidebar.title("🎛️ Controls")
page = st.sidebar.radio("Navigate", [
    "📊 Overview Dashboard",
    "🌡️ Climate Explorer",
    "🦠 Disease Analysis",
    "🗺️ District Risk Map",
    "🚦 Early Warning Tool",
    "📈 Forecast Viewer",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**📚 Data Sources**")
st.sidebar.markdown("""
- Climate: Open-Meteo ERA5
- Cholera: Bi et al. 2018; WHO AFRO
- Malaria: Abbas et al. 2023
- Districts: Tanzania NBS
""")
st.sidebar.markdown("**👩‍🔬 Author:** Juweariya Farouk")
st.sidebar.markdown("**📅 Date:** May 2026")

# ════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW DASHBOARD
# ════════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.header("📊 Project Overview")

    # Key metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "5,964", "Days of Climate Data"),
        (c2, "194",   "Months Analysed"),
        (c3, "11",    "Zanzibar Districts"),
        (c4, "4,479", "Total Cholera Cases"),
        (c5, "7.17",  "Best Model MAE"),
    ]
    for col, val, lbl in metrics:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Two column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🦠 Cholera Cases Over Time")
        fig1, ax1 = plt.subplots(figsize=(8, 4.5))
        
        colors = ["#d9534f" if o else "#f0ad4e" for o in df["outbreak_year"]]
        ax1.bar(df["date"], df["cholera_cases"], width=25, color=colors, alpha=0.9)
        
        ax1.set_ylabel("Reported Clinical Cases", fontweight="bold")
        ax1.set_xlabel("Epidemiological Timeline", fontweight="bold")
        ax1.set_title("Monthly Cholera Cases Profile (2010–2026)", fontsize=12, fontweight="bold", loc="left")
        
        ax1.xaxis.set_major_locator(mdates.YearLocator(2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
        
        from matplotlib.patches import Patch
        ax1.legend(handles=[
            Patch(color="#d9534f", label="Outbreak Year"),
            Patch(color="#f0ad4e", label="Non-Outbreak Observation")
        ], loc="upper right")
        
        st.pyplot(fig1, use_container_width=True)
        plt.close()

    with col2:
        st.subheader("🌡️ Heat Index Trend")
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        
        ax2.plot(df["date"], df["heat_index_c"], color="#c0392b", linewidth=1.8, label="Calculated Heat Index")
        ax2.fill_between(df["date"], df["heat_index_c"], alpha=0.15, color="#c0392b")
        ax2.axhline(y=32, color="#e67e22", linestyle="--", alpha=0.8, linewidth=1.5, label="Caution Threshold (32°C)")
        
        ax2.set_ylabel("Heat Index Value (°C)", fontweight="bold")
        ax2.set_xlabel("Observation Timeline", fontweight="bold")
        ax2.set_title("Monthly Mean Heat Index Profile (2010–2026)", fontsize=12, fontweight="bold", loc="left")
        
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower left")
        
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.subheader("🔑 Key System Revelations")
    
    f1, f2, f3 = st.columns(3)
    with f1:
        st.info("""🌧️ **Rainfall Lag Horizon**\n\nA 2-month lagged environmental precipitation window acts as the dominant predictor of cholera index spikes across Zanzibar's administrative bounds.""")
    with f2:
        st.warning("""🌡️ **Heat Index Multipliers**\n\nThermal extremes paired with compounding relative humidity levels multiply biological viral propagation bounds — raw surface temperature shifts alone do not capture this footprint.""")
    with f3:
        st.success("""⏱️ **Temporal Lead Window**\n\nClimatic trend shifts directly precede historical clinical tracking metrics by 6 to 8 weeks — forming a core advance warning deployment asset for public health units.""")

# ════════════════════════════════════════════════════════════
# PAGE 2: CLIMATE EXPLORER
# ════════════════════════════════════════════════════════════
elif page == "🌡️ Climate Explorer":
    st.header("🌡️ Climate Data Explorer")

    year_range = st.slider("Select Year Range", int(df["year"].min()), int(df["year"].max()), (2010, 2026))
    df_filtered = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

    var = st.selectbox("Select Climate Variable", [
        "temp_mean_c", "temp_max_c", "rainfall_mm",
        "heat_index_c", "humidity_mean_pct", "extreme_heat_days"
    ])

    var_labels = {
        "temp_mean_c":      "Mean Temperature (°C)",
        "temp_max_c":       "Max Temperature (°C)",
        "rainfall_mm":      "Rainfall (mm)",
        "heat_index_c":     "Heat Index (°C)",
        "humidity_mean_pct":"Humidity (%)",
        "extreme_heat_days":"Extreme Heat Days"
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(df_filtered["date"], df_filtered[var], color="#065A82", linewidth=1.5)
    axes[0].fill_between(df_filtered["date"], df_filtered[var], alpha=0.15, color="#065A82")
    axes[0].set_title(f"{var_labels[var]} — {year_range[0]} to {year_range[1]}")
    axes[0].set_ylabel(var_labels[var])
    axes[0].grid(alpha=0.3)

    monthly = df_filtered.groupby("month")[var].mean()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    axes[1].bar(month_names, monthly.values, color="#1C7293", alpha=0.85)
    axes[1].set_title(f"Average {var_labels[var]} by Month")
    axes[1].set_ylabel(var_labels[var])
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2, col3 = st.columns(3)
    col1.metric("Average", f"{df_filtered[var].mean():.1f}")
    col2.metric("Maximum", f"{df_filtered[var].max():.1f}")
    col3.metric("Minimum", f"{df_filtered[var].min():.1f}")

# ════════════════════════════════════════════════════════════
# PAGE 3: DISEASE ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "🦠 Disease Analysis":
    st.header("🦠 Disease Pattern Analysis")

    tab1, tab2, tab3 = st.tabs(["Cholera Trends", "Seasonal Patterns", "Climate Correlation"])

    with tab1:
        st.subheader("Cholera Cases 2010–2026")
        show_outbreak = st.checkbox("Highlight outbreak years", value=True)

        fig, ax = plt.subplots(figsize=(14, 5))
        colors = ["#e74c3c" if (o and show_outbreak) else "#e67e22" for o in df["outbreak_year"]]
        ax.bar(df["date"], df["cholera_cases"], width=25, color=colors, alpha=0.85)

        for year in [2015, 2016]:
            peak = df[df["year"]==year]["cholera_cases"].max()
            peak_dates = df[(df["year"]==year) & (df["cholera_cases"]==peak)]["date"].values
            if len(peak_dates) > 0:
                peak_date = peak_dates[0]
                ax.annotate(f"{year} ({peak} cases)", xy=(peak_date, peak), xytext=(0, 15),
                            textcoords="offset points", ha="center", fontsize=8, color="red",
                            arrowprops=dict(arrowstyle="->", color="red"))

        ax.set_ylabel("Cases")
        ax.grid(alpha=0.3, axis="y")
        st.pyplot(fig)
        plt.close()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Cases",  f"{int(df['cholera_cases'].sum()):,}")
        col2.metric("Peak Month",   df.loc[df['cholera_cases'].idxmax(),'date'].strftime('%b %Y'))
        col3.metric("Peak Cases",   f"{int(df['cholera_cases'].max())}")
        col4.metric("Avg/Month",    f"{df['cholera_cases'].mean():.1f}")

    with tab2:
        st.subheader("Seasonal Disease & Climate Patterns")
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly = df.groupby("month").agg(
            cholera=("cholera_cases","mean"),
            temp=("temp_mean_c","mean"),
            rain=("rainfall_mm","mean"),
            heat=("heat_index_c","mean"),
        ).reset_index()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, col, title, color in zip(axes.flat,
            ["cholera","temp","rain","heat"],
            ["Avg Cholera Cases","Avg Temperature (°C)","Avg Rainfall (mm)","Avg Heat Index (°C)"],
            ["#e74c3c","#e67e22","#3498db","#c0392b"]):
            ax.bar(month_names, monthly[col], color=color, alpha=0.85)
            ax.set_title(title, fontweight="bold")
            ax.grid(alpha=0.3, axis="y")
            for mi in [2,3,4,9,10,11]:
                ax.get_children()[mi].set_edgecolor("#2980b9")
                ax.get_children()[mi].set_linewidth(2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Blue borders = rainy season months (Mar–May, Oct–Dec)")

    with tab3:
        st.subheader("Climate-Cholera Correlation Analysis")
        corr_vars = {
            "Mean Temperature":    "temp_mean_c",
            "Rainfall":            "rainfall_mm",
            "Heat Index":          "heat_index_c",
            "Temp (1-month lag)":  "temp_mean_lag1",
            "Temp (2-month lag)":  "temp_mean_lag2",
            "Rainfall (1-month lag)": "rainfall_lag1",
            "Rainfall (2-month lag)": "rainfall_lag2",
            "3-month Cumul. Rain":    "rainfall_3m_cumul",
        }
        corr_data = []
        for label, col in corr_vars.items():
            if col in df.columns:
                r = df[col].corr(df["cholera_cases"])
                corr_data.append({"Variable": label, "Pearson r": round(r,3)})
        corr_df = pd.DataFrame(corr_data).sort_values("Pearson r")

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in corr_df["Pearson r"]]
        ax.barh(corr_df["Variable"], corr_df["Pearson r"], color=colors, alpha=0.85)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_title("Correlation with Cholera Cases")
        ax.set_xlabel("Pearson r")
        ax.grid(alpha=0.3, axis="x")
        st.pyplot(fig)
        plt.close()

        st.dataframe(corr_df, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 4: DISTRICT RISK MAP
# ════════════════════════════════════════════════════════════
elif page == "🗺️ District Risk Map":
    st.header("🗺️ District Vulnerability Map")

    risk_var = st.selectbox("Show risk type:", ["combined_risk","cholera_risk","heat_risk"])

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_facecolor("#a8d8ea")
    gdf.plot(column=risk_var, ax=ax, cmap="RdYlGn_r", vmin=0.3, vmax=0.9,
             edgecolor="white", linewidth=1.2, alpha=0.92, legend=True,
             legend_kwds={"label":"Risk Score","shrink":0.6})

    for _, row in gdf.iterrows():
        cx = row.geometry.centroid.x
        cy = row.geometry.centroid.y
        ax.annotate(row["district"], xy=(cx, cy), ha="center", va="center", fontsize=7.5,
                    fontweight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, lw=0))

    ax.text(39.28, -5.70, "UNGUJA ISLAND", fontsize=10, color="#2c3e50", fontweight="bold", ha="center", style="italic")
    ax.text(39.80, -5.20, "PEMBA ISLAND", fontsize=10, color="#2c3e50", fontweight="bold", ha="center", style="italic")
    ax.set_title(f"Zanzibar — {risk_var.replace('_',' ').title()}", fontsize=13, fontweight="bold")
    ax.set_xlim(39.00, 39.98)
    ax.set_ylim(-6.65, -4.65)
    ax.grid(alpha=0.25)
    ax.annotate("N ▲", xy=(39.06,-4.80), fontsize=12, fontweight="bold")
    ax.text(39.06,-6.58,"© NBS Tanzania / Zanzibar OCGS", fontsize=7, color="#888")
    st.pyplot(fig)
    plt.close()

    st.subheader("District Risk Summary")
    display = gdf[["district","region","island","population","cholera_risk","heat_risk","combined_risk","risk_category"]].copy()
    display = display.sort_values("combined_risk", ascending=False)
    display.columns = ["District","Region","Island","Population","Cholera Risk","Heat Risk","Combined Risk","Category"]

    def color_category(val):
        if val == "High":   return "background-color: #e74c3c33; color: #e74c3c"
        if val == "Medium": return "background-color: #e67e2233; color: #e67e22"
        return "background-color: #2ecc7133; color: #2ecc71"

    st.dataframe(display.style.map(color_category, subset=["Category"]), use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 5: EARLY WARNING TOOL
# ════════════════════════════════════════════════════════════
elif page == "🚦 Early Warning Tool":
    st.header("🚦 Early Warning Risk Calculator")
    st.markdown("Adjust climate and disease inputs to compute the current health risk score.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌡️ Climate Inputs")
        temp = st.slider("Mean Temperature (°C)", 22.0, 35.0, 27.0, 0.1)
        rainfall = st.slider("Monthly Rainfall (mm)", 0, 600, 100, 10)
        heat_idx = st.slider("Heat Index (°C)", 24.0, 42.0, 29.0, 0.1)
        temp_anom = st.slider("Temperature Anomaly (°C above normal)", -3.0, 3.0, 0.0, 0.1)

    with col2:
        st.subheader("🦠 Disease Inputs")
        cholera = st.slider("Cholera Cases (this month)", 0, 500, 10, 1)
        malaria = st.slider("Malaria Cases (this month)", 0, 2000, 200, 10)
        month_sel = st.selectbox("Month", [
            "January","February","March","April","May","June",
            "July","August","September","October","November","December"
        ], index=3)
        season = "Long Rains" if month_sel in ["March","April","May"] else "Short Rains" if month_sel in ["October","November","December"] else "Dry Season"
        st.info(f"📅 Season: **{season}**")

    score = compute_risk_score(temp_anom, rainfall, heat_idx, cholera, malaria)
    color, label = get_alert_color(score)

    st.markdown("---")
    st.subheader("🎯 Risk Assessment Result")

    css_class = {
        "🔴 CRITICAL": "alert-red",
        "🟠 HIGH":     "alert-orange",
        "🟡 MODERATE": "alert-yellow",
        "🟢 LOW":      "alert-green",
    }[label]

    actions = {
        "🔴 CRITICAL": "⚠️ Immediate response required. Deploy emergency water treatment teams. Issue public health alerts to all districts. Activate district response teams immediately.",
        "🟠 HIGH":     "⚠️ Heightened surveillance. Pre-position oral rehydration salts at health facilities. Alert district health officers. Increase water quality monitoring.",
        "🟡 MODERATE": "ℹ️ Enhanced monitoring recommended. Launch community awareness campaigns. Inspect water infrastructure. Prepare health worker briefings.",
        "🟢 LOW":      "✅ Routine surveillance. Maintain standard prevention messaging. Continue regular water quality checks.",
    }[label]

    st.markdown(f"""
    <div class="{css_class}">
        <h2 style="margin:0">Risk Score: {score}/100 — {label}</h2>
        <p style="margin:10px 0 0 0">{actions}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Score Breakdown")
    def norm(v, lo, hi): return max(0, min(1, (v-lo)/(hi-lo))) * 20
    breakdown = pd.DataFrame({
        "Component":  ["Temperature Anomaly","Rainfall","Heat Index","Cholera Cases","Malaria Cases"],
        "Score (0-20)":[round(norm(temp_anom,-3,3),1), round(norm(rainfall,0,500),1),
                        round(norm(heat_idx,24,40),1), round(norm(cholera,0,500),1),
                        round(norm(malaria,0,2000),1)],
        "Weight": ["20%","20%","20%","20%","20%"]
    })
    st.dataframe(breakdown, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 6: FORECAST VIEWER
# ════════════════════════════════════════════════════════════
elif page == "📈 Forecast Viewer":
    st.header("📈 Historical Trends & Forecasting")

    disease = st.radio("Select Disease", ["Cholera", "Malaria"], horizontal=True)
    col_name = "cholera_cases" if disease == "Cholera" else "malaria_cases"

    if col_name not in df.columns:
        st.warning(f"{disease} data not found in dataset.")
    else:
        years = st.slider("Year Range", 2010, 2026, (2010, 2026))
        df_f = df[(df["year"] >= years[0]) & (df["year"] <= years[1])]

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

        color = "#e74c3c" if disease == "Cholera" else "#27ae60"
        axes[0].bar(df_f["date"], df_f[col_name], width=26, color=color, alpha=0.8)
        axes[0].set_title(f"{disease} Cases Over Time", fontweight="bold")
        axes[0].set_ylabel("Cases")
        axes[0].grid(alpha=0.3)

        axes[1].plot(df_f["date"], df_f["heat_index_c"], color="#c0392b", linewidth=1.5, label="Heat Index (°C)")
        ax2 = axes[1].twinx()
        ax2.bar(df_f["date"], df_f["rainfall_mm"], width=26, color="#3498db", alpha=0.4, label="Rainfall (mm)")
        axes[1].set_title("Climate Drivers", fontweight="bold")
        axes[1].set_ylabel("Heat Index (°C)", color="#c0392b")
        ax2.set_ylabel("Rainfall (mm)", color="#3498db")
        axes[1].legend(loc="upper left")
        ax2.legend(loc="upper right")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("📊 Model Performance Summary")
        results = pd.DataFrame([
            {"Disease": "Cholera", "Model": "Hybrid Ensemble", "MAE": 7.17, "RMSE": 9.33, "R²": 0.210, "Status": "✅ Best"},
            {"Disease": "Cholera", "Model": "XGBoost", "MAE": 7.23, "RMSE": 9.54, "R²": 0.173, "Status": "Good"},
            {"Disease": "Cholera", "Model": "SARIMA", "MAE": 9.25, "RMSE": 11.40, "R²": -0.181, "Status": "Baseline"},
            {"Disease": "Malaria", "Model": "XGBoost", "MAE": 262.47, "RMSE": 359.42, "R²": -0.081, "Status": "✅ Best"},
        ])
        filtered = results[results["Disease"] == disease]
        st.dataframe(filtered, use_container_width=True)

        st.info("""
        **📌 Model Notes:**
        - **Cholera Hybrid Model** (MAE=7.17) is the best performing model
        - **Malaria model** performance is limited by the 2024 epidemic spike
        - Both models use climate lag variables (1–2 month lags) as key features
        - Rainfall_lag2 is the #1 predictor for cholera
        """)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94B8C8; font-size:12px; padding:10px">
    Zanzibar Health Early Warning System •
    Juweariya Farouk • Academic Research Thesis • May 2026<br>
    Data: Open-Meteo ERA5 | Bi et al. 2018 | Abbas et al. 2023 |
    WHO AFRO | Malaria Journal 2025
</div>
""", unsafe_allow_html=True)
