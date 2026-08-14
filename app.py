import os
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agri Smart | Crop Yield Prediction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "FAO_Crop_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "crop_yield_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "crop_encoder.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

FEATURE_ORDER = [
    "rainfall_mm",
    "temperature_C",
    "humidity_pct",
    "soil_N_kg_ha",
    "soil_P_kg_ha",
    "soil_K_kg_ha",
    "soil_pH",
    "crop_ENC",
]

# ----------------------------------------------------------------------------
# THEME / CSS  (dark charcoal main area, dark green sidebar, green accents)
# ----------------------------------------------------------------------------
CSS = """
<style>
:root{
    --bg-main:#0e1414;
    --bg-card:#161f1c;
    --bg-card-border:#233029;
    --sidebar-bg:#0b2119;
    --sidebar-bg-2:#0d2a1f;
    --green:#22c55e;
    --green-dim:#16a34a;
    --green-soft:rgba(34,197,94,0.15);
    --blue:#3b82f6;
    --text-main:#f5f7f6;
    --text-dim:#9ca8a3;
    --text-faint:#6b7670;
}

html, body, [data-testid="stAppViewContainer"]{
    background-color: var(--bg-main) !important;
    color: var(--text-main);
}
[data-testid="stHeader"]{ background-color: rgba(0,0,0,0); }
[data-testid="stAppViewContainer"] > .main { background-color: var(--bg-main); }
.block-container{ padding-top: 2rem; padding-bottom: 2rem; max-width: 1500px; }

/* Sidebar */
[data-testid="stSidebar"]{
    background: linear-gradient(180deg, var(--sidebar-bg) 0%, var(--sidebar-bg-2) 100%);
    border-right: 1px solid #0a1a13;
}
[data-testid="stSidebar"] .block-container{ padding-top: 1.6rem; }

.brand-row{ display:flex; align-items:center; gap:10px; margin-bottom: 2px;}
.brand-emoji{ font-size: 26px; }
.brand-title{ font-size: 21px; font-weight: 800; color: #ffffff; letter-spacing: -0.3px; }
.brand-sub{ font-size: 12.5px; color: #7fdba0; margin-left: 36px; margin-top: -6px; margin-bottom: 14px;}

.sb-divider{ border-top: 1px solid #1c3327; margin: 14px 0 16px 0; }

.sb-heading{
    font-size: 11px; font-weight: 700; letter-spacing: 1.2px;
    color: #6fae86; margin: 4px 0 10px 2px; text-transform: uppercase;
}

/* Nav buttons */
div[data-testid="stSidebar"] .stButton > button{
    width: 100%;
    text-align: left;
    background: transparent;
    color: #d6e6dc;
    border: none;
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 14.5px;
    font-weight: 500;
    margin-bottom: 3px;
    transition: background 0.15s ease;
}
div[data-testid="stSidebar"] .stButton > button:hover{
    background: rgba(34,197,94,0.10);
    color: #ffffff;
}
div[data-testid="stSidebar"] .stButton > button:focus{
    box-shadow: none !important;
}
.nav-active > button{
    background: var(--green-dim) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Dataset status card */
.dataset-card{
    background: rgba(34,197,94,0.06);
    border: 1px solid rgba(34,197,94,0.35);
    border-radius: 12px;
    padding: 14px 14px;
    margin-bottom: 12px;
}
.dataset-status{
    display:flex; align-items:center; justify-content:space-between;
    font-size: 13.5px; font-weight: 700; color: var(--green);
    margin-bottom: 6px;
}
.dataset-name{ font-size: 13.5px; color: #e8f3ec; font-weight:600; margin-bottom:2px;}
.dataset-shape{ font-size: 12px; color: var(--text-dim); }

.upload-hint{
    font-size: 11.5px; color: var(--text-faint); text-align:center; margin-top: 8px; line-height:1.4;
}

.sb-footer{
    font-size: 11.5px; color: var(--text-faint); margin-top: 18px; line-height: 1.5;
}

/* Main headings */
.page-title-row{ display:flex; align-items:center; gap: 14px; margin-bottom: 2px; }
.page-title-emoji{ font-size: 40px; }
.page-title{ font-size: 38px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px; }
.page-subtitle{ font-size: 15px; color: var(--text-dim); margin: 4px 0 24px 0; }

/* KPI cards */
.kpi-card{
    background: var(--bg-card);
    border: 1px solid var(--bg-card-border);
    border-radius: 14px;
    padding: 20px 20px;
    display:flex;
    align-items:center;
    gap: 16px;
}
.kpi-icon{
    font-size: 24px;
    background: var(--green-soft);
    border-radius: 10px;
    width: 46px; height: 46px;
    display:flex; align-items:center; justify-content:center;
    flex-shrink:0;
}
.kpi-value{ font-size: 26px; font-weight: 800; color: var(--green); line-height:1.1; }
.kpi-label{ font-size: 11.5px; color: var(--text-dim); letter-spacing: 0.6px; font-weight:600; margin-top:3px;}

/* Chart cards */
.chart-card{
    background: var(--bg-card);
    border: 1px solid var(--bg-card-border);
    border-radius: 14px;
    padding: 20px 20px 8px 20px;
    margin-bottom: 20px;
}
.chart-title{ font-size: 16.5px; font-weight: 700; color: #ffffff; margin-bottom: 6px; }

/* Info box */
.info-box{
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.30);
    border-radius: 12px;
    padding: 14px 18px;
    display:flex; gap: 12px; align-items:flex-start;
    margin-top: 4px;
}
.info-icon{
    background:#3b82f6; color:white; border-radius:50%;
    width:22px; height:22px; min-width:22px; display:flex; align-items:center; justify-content:center;
    font-size: 12px; font-weight:800; margin-top:2px;
}
.info-title{ font-size: 14px; font-weight: 700; color: #7fb2ff; }
.info-sub{ font-size: 12.5px; color: var(--text-dim); margin-top: 2px; }

/* Predict result card */
.result-card{
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.04));
    border: 1px solid rgba(34,197,94,0.45);
    border-radius: 16px;
    padding: 30px;
    text-align:center;
    margin-top: 10px;
}
.result-label{ font-size: 13px; font-weight:700; letter-spacing:1.2px; color: var(--text-dim); }
.result-value{ font-size: 46px; font-weight: 900; color: var(--green); margin: 8px 0 0 0; }
.result-unit{ font-size: 16px; color: var(--text-dim); font-weight:600; }

/* Metric mini cards */
.metric-mini{
    background: var(--bg-card);
    border: 1px solid var(--bg-card-border);
    border-radius: 12px;
    padding: 16px;
    text-align:center;
}
.metric-mini-value{ font-size: 22px; font-weight:800; color:#ffffff; }
.metric-mini-label{ font-size: 11.5px; color: var(--text-dim); margin-top:4px; letter-spacing:0.5px; }

/* Inputs */
.stNumberInput input, .stSelectbox div[data-baseweb="select"]{
    background-color: #111a17 !important;
    color: #f5f7f6 !important;
    border-radius: 8px !important;
}
label{ color: #cfe0d6 !important; font-size: 13.5px !important; font-weight:600 !important; }

.stButton > button[kind="primary"]{
    background: var(--green-dim);
    border: none;
    font-weight: 700;
    border-radius: 10px;
    padding: 0.6rem 1rem;
}

hr{ border-color: #1c2b24; }

/* Crop analysis card */
.crop-stat-card{
    background: var(--bg-card);
    border: 1px solid var(--bg-card-border);
    border-radius: 14px;
    padding: 18px;
    text-align:center;
}
.crop-stat-value{ font-size: 24px; font-weight:800; color: var(--green); }
.crop-stat-label{ font-size: 12px; color: var(--text-dim); margin-top:4px; }

::-webkit-scrollbar{ width: 8px; }
::-webkit-scrollbar-track{ background: var(--bg-main); }
::-webkit-scrollbar-thumb{ background: #1e3a2c; border-radius: 4px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# CACHED LOADERS
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_encoder():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)):
        train_from_scratch()
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return model, encoder, metrics


def train_from_scratch():
    """Fallback: retrain using the exact notebook pipeline if pkl files
    are missing (keeps the app self-sufficient)."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    df = pd.read_csv(DEFAULT_DATA_PATH)
    df_model = df.drop(columns=["district", "state", "year"], errors="ignore")
    df_model = df_model.dropna()

    le_crop = LabelEncoder()
    df_model["crop_ENC"] = le_crop.fit_transform(df_model["crop"])
    df_model = df_model.drop(columns=["crop"])

    X = df_model[FEATURE_ORDER]
    y = df_model["yield_kg_ha"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": r2_score(y_test, y_pred),
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le_crop, ENCODER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)


@st.cache_data
def load_default_data():
    return pd.read_csv(DEFAULT_DATA_PATH)


def dataset_summary(df):
    return {
        "records": len(df),
        "crops": df["crop"].nunique() if "crop" in df.columns else 0,
        "states": df["state"].nunique() if "state" in df.columns else 0,
        "year_min": int(df["year"].min()) if "year" in df.columns else None,
        "year_max": int(df["year"].max()) if "year" in df.columns else None,
    }


# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "custom_df" not in st.session_state:
    st.session_state.custom_df = None
if "custom_filename" not in st.session_state:
    st.session_state.custom_filename = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

model, crop_encoder, metrics = load_model_and_encoder()
default_df = load_default_data()

using_custom = st.session_state.custom_df is not None
active_df = st.session_state.custom_df if using_custom else default_df
active_label = st.session_state.custom_filename if using_custom else "FAO_Crop_data.csv"

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
            <span class="brand-emoji">🌱</span>
            <span class="brand-title">Agri Smart</span>
        </div>
        <div class="brand-sub">AI for Agriculture</div>
        <div class="sb-divider"></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-heading">NAVIGATION</div>', unsafe_allow_html=True)

    nav_items = [
        ("Dashboard", "🏠 Dashboard"),
        ("Predict Yield", "🌱 Predict Yield"),
        ("Data Insights", "📊 Data Insights"),
        ("Crop Analysis", "🌾 Crop Analysis"),
        ("About Project", "ℹ️ About Project"),
    ]

    for key, label in nav_items:
        active = st.session_state.page == key
        wrap_class = "nav-active" if active else "nav-inactive"
        st.markdown(f'<div class="{wrap_class}">', unsafe_allow_html=True)
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-heading">DATASET</div>', unsafe_allow_html=True)

    summary = dataset_summary(active_df)
    status_label = "Custom Dataset Loaded" if using_custom else "Default Dataset Loaded"

    st.markdown(
        f"""
        <div class="dataset-card">
            <div class="dataset-status"><span>✓ {status_label}</span></div>
            <div class="dataset-name">{active_label}</div>
            <div class="dataset-shape">{summary['records']:,} rows × {active_df.shape[1]} columns</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "📤 Upload New Dataset", type=["csv"], label_visibility="collapsed"
    )
    st.markdown(
        '<div class="upload-hint">Upload only if you want to try<br>your own data.</div>',
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            required_cols = {"crop", "yield_kg_ha"}
            if not required_cols.issubset(set(new_df.columns)):
                st.error("This file doesn't look like a valid crop dataset. Keeping default dataset.")
            else:
                st.session_state.custom_df = new_df
                st.session_state.custom_filename = uploaded_file.name
                st.rerun()
        except Exception:
            st.error("Couldn't read that file. Keeping default dataset.")

    if using_custom:
        if st.button("↺ Reset to default dataset", use_container_width=True):
            st.session_state.custom_df = None
            st.session_state.custom_filename = None
            st.rerun()

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-footer">© 2024 Agri Smart<br>All rights reserved.</div>',
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# SHARED HELPERS
# ----------------------------------------------------------------------------
def page_header(emoji, title, subtitle):
    st.markdown(
        f"""
        <div class="page-title-row">
            <span class="page-title-emoji">{emoji}</span>
            <span class="page-title">{title}</span>
        </div>
        <div class="page-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(icon, value, label):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def styled_bar_fig(x, y, orientation, color):
    fig = go.Figure()
    if orientation == "v":
        fig.add_bar(x=x, y=y, marker_color=color, marker_line_width=0)
    else:
        fig.add_bar(x=x, y=y, orientation="h", marker_color=color, marker_line_width=0)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cfe0d6", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#22302a", zeroline=False),
        yaxis=dict(gridcolor="#22302a", zeroline=False),
        height=380,
    )
    return fig


def info_box(html_title, html_sub):
    st.markdown(
        f"""
        <div class="info-box">
            <div class="info-icon">i</div>
            <div>
                <div class="info-title">{html_title}</div>
                <div class="info-sub">{html_sub}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# PAGE: DASHBOARD
# ----------------------------------------------------------------------------
def render_dashboard():
    page_header(
        "🌱",
        "Crop Yield Prediction",
        "Predict crop yield based on environmental, soil and crop conditions",
    )

    s = dataset_summary(active_df)
    year_range = f"{s['year_min']}–{s['year_max']}" if s["year_min"] else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("📄", f"{s['records']:,}", "TOTAL RECORDS")
    with c2:
        kpi_card("🌿", s["crops"], "TOTAL CROPS")
    with c3:
        kpi_card("🗺️", s["states"], "TOTAL STATES")
    with c4:
        kpi_card("📅", year_range, "YEAR RANGE")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🌾 Average Yield by Crop</div>', unsafe_allow_html=True)
        if "crop" in active_df.columns and "yield_kg_ha" in active_df.columns:
            crop_avg = (
                active_df.dropna(subset=["yield_kg_ha"])
                .groupby("crop")["yield_kg_ha"]
                .mean()
                .sort_values(ascending=False)
            )
            fig = styled_bar_fig(crop_avg.index, crop_avg.values, "v", "#22c55e")
            fig.update_yaxes(title="Avg Yield (kg/ha)")
            fig.update_xaxes(title="Crop")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Dataset missing 'crop' or 'yield_kg_ha' columns.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🗺️ Average Yield by State</div>', unsafe_allow_html=True)
        if "state" in active_df.columns and "yield_kg_ha" in active_df.columns:
            state_avg = (
                active_df.dropna(subset=["yield_kg_ha"])
                .groupby("state")["yield_kg_ha"]
                .mean()
                .sort_values(ascending=True)
            )
            fig = styled_bar_fig(state_avg.values, state_avg.index, "h", "#3b82f6")
            fig.update_xaxes(title="Avg Yield (kg/ha)")
            fig.update_yaxes(title="State")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Dataset missing 'state' or 'yield_kg_ha' columns.")
        st.markdown("</div>", unsafe_allow_html=True)

    if using_custom:
        info_box(
            f"Using custom dataset: {active_label}",
            "This data is used only for this session's dashboard analytics.",
        )
    else:
        info_box(
            "Using default dataset: FAO_Crop_data.csv",
            "You can upload your own dataset from the sidebar if you want to try custom data.",
        )


# ----------------------------------------------------------------------------
# PAGE: PREDICT YIELD
# ----------------------------------------------------------------------------
def render_predict():
    page_header("🌱", "Predict Yield", "Estimate crop yield using the trained Random Forest model")

    crop_classes = list(crop_encoder.classes_)

    with st.container():
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        colA, colB = st.columns(2)

        with colA:
            st.markdown("**Environmental Conditions**")
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=1000.0, step=10.0)
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=27.0, step=0.5)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            crop = st.selectbox("Crop", crop_classes)

        with colB:
            st.markdown("**Soil Conditions**")
            n = st.number_input("Soil Nitrogen (N)", min_value=0.0, max_value=500.0, value=80.0, step=1.0)
            p = st.number_input("Soil Phosphorus (P)", min_value=0.0, max_value=500.0, value=25.0, step=1.0)
            k = st.number_input("Soil Potassium (K)", min_value=0.0, max_value=500.0, value=45.0, step=1.0)
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        predict_clicked = st.button("🔮 Predict Yield", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        crop_enc = crop_encoder.transform([crop])[0]
        features = np.array([[rainfall, temperature, humidity, n, p, k, ph, crop_enc]])
        prediction = model.predict(features)[0]
        st.session_state.prediction_result = prediction

    if st.session_state.prediction_result is not None:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">PREDICTED CROP YIELD</div>
                <div class="result-value">{st.session_state.prediction_result:,.1f}</div>
                <div class="result-unit">kg/ha</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if metrics:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Model Performance</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-mini"><div class="metric-mini-value">{metrics["r2"]:.4f}</div>'
                f'<div class="metric-mini-label">R² SCORE</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-mini"><div class="metric-mini-value">{metrics["mae"]:.2f}</div>'
                f'<div class="metric-mini-label">MAE</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-mini"><div class="metric-mini-value">{metrics["rmse"]:.2f}</div>'
                f'<div class="metric-mini-label">RMSE</div></div>',
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------
# PAGE: DATA INSIGHTS
# ----------------------------------------------------------------------------
def render_insights():
    page_header("📊", "Data Insights", "Explore the underlying dataset in more depth")

    s = dataset_summary(active_df)
    year_range = f"{s['year_min']}–{s['year_max']}" if s["year_min"] else "N/A"
    missing = int(active_df.isnull().sum().sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("📄", f"{s['records']:,}", "TOTAL RECORDS")
    with c2:
        kpi_card("🌿", s["crops"], "TOTAL CROPS")
    with c3:
        kpi_card("🗺️", s["states"], "TOTAL STATES")
    with c4:
        kpi_card("📅", year_range, "YEAR RANGE")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🧩 Missing Values by Column</div>', unsafe_allow_html=True)
        miss_by_col = active_df.isnull().sum()
        miss_by_col = miss_by_col[miss_by_col > 0].sort_values(ascending=False)
        if len(miss_by_col) > 0:
            fig = styled_bar_fig(miss_by_col.index, miss_by_col.values, "v", "#f59e0b")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No missing values in this dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📈 Records by Year</div>', unsafe_allow_html=True)
        if "year" in active_df.columns:
            by_year = active_df["year"].value_counts().sort_index()
            fig = styled_bar_fig(by_year.index.astype(str), by_year.values, "v", "#22c55e")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Dataset has no 'year' column.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📋 Dataset Columns</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(
            {
                "Column": active_df.columns,
                "Dtype": [str(t) for t in active_df.dtypes],
                "Missing": active_df.isnull().sum().values,
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(f"<div style='color:#9ca8a3;font-size:12.5px;margin-top:6px;'>Total missing values across dataset: {missing:,}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PAGE: CROP ANALYSIS
# ----------------------------------------------------------------------------
def render_crop_analysis():
    page_header("🌾", "Crop Analysis", "Drill into yield statistics for a specific crop")

    if "crop" not in active_df.columns or "yield_kg_ha" not in active_df.columns:
        st.info("Dataset missing 'crop' or 'yield_kg_ha' columns.")
        return

    crops = sorted(active_df["crop"].dropna().unique())
    selected_crop = st.selectbox("Select Crop", crops)

    crop_df = active_df[active_df["crop"] == selected_crop].dropna(subset=["yield_kg_ha"])

    if len(crop_df) == 0:
        st.info("No yield records for this crop.")
        return

    avg_y = crop_df["yield_kg_ha"].mean()
    min_y = crop_df["yield_kg_ha"].min()
    max_y = crop_df["yield_kg_ha"].max()
    n_records = len(crop_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="crop-stat-card"><div class="crop-stat-value">{avg_y:,.1f}</div><div class="crop-stat-label">AVG YIELD (kg/ha)</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="crop-stat-card"><div class="crop-stat-value">{min_y:,.1f}</div><div class="crop-stat-label">MIN YIELD (kg/ha)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="crop-stat-card"><div class="crop-stat-value">{max_y:,.1f}</div><div class="crop-stat-label">MAX YIELD (kg/ha)</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="crop-stat-card"><div class="crop-stat-value">{n_records:,}</div><div class="crop-stat-label">RECORDS</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-title">📊 {selected_crop} — Yield Distribution</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_histogram(x=crop_df["yield_kg_ha"], marker_color="#22c55e", nbinsx=30)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cfe0d6", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Yield (kg/ha)", gridcolor="#22302a"),
        yaxis=dict(title="Count", gridcolor="#22302a"),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    if "state" in crop_df.columns:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="chart-title">🗺️ {selected_crop} — Avg Yield by State</div>', unsafe_allow_html=True)
        state_avg = crop_df.groupby("state")["yield_kg_ha"].mean().sort_values(ascending=True)
        fig2 = styled_bar_fig(state_avg.values, state_avg.index, "h", "#3b82f6")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PAGE: ABOUT PROJECT
# ----------------------------------------------------------------------------
def render_about():
    page_header("ℹ️", "About Project", "Learn more about Agri Smart")

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chart-title">🌱 Agri Smart — AI for Agriculture</div>
        <p style="color:#cfe0d6; line-height:1.7; font-size:14.5px;">
        Agri Smart is a crop yield prediction dashboard built on real agricultural data
        spanning Indian states, districts, and crop types from 2020–2024. It combines
        environmental factors (rainfall, temperature, humidity) with soil health
        indicators (nitrogen, phosphorus, potassium, pH) to estimate expected crop yield.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🧠 Model</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <p style="color:#cfe0d6; font-size:14px; line-height:1.7;">
            • Algorithm: <b>Random Forest Regressor</b><br>
            • n_estimators: 300<br>
            • random_state: 42<br>
            • Crop feature encoded with <b>LabelEncoder</b><br>
            • Train/test split: 80% / 20%
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📥 Prediction Inputs</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <p style="color:#cfe0d6; font-size:14px; line-height:1.7;">
            Rainfall (mm), Temperature (°C), Humidity (%), Soil Nitrogen,
            Soil Phosphorus, Soil Potassium, Soil pH, and Crop type.
            District, state, and year are excluded from the model —
            they're used only for dashboard analytics.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------------
PAGES = {
    "Dashboard": render_dashboard,
    "Predict Yield": render_predict,
    "Data Insights": render_insights,
    "Crop Analysis": render_crop_analysis,
    "About Project": render_about,
}

PAGES[st.session_state.page]()
