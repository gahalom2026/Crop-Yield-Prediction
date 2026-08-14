import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Agri Smart | AI for Agriculture",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "FAO_Crop_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_yield_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "crop_encoder.pkl")

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

# ============================================================
# CUSTOM CSS — dark charcoal theme, green sidebar
# ============================================================
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background-color: #0d1117;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f2818;
        border-right: 1px solid #1e3a26;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .sidebar-brand {
        font-size: 1.3rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0;
    }

    .sidebar-sub {
        color: #8fbf9f;
        font-size: 0.85rem;
        margin-top: -8px;
        margin-bottom: 10px;
    }

    .nav-label {
        color: #6fa87f;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-top: 6px;
        margin-bottom: 6px;
    }

    div[data-testid="stSidebar"] .stButton button {
        width: 100%;
        text-align: left;
        background-color: transparent;
        color: #d7e8dd;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 2px;
        transition: background-color 0.15s ease;
    }

    div[data-testid="stSidebar"] .stButton button:hover {
        background-color: #16351f;
        color: #ffffff;
    }

    div[data-testid="stSidebar"] .stButton button:focus {
        box-shadow: none;
    }

    .nav-active button {
        background-color: #22c55e !important;
        color: #06210f !important;
        font-weight: 700 !important;
    }

    .dataset-box {
        background-color: #142a1c;
        border: 1px solid #2a4a34;
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 6px;
    }

    .dataset-ok {
        color: #4ade80;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .dataset-file {
        color: #cfe8d7;
        font-size: 0.85rem;
        margin-top: 2px;
    }

    .dataset-shape {
        color: #7fa88c;
        font-size: 0.8rem;
    }

    .upload-hint {
        color: #6a8a75;
        font-size: 0.78rem;
        margin-top: 6px;
    }

    .sidebar-footer {
        color: #5a7d67;
        font-size: 0.78rem;
        margin-top: 10px;
    }

    /* KPI cards */
    .kpi-card {
        background-color: #12181f;
        border: 1px solid #232b35;
        border-radius: 14px;
        padding: 18px 20px;
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .kpi-icon {
        font-size: 1.6rem;
    }

    .kpi-value {
        color: #4ade80;
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .kpi-label {
        color: #8b98a5;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-top: 2px;
    }

    /* chart card */
    .chart-card {
        background-color: #12181f;
        border: 1px solid #232b35;
        border-radius: 14px;
        padding: 18px 20px;
    }

    .chart-title {
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .info-box {
        background-color: #0f2537;
        border: 1px solid #1c3a52;
        border-radius: 12px;
        padding: 14px 18px;
        display: flex;
        gap: 12px;
        align-items: flex-start;
    }

    .info-title {
        color: #60a5fa;
        font-weight: 700;
        font-size: 0.95rem;
    }

    .info-sub {
        color: #93a3b3;
        font-size: 0.85rem;
        margin-top: 2px;
    }

    .main-title {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .main-sub {
        color: #8b98a5;
        font-size: 1rem;
        margin-top: -6px;
    }

    .result-card {
        background: linear-gradient(135deg, #16351f, #0f2818);
        border: 1px solid #2a4a34;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    .result-label {
        color: #8fbf9f;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    .result-value {
        color: #4ade80;
        font-size: 3rem;
        font-weight: 800;
        margin: 6px 0;
    }

    .metric-card {
        background-color: #12181f;
        border: 1px solid #232b35;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .metric-value {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 800;
    }

    .metric-label {
        color: #8b98a5;
        font-size: 0.78rem;
        margin-top: 2px;
    }

    div.stButton > button[kind="primary"] {
        background-color: #22c55e;
        color: #06210f;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.6rem 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=False)
def load_default_data():
    if os.path.exists(DEFAULT_DATA_PATH):
        return pd.read_csv(DEFAULT_DATA_PATH)
    return None


def load_uploaded_data(file):
    try:
        df = pd.read_csv(file)
        required = {"rainfall_mm", "temperature_C", "humidity_pct", "soil_N_kg_ha",
                    "soil_P_kg_ha", "soil_K_kg_ha", "soil_pH", "crop", "yield_kg_ha"}
        if not required.issubset(set(df.columns)):
            return None
        return df
    except Exception:
        return None


# ============================================================
# MODEL TRAINING (matches notebook exactly)
# ============================================================
@st.cache_resource(show_spinner=True)
def get_model_and_encoder(df: pd.DataFrame):
    # If a saved model/encoder already exist, use them
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        le_crop = joblib.load(ENCODER_PATH)
        metrics = None
        return model, le_crop, metrics

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
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    try:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(le_crop, ENCODER_PATH)
    except Exception:
        pass

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    return model, le_crop, metrics


# ============================================================
# SESSION STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "custom_df" not in st.session_state:
    st.session_state.custom_df = None
if "custom_filename" not in st.session_state:
    st.session_state.custom_filename = None
if "upload_error" not in st.session_state:
    st.session_state.upload_error = None

default_df = load_default_data()

if st.session_state.custom_df is not None:
    active_df = st.session_state.custom_df
    active_source = st.session_state.custom_filename
    using_custom = True
else:
    active_df = default_df
    active_source = "FAO_Crop_data.csv"
    using_custom = False

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<p class="sidebar-brand">🌱 Agri Smart</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">AI for Agriculture</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="nav-label">NAVIGATION</p>', unsafe_allow_html=True)

    nav_items = [
        ("Dashboard", "🏠 Dashboard"),
        ("Predict Yield", "🌱 Predict Yield"),
        ("Data Insights", "📊 Data Insights"),
        ("Crop Analysis", "🌾 Crop Analysis"),
        ("About Project", "ℹ️ About Project"),
    ]

    for key, label in nav_items:
        is_active = st.session_state.page == key
        wrapper_class = "nav-active" if is_active else "nav-inactive"
        st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="nav-label">DATASET</p>', unsafe_allow_html=True)

    if using_custom:
        st.markdown(f"""
        <div class="dataset-box">
            <div class="dataset-ok">✓ Custom Dataset Loaded</div>
            <div class="dataset-file">{active_source}</div>
            <div class="dataset-shape">{active_df.shape[0]:,} rows × {active_df.shape[1]} columns</div>
        </div>
        """, unsafe_allow_html=True)
    elif active_df is not None:
        st.markdown(f"""
        <div class="dataset-box">
            <div class="dataset-ok">✓ Default Dataset Loaded</div>
            <div class="dataset-file">FAO_Crop_data.csv</div>
            <div class="dataset-shape">{active_df.shape[0]:,} rows × {active_df.shape[1]} columns</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Default dataset not found. Please upload a CSV to continue.")

    if st.session_state.upload_error:
        st.warning(st.session_state.upload_error)

    with st.expander("📤 Upload New Dataset"):
        uploaded_file = st.file_uploader("Upload FAO_Crop_data.csv", type=["csv"], label_visibility="collapsed")
        if uploaded_file is not None:
            new_df = load_uploaded_data(uploaded_file)
            if new_df is not None:
                st.session_state.custom_df = new_df
                st.session_state.custom_filename = uploaded_file.name
                st.session_state.upload_error = None
                st.rerun()
            else:
                st.session_state.upload_error = "⚠️ Invalid dataset format. Continuing with the default dataset."

    st.markdown('<p class="upload-hint">Upload only if you want to try<br>your own data.</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p class="sidebar-footer">© 2024 Agri Smart<br>All rights reserved.</p>
    """, unsafe_allow_html=True)


# ============================================================
# GUARD: no data available at all
# ============================================================
if active_df is None:
    st.markdown('<p class="main-title">🌱 Agri Smart</p>', unsafe_allow_html=True)
    st.error(
        "No dataset found. Please place **FAO_Crop_data.csv** next to app.py, "
        "or upload a CSV from the sidebar."
    )
    st.stop()

df = active_df.copy()

# ============================================================
# DASHBOARD PAGE
# ============================================================
def render_dashboard(df, using_custom, active_source):
    st.markdown('<p class="main-title">🌱 Crop Yield Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-sub">Predict crop yield based on environmental, soil and crop conditions</p>', unsafe_allow_html=True)
    st.write("")

    total_records = len(df)
    total_crops = df["crop"].nunique() if "crop" in df.columns else 0
    total_states = df["state"].nunique() if "state" in df.columns else 0
    if "year" in df.columns and df["year"].notna().any():
        year_range = f"{int(df['year'].min())}–{int(df['year'].max())}"
    else:
        year_range = "N/A"

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, "📄", f"{total_records:,}", "TOTAL RECORDS"),
        (c2, "🌿", f"{total_crops}", "TOTAL CROPS"),
        (c3, "🗺️", f"{total_states}", "TOTAL STATES"),
        (c4, "📅", year_range, "YEAR RANGE"),
    ]
    for col, icon, value, label in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-label">{label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">🌾 Average Yield by Crop</p>', unsafe_allow_html=True)
        if "crop" in df.columns and "yield_kg_ha" in df.columns:
            crop_avg = (
                df.dropna(subset=["yield_kg_ha"])
                .groupby("crop")["yield_kg_ha"].mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            fig = px.bar(crop_avg, x="crop", y="yield_kg_ha", color_discrete_sequence=["#22c55e"])
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d4dd",
                xaxis_title="Crop",
                yaxis_title="Avg Yield (kg/ha)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=380,
            )
            fig.update_xaxes(gridcolor="#1f2733")
            fig.update_yaxes(gridcolor="#1f2733")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Dataset missing 'crop' or 'yield_kg_ha' columns.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">🗺️ Average Yield by State</p>', unsafe_allow_html=True)
        if "state" in df.columns and "yield_kg_ha" in df.columns:
            state_avg = (
                df.dropna(subset=["yield_kg_ha"])
                .groupby("state")["yield_kg_ha"].mean()
                .sort_values(ascending=True)
                .reset_index()
            )
            fig2 = px.bar(state_avg, x="yield_kg_ha", y="state", orientation="h",
                          color_discrete_sequence=["#3b82f6"])
            fig2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d4dd",
                xaxis_title="Avg Yield (kg/ha)",
                yaxis_title="State",
                margin=dict(l=10, r=10, t=10, b=10),
                height=380,
            )
            fig2.update_xaxes(gridcolor="#1f2733")
            fig2.update_yaxes(gridcolor="#1f2733")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Dataset missing 'state' or 'yield_kg_ha' columns.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    if using_custom:
        st.markdown(f"""
        <div class="info-box">
            <div>ℹ️</div>
            <div>
                <div class="info-title">Using custom dataset: {active_source}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <div>ℹ️</div>
            <div>
                <div class="info-title">Using default dataset: FAO_Crop_data.csv</div>
                <div class="info-sub">You can upload your own dataset from the sidebar if you want to try custom data.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PREDICT YIELD PAGE
# ============================================================
def render_predict(df):
    st.markdown('<p class="main-title">🌱 Predict Yield</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-sub">Enter environmental, soil and crop conditions to predict yield</p>', unsafe_allow_html=True)
    st.write("")

    model, le_crop, metrics = get_model_and_encoder(df)

    if metrics:
        m1, m2, m3 = st.columns(3)
        for col, label, value in [
            (m1, "R²", f"{metrics['r2']:.4f}"),
            (m2, "MAE", f"{metrics['mae']:.1f}"),
            (m3, "RMSE", f"{metrics['rmse']:.1f}"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        st.write("")

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=1000.0, step=10.0)
        temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=55.0, value=25.0, step=0.5)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
        soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    with col2:
        soil_n = st.number_input("Soil Nitrogen (N)", min_value=0.0, max_value=500.0, value=80.0, step=1.0)
        soil_p = st.number_input("Soil Phosphorus (P)", min_value=0.0, max_value=500.0, value=30.0, step=1.0)
        soil_k = st.number_input("Soil Potassium (K)", min_value=0.0, max_value=500.0, value=45.0, step=1.0)
        crop = st.selectbox("Crop", options=list(le_crop.classes_))

    predict_clicked = st.button("🔮 Predict Yield", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        crop_enc = le_crop.transform([crop])[0]
        features = pd.DataFrame(
            [[rainfall, temperature, humidity, soil_n, soil_p, soil_k, soil_ph, crop_enc]],
            columns=FEATURE_ORDER,
        )
        prediction = model.predict(features)[0]

        st.write("")
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">PREDICTED CROP YIELD</div>
            <div class="result-value">{prediction:,.1f}</div>
            <div class="result-label">kg/ha</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# DATA INSIGHTS PAGE
# ============================================================
def render_insights(df):
    st.markdown('<p class="main-title">📊 Data Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-sub">Explore the dataset behind the predictions</p>', unsafe_allow_html=True)
    st.write("")

    total_records = len(df)
    total_crops = df["crop"].nunique() if "crop" in df.columns else 0
    total_states = df["state"].nunique() if "state" in df.columns else 0
    if "year" in df.columns and df["year"].notna().any():
        year_range = f"{int(df['year'].min())}–{int(df['year'].max())}"
    else:
        year_range = "N/A"
    missing_values = int(df.isnull().sum().sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "📄", f"{total_records:,}", "TOTAL RECORDS"),
        (c2, "🌿", f"{total_crops}", "TOTAL CROPS"),
        (c3, "🗺️", f"{total_states}", "TOTAL STATES"),
        (c4, "📅", year_range, "YEAR RANGE"),
        (c5, "⚠️", f"{missing_values:,}", "MISSING VALUES"),
    ]
    for col, icon, value, label in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div>
                    <div class="kpi-value" style="font-size:1.3rem;">{value}</div>
                    <div class="kpi-label">{label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="chart-title">📋 Dataset Columns</p>', unsafe_allow_html=True)
    st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">📈 Records by Year</p>', unsafe_allow_html=True)
        if "year" in df.columns:
            year_counts = df["year"].value_counts().sort_index().reset_index()
            year_counts.columns = ["year", "count"]
            fig = px.bar(year_counts, x="year", y="count", color_discrete_sequence=["#22c55e"])
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               font_color="#c9d4dd", height=340, margin=dict(l=10, r=10, t=10, b=10))
            fig.update_xaxes(gridcolor="#1f2733")
            fig.update_yaxes(gridcolor="#1f2733")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">🌾 Records by Crop</p>', unsafe_allow_html=True)
        if "crop" in df.columns:
            crop_counts = df["crop"].value_counts().reset_index()
            crop_counts.columns = ["crop", "count"]
            fig2 = px.pie(crop_counts, names="crop", values="count", hole=0.5,
                          color_discrete_sequence=px.colors.sequential.Greens_r)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d4dd",
                                height=340, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# CROP ANALYSIS PAGE
# ============================================================
def render_crop_analysis(df):
    st.markdown('<p class="main-title">🌾 Crop Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-sub">Deep dive into a specific crop\'s yield performance</p>', unsafe_allow_html=True)
    st.write("")

    if "crop" not in df.columns or "yield_kg_ha" not in df.columns:
        st.info("Dataset missing 'crop' or 'yield_kg_ha' columns.")
        return

    crop_options = sorted(df["crop"].dropna().unique().tolist())
    selected_crop = st.selectbox("Select a crop", crop_options)

    crop_df = df[df["crop"] == selected_crop].dropna(subset=["yield_kg_ha"])

    avg_yield = crop_df["yield_kg_ha"].mean()
    min_yield = crop_df["yield_kg_ha"].min()
    max_yield = crop_df["yield_kg_ha"].max()
    records = len(crop_df)

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, "📊", f"{avg_yield:,.1f}", "AVG YIELD (kg/ha)"),
        (c2, "🔻", f"{min_yield:,.1f}", "MIN YIELD (kg/ha)"),
        (c3, "🔺", f"{max_yield:,.1f}", "MAX YIELD (kg/ha)"),
        (c4, "📄", f"{records:,}", "RECORDS"),
    ]
    for col, icon, value, label in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div>
                    <div class="kpi-value" style="font-size:1.4rem;">{value}</div>
                    <div class="kpi-label">{label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(f'<p class="chart-title">📈 {selected_crop} — Yield Distribution</p>', unsafe_allow_html=True)
    fig = px.histogram(crop_df, x="yield_kg_ha", nbins=30, color_discrete_sequence=["#22c55e"])
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       font_color="#c9d4dd", height=380, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="Yield (kg/ha)", yaxis_title="Count")
    fig.update_xaxes(gridcolor="#1f2733")
    fig.update_yaxes(gridcolor="#1f2733")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    if "state" in df.columns:
        st.write("")
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(f'<p class="chart-title">🗺️ {selected_crop} — Avg Yield by State</p>', unsafe_allow_html=True)
        state_avg = crop_df.groupby("state")["yield_kg_ha"].mean().sort_values().reset_index()
        fig2 = px.bar(state_avg, x="yield_kg_ha", y="state", orientation="h",
                      color_discrete_sequence=["#3b82f6"])
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#c9d4dd", height=420, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Avg Yield (kg/ha)", yaxis_title="State")
        fig2.update_xaxes(gridcolor="#1f2733")
        fig2.update_yaxes(gridcolor="#1f2733")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================
def render_about():
    st.markdown('<p class="main-title">ℹ️ About Project</p>', unsafe_allow_html=True)
    st.write("")

    st.markdown("""
    <div class="chart-card">
        <p class="chart-title">🌱 Agri Smart — AI for Agriculture</p>
        <p style="color:#c9d4dd; line-height:1.7;">
        Agri Smart is a crop yield prediction dashboard that combines environmental,
        soil and crop data to help estimate expected agricultural yield. The application
        loads the <b>FAO_Crop_data.csv</b> dataset by default, covering rainfall, temperature,
        humidity, soil nutrients (N, P, K), soil pH and yield records across multiple
        Indian states and crops.
        </p>
        <p style="color:#c9d4dd; line-height:1.7;">
        The prediction engine is a <b>Random Forest Regressor</b> (n_estimators=300,
        random_state=42) trained on rainfall, temperature, humidity, soil nutrients,
        soil pH and crop type (label-encoded). Records with missing values are dropped
        before training, and the district/state/year columns are excluded from the
        model since they are not used as predictive features.
        </p>
        <p style="color:#c9d4dd; line-height:1.7;">
        The dashboard, data insights and crop analysis pages use the full dataset
        (including state and year) purely for exploratory analytics, while the
        Predict Yield page uses only the exact feature set the model was trained on.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ROUTER
# ============================================================
page = st.session_state.page

if page == "Dashboard":
    render_dashboard(df, using_custom, active_source)
elif page == "Predict Yield":
    render_predict(df)
elif page == "Data Insights":
    render_insights(df)
elif page == "Crop Analysis":
    render_crop_analysis(df)
elif page == "About Project":
    render_about()
