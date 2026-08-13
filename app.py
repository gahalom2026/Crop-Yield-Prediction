"""
Crop Yield Prediction Dashboard
--------------------------------
A professional Streamlit application that trains a regression pipeline on
FAO_Crop_data.csv and lets a user predict crop yield (yield_kg_ha) from
soil, weather, crop and location/year features.

Run with:
    streamlit run app.py

Dataset columns used (EXACTLY as in FAO_Crop_data.csv, nothing invented):
    year, state, rainfall_mm, temperature_C, humidity_pct,
    soil_N_kg_ha, soil_P_kg_ha, soil_K_kg_ha, soil_pH, crop, yield_kg_ha

NOTE: There is intentionally NO District field anywhere in this app —
the dataset does not contain a district column, so nothing about
location below "state" is invented, shown, or assumed.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CONSTANTS — dataset schema (must match FAO_Crop_data.csv)
# ============================================================
NUMERIC_FEATURES = [
    "year", "rainfall_mm", "temperature_C", "humidity_pct",
    "soil_N_kg_ha", "soil_P_kg_ha", "soil_K_kg_ha", "soil_pH",
]
CATEGORICAL_FEATURES = ["state", "crop"]  # no "district" — not in the dataset
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "yield_kg_ha"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

CANDIDATE_DATA_PATHS = [
    "FAO_Crop_data.csv",
    "data/FAO_Crop_data.csv",
    os.path.join(os.path.dirname(__file__), "FAO_Crop_data.csv"),
    os.path.join(os.path.dirname(__file__), "data", "FAO_Crop_data.csv"),
]

# Sensible / physically-possible hard limits used for input validation.
# These are intentionally NOT shown on screen (per design brief) but are
# enforced both on the widgets themselves (min/max) and again before the
# model is called, so an impossible value can never reach the pipeline.
NUMERIC_INPUT_LIMITS = {
    "temperature_C": {"label": "Temperature", "min": -10.0, "max": 60.0, "step": 0.1},
    "rainfall_mm": {"label": "Rainfall", "min": 0.0, "max": 5000.0, "step": 1.0},
    "humidity_pct": {"label": "Humidity", "min": 0.0, "max": 100.0, "step": 0.1},
    "soil_N_kg_ha": {"label": "Soil Nitrogen", "min": 0.0, "max": 500.0, "step": 0.1},
    "soil_P_kg_ha": {"label": "Soil Phosphorus", "min": 0.0, "max": 500.0, "step": 0.1},
    "soil_K_kg_ha": {"label": "Soil Potassium", "min": 0.0, "max": 500.0, "step": 0.1},
    "soil_pH": {"label": "Soil pH", "min": 0.0, "max": 14.0, "step": 0.1},
}

STATE_PLACEHOLDER = "Select state"
YEAR_PLACEHOLDER = "Select year"
CROP_PLACEHOLDER = "Select crop"

# ============================================================
# PAGE CONFIG + CUSTOM CSS (green agri theme)
# ============================================================
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background-color: #0b3d24;
    }
    section[data-testid="stSidebar"] * {
        color: #eafaf0 !important;
    }
    .sidebar-note {
        background-color: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 14px 16px;
        font-size: 13px;
        line-height: 1.5;
        margin-top: 8px;
    }

    /* White "card" look for bordered containers (Streamlit's st.container(border=True)) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 16px !important;
        border: 1px solid #e3ede6 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e3ede6;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        text-align: center;
    }
    .metric-card .big {
        font-size: 30px;
        font-weight: 800;
        color: #1e7c3f;
    }
    .metric-card .label {
        font-size: 13px;
        color: #5a6b60;
        text-transform: uppercase;
        letter-spacing: .04em;
    }

    /* Estimated-yield hero panel */
    .yield-hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #f4faf6;
        border: 1px solid #dcefe1;
        border-radius: 16px;
        padding: 24px 28px;
        gap: 20px;
        flex-wrap: wrap;
    }
    .yield-hero .label {
        font-size: 13px;
        color: #5a6b60;
        text-transform: uppercase;
        letter-spacing: .04em;
        margin-bottom: 4px;
    }
    .yield-hero .value {
        font-size: 42px;
        font-weight: 800;
        color: #14532d;
        line-height: 1.1;
    }
    .yield-hero .value .unit {
        font-size: 18px;
        font-weight: 600;
        color: #3f5b48;
    }
    .yield-hero .tons {
        font-size: 13px;
        color: #5a6b60;
        margin-top: 4px;
    }
    .yield-badge {
        width: 108px;
        height: 108px;
        min-width: 108px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .yield-badge .icon { font-size: 20px; margin-bottom: 2px; }
    .yield-badge .cat { font-weight: 800; font-size: 14px; line-height: 1.2; }
    .yield-badge .sub { font-weight: 700; font-size: 11px; letter-spacing: .04em; }

    /* Small stat cards under the hero panel */
    .stat-card {
        background-color: #ffffff;
        border: 1px solid #e3ede6;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        height: 100%;
    }
    .stat-card .title {
        font-size: 13px;
        font-weight: 600;
        color: #3f5b48;
        margin-bottom: 8px;
    }
    .stat-card .value {
        font-size: 24px;
        font-weight: 800;
        color: #163d27;
    }
    .stat-card .delta {
        font-size: 13px;
        font-weight: 700;
        margin-top: 6px;
    }
    .stat-card .delta.up { color: #1e7c3f; }
    .stat-card .delta.down { color: #b91c1c; }
    .stat-card .caption {
        font-size: 12px;
        color: #5a6b60;
        margin-top: 2px;
    }

    /* Data-driven observation boxes */
    .obs-box {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .obs-box .obs-icon {
        width: 26px;
        height: 26px;
        min-width: 26px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        color: #ffffff;
    }
    .obs-box.positive { background-color: #f0faf3; border: 1px solid #cdeedb; }
    .obs-box.positive .obs-icon { background-color: #1e7c3f; }
    .obs-box.positive .obs-text { color: #163d27 !important; }
    .obs-box.negative { background-color: #fdf1f1; border: 1px solid #f6cfcf; }
    .obs-box.negative .obs-icon { background-color: #b91c1c; }
    .obs-box.negative .obs-text { color: #5c1414 !important; }
    .obs-box.neutral { background-color: #fff9ec; border: 1px solid #f3e3b8; }
    .obs-box.neutral .obs-icon { background-color: #b45309; }
    .obs-box.neutral .obs-text { color: #4a3105 !important; }
    .obs-box .obs-text, .obs-box .obs-text * { color: inherit !important; }

    .hint-note {
        background-color: #fff9ec;
        border: 1px solid #f3e3b8;
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 13px;
        color: #4a3105;
        margin-top: 14px;
    }
    .info-note {
        background-color: #eef4ff;
        border: 1px solid #cddcfa;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 13px;
        color: #1f2f52;
        margin-top: 6px;
    }
    .info-note b { color: #142147; }

    h1, h2, h3 { color: #163d27; }

    /* Hide Streamlit's built-in "Press Enter to submit form" / "Press Enter to apply"
       instruction tooltip — it overlaps the placeholder text inside number inputs. */
    div[data-testid="InputInstructions"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING + CLEANING
# ============================================================
@st.cache_data(show_spinner="Loading dataset...")
def load_data(uploaded_file=None):
    """Load FAO_Crop_data.csv from disk, or from an uploaded file."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df

    for path in CANDIDATE_DATA_PATHS:
        if os.path.exists(path):
            return pd.read_csv(path)

    return None


def validate_schema(df: pd.DataFrame):
    """Make sure every column the app relies on actually exists."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a missing target (can't train/evaluate on those).
    Feature-level missing values are handled inside the ML pipeline."""
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=[TARGET_COLUMN])
    return df_clean


# ============================================================
# MODEL TRAINING
# ============================================================
def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return preprocessor


@st.cache_resource(show_spinner="Training models (this runs once)...")
def train_model(df_hash: str, df: pd.DataFrame):
    """Train several regressors, evaluate them, and return the best pipeline
    plus a comparison table and feature importance (only real, not invented)."""
    df_clean = clean_data(df)
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=12),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = []
    fitted_pipelines = {}

    for name, regressor in candidate_models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2 Score": r2})
        fitted_pipelines[name] = pipe

    comparison_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False).reset_index(drop=True)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_pipeline = fitted_pipelines[best_model_name]

    return best_pipeline, best_model_name, comparison_df


def get_feature_importance(pipeline, model_name):
    """Return (DataFrame, label) of real importances/coefficients from the
    trained model, or None if the model type doesn't expose any."""
    regressor = pipeline.named_steps["regressor"]
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    if hasattr(regressor, "feature_importances_"):
        values = regressor.feature_importances_
        label = "Importance"
    elif hasattr(regressor, "coef_"):
        values = np.abs(regressor.coef_)
        label = "Coefficient Magnitude"
    else:
        return None

    fi_df = pd.DataFrame({"Feature": feature_names, label: values})
    fi_df = fi_df.sort_values(label, ascending=False).head(15)
    return fi_df, label


def make_prediction(pipeline, input_dict: dict) -> float:
    input_df = pd.DataFrame([input_dict], columns=FEATURE_COLUMNS)
    prediction = pipeline.predict(input_df)[0]
    return float(prediction)


def classify_yield(pred_kg: float, target_series: pd.Series) -> str:
    """Classify a prediction as Low / Medium / High yield using the
    tertiles (33rd / 66th percentile) of the actual training-data
    distribution — dynamic, not a hard-coded threshold."""
    low_cut = target_series.quantile(1 / 3)
    high_cut = target_series.quantile(2 / 3)
    if pred_kg < low_cut:
        return "LOW"
    elif pred_kg > high_cut:
        return "HIGH"
    return "MEDIUM"


def validate_numeric_inputs(values: dict) -> list:
    """Internal validation against sensible/physical limits. Returns a list
    of human-readable error strings (empty list = all good)."""
    errors = []
    for key, cfg in NUMERIC_INPUT_LIMITS.items():
        val = values.get(key)
        if val is None:
            errors.append(f"{cfg['label']} is required.")
        elif val < cfg["min"] or val > cfg["max"]:
            errors.append(f"{cfg['label']} must be between {cfg['min']:g} and {cfg['max']:g}.")
    return errors


# ============================================================
# LOAD DATA (with graceful error handling)
# ============================================================
st.sidebar.markdown("## 🌱 Agri Smart\n**AI for Agriculture**")
st.sidebar.markdown("---")

df_raw = load_data()

if df_raw is None:
    uploaded = st.file_uploader("Upload FAO_Crop_data.csv", type=["csv"])
    if uploaded is None:
        st.stop()
    df_raw = load_data(uploaded_file=uploaded)

missing_cols = validate_schema(df_raw)
if missing_cols:
    st.error(
        "❌ The uploaded CSV does not match the expected schema. "
        f"The following columns are missing: {missing_cols}"
    )
    st.stop()

if df_raw.empty:
    st.error("❌ The dataset is empty — please upload a CSV with valid data.")
    st.stop()

df = df_raw.copy()
df_hash = str(pd.util.hash_pandas_object(df).sum())

try:
    best_pipeline, best_model_name, comparison_df = train_model(df_hash, df)
    fi_result = get_feature_importance(best_pipeline, best_model_name)
except Exception as e:
    st.error(f"❌ An error occurred while training the model: {e}")
    st.stop()

df_clean = clean_data(df)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
page = st.sidebar.radio(
    "NAVIGATION",
    ["🏠 Dashboard", "🌱 Predict Yield", "📊 Data Insights", "🌾 Crop Analysis", "ℹ️ About Project"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### DATASET OVERVIEW")
st.sidebar.write(f"💧 **Total Records**  \n{len(df):,}")
st.sidebar.write(f"🌾 **Total Crops**  \n{df['crop'].nunique()}")
st.sidebar.write(f"🗺️ **States**  \n{df['state'].nunique()}")
st.sidebar.write(f"📅 **Years**  \n{int(df['year'].min())} – {int(df['year'].max())}")
st.sidebar.markdown("---")

records_rounded = (len(df) // 1000) * 1000
st.sidebar.markdown(
    f'<div class="sidebar-note">🌱 This model is trained on {records_rounded:,}+ records '
    f'from Indian states with soil, weather and crop information.</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption(
    f"Best performing model: **{best_model_name}** "
    f"(R² = {comparison_df.iloc[0]['R2 Score']:.3f})"
)


# ============================================================
# PAGE 1: DASHBOARD
# ============================================================
if page == "🏠 Dashboard":
    st.markdown("# 🌱 Crop Yield Prediction")
    st.caption("Predict crop yield based on environmental, soil and crop conditions")

    c1, c2, c3, c4 = st.columns(4)
    for col, (label, value) in zip(
        [c1, c2, c3, c4],
        [
            ("Total Records", f"{len(df):,}"),
            ("Total Crops", f"{df['crop'].nunique()}"),
            ("Total States", f"{df['state'].nunique()}"),
            ("Year Range", f"{int(df['year'].min())}–{int(df['year'].max())}"),
        ],
    ):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="big">{value}</div>'
                f'<div class="label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Average Yield by Crop")
        avg_crop = df_clean.groupby("crop")[TARGET_COLUMN].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(avg_crop, x="crop", y=TARGET_COLUMN, color_discrete_sequence=["#2e8b57"])
        fig.update_layout(xaxis_title="Crop", yaxis_title="Avg Yield (kg/ha)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Average Yield by State")
        avg_state = df_clean.groupby("state")[TARGET_COLUMN].mean().sort_values().reset_index()
        fig = px.bar(avg_state, x=TARGET_COLUMN, y="state", orientation="h", color_discrete_sequence=["#3b82c4"])
        fig.update_layout(xaxis_title="Avg Yield (kg/ha)", yaxis_title="State", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Historical Yield Trend")
        crop_for_trend = st.selectbox("Select crop", sorted(df_clean["crop"].unique()), key="trend_crop")
        trend = (
            df_clean[df_clean["crop"] == crop_for_trend]
            .groupby("year")[TARGET_COLUMN].mean().reset_index()
        )
        fig = px.line(trend, x="year", y=TARGET_COLUMN, markers=True, color_discrete_sequence=["#7b2fd6"])
        fig.update_layout(xaxis_title="Year", yaxis_title="Avg Yield (kg/ha)")
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Feature Importance")
        if fi_result is not None:
            fi_df, label = fi_result
            fig = px.bar(
                fi_df.sort_values(label), x=label, y="Feature", orientation="h",
                color_discrete_sequence=["#e08b1d"],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Extracted directly from the trained **{best_model_name}** model.")
        else:
            st.info(f"{best_model_name} doesn't expose feature importances.")

    st.subheader("Data Quality Summary")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Missing Values (total)", int(df.isnull().sum().sum()))
    q2.metric("Duplicate Records", int(df.duplicated().sum()))
    q3.metric("Unique Crops", df["crop"].nunique())
    q4.metric("Unique States", df["state"].nunique())


# ============================================================
# PAGE 2: PREDICT YIELD
# ============================================================
elif page == "🌱 Predict Yield":
    st.caption("Fill in field conditions to get a predicted yield")

    left, right = st.columns([1, 1])

    # ---------------- LEFT: INPUT PARAMETERS ----------------
    with left:
        with st.container(border=True):
            st.markdown("### 📋 Input Parameters")

            with st.form("predict_form"):
                r1c1, r1c2 = st.columns(2)
                with r1c1:
                    state = st.selectbox(
                        "State", [STATE_PLACEHOLDER] + sorted(df["state"].dropna().unique()), index=0,
                    )
                with r1c2:
                    year_choices = sorted(set(df["year"].dropna().unique().tolist() + [int(df["year"].max()) + 1]))
                    year = st.selectbox("Year", [YEAR_PLACEHOLDER] + year_choices, index=0)

                r2c1, r2c2 = st.columns(2)
                with r2c1:
                    crop = st.selectbox(
                        "Crop", [CROP_PLACEHOLDER] + sorted(df["crop"].dropna().unique()), index=0,
                    )
                with r2c2:
                    temperature = st.number_input(
                        "Temperature (°C)", value=None, placeholder="Enter temperature",
                        min_value=NUMERIC_INPUT_LIMITS["temperature_C"]["min"],
                        max_value=NUMERIC_INPUT_LIMITS["temperature_C"]["max"],
                        step=NUMERIC_INPUT_LIMITS["temperature_C"]["step"],
                    )

                r3c1, r3c2 = st.columns(2)
                with r3c1:
                    rainfall = st.number_input(
                        "Rainfall (mm)", value=None, placeholder="Enter rainfall",
                        min_value=NUMERIC_INPUT_LIMITS["rainfall_mm"]["min"],
                        max_value=NUMERIC_INPUT_LIMITS["rainfall_mm"]["max"],
                        step=NUMERIC_INPUT_LIMITS["rainfall_mm"]["step"],
                    )
                with r3c2:
                    humidity = st.number_input(
                        "Humidity (%)", value=None, placeholder="Enter humidity",
                        min_value=NUMERIC_INPUT_LIMITS["humidity_pct"]["min"],
                        max_value=NUMERIC_INPUT_LIMITS["humidity_pct"]["max"],
                        step=NUMERIC_INPUT_LIMITS["humidity_pct"]["step"],
                    )

                r4c1, r4c2 = st.columns(2)
                with r4c1:
                    n = st.number_input(
                        "Soil Nitrogen (kg/ha)", value=None, placeholder="Enter nitrogen",
                        min_value=NUMERIC_INPUT_LIMITS["soil_N_kg_ha"]["min"],
                        max_value=NUMERIC_INPUT_LIMITS["soil_N_kg_ha"]["max"],
                        step=NUMERIC_INPUT_LIMITS["soil_N_kg_ha"]["step"],
                    )
                with r4c2:
                    p = st.number_input(
                        "Soil Phosphorus (kg/ha)", value=None, placeholder="Enter phosphorus",
                        min_value=NUMERIC_INPUT_LIMITS["soil_P_kg_ha"]["min"],
                        max_value=NUMERIC_INPUT_LIMITS["soil_P_kg_ha"]["max"],
                        step=NUMERIC_INPUT_LIMITS["soil_P_kg_ha"]["step"],
                    )

                r5c1, r5c2 = st.columns(2)
                with r5c1:
                    k = st.number_input(
                        "Soil Potassium (kg/ha)", value=None, placeholder="Enter potassium",
                        min_value=NUMERIC_INPUT_LIMITS["soil_K_kg_ha"]["min"],
                        max_value=NUMERIC_INPUT_LIMITS["soil_K_kg_ha"]["max"],
                        step=NUMERIC_INPUT_LIMITS["soil_K_kg_ha"]["step"],
                    )
                with r5c2:
                    ph = st.number_input(
                        "Soil pH", value=None, placeholder="Enter soil pH",
                        min_value=NUMERIC_INPUT_LIMITS["soil_pH"]["min"],
                        max_value=NUMERIC_INPUT_LIMITS["soil_pH"]["max"],
                        step=NUMERIC_INPUT_LIMITS["soil_pH"]["step"],
                    )

                submitted = st.form_submit_button("🌱 Predict Yield", use_container_width=True)

            st.markdown(
                '<div class="hint-note">ℹ️ Note: Please fill all fields correctly to get an accurate prediction.</div>',
                unsafe_allow_html=True,
            )

    # ---------------- RIGHT: PREDICTED YIELD ----------------
    with right:
        with st.container(border=True):
            st.markdown("### 🌱 Predicted Yield")

            if not submitted:
                st.info("Fill in the fields and click **Predict Yield** to see the result here.")
            else:
                # Validate dropdowns
                errors = []
                if state == STATE_PLACEHOLDER:
                    errors.append("Please select a State.")
                if year == YEAR_PLACEHOLDER:
                    errors.append("Please select a Year.")
                if crop == CROP_PLACEHOLDER:
                    errors.append("Please select a Crop.")

                # Validate numeric fields (required + sensible ranges)
                numeric_values = {
                    "temperature_C": temperature, "rainfall_mm": rainfall, "humidity_pct": humidity,
                    "soil_N_kg_ha": n, "soil_P_kg_ha": p, "soil_K_kg_ha": k, "soil_pH": ph,
                }
                errors.extend(validate_numeric_inputs(numeric_values))

                if errors:
                    for err in errors:
                        st.error(f"❌ {err}")
                else:
                    input_dict = {
                        "year": year, "rainfall_mm": rainfall, "temperature_C": temperature,
                        "humidity_pct": humidity, "soil_N_kg_ha": n, "soil_P_kg_ha": p,
                        "soil_K_kg_ha": k, "soil_pH": ph, "state": state, "crop": crop,
                    }
                    try:
                        pred_kg = make_prediction(best_pipeline, input_dict)
                    except Exception as e:
                        st.error(f"❌ An error occurred while making the prediction: {e}")
                        st.stop()

                    pred_tons = pred_kg / 1000
                    category = classify_yield(pred_kg, df_clean[TARGET_COLUMN])
                    badge_colors = {
                        "HIGH": ("#e9f7ee", "#1e7c3f"),
                        "MEDIUM": ("#fff8e6", "#b45309"),
                        "LOW": ("#fdecec", "#b91c1c"),
                    }
                    badge_bg, badge_fg = badge_colors[category]

                    st.markdown(
                        f'''
                        <div class="yield-hero">
                            <div>
                                <div class="label">Estimated Yield</div>
                                <div class="value">{pred_kg:,.0f} <span class="unit">kg/ha</span></div>
                                <div class="tons">≈ {pred_tons:,.2f} TONS/HECTARE</div>
                            </div>
                            <div class="yield-badge" style="background-color:{badge_bg};">
                                <div class="icon">🌱</div>
                                <div class="cat" style="color:{badge_fg};">{category}</div>
                                <div class="sub" style="color:{badge_fg};">YIELD</div>
                            </div>
                        </div>
                        ''',
                        unsafe_allow_html=True,
                    )

                    st.write("")

                    crop_avg = df_clean[df_clean["crop"] == crop][TARGET_COLUMN].mean()
                    state_avg = df_clean[df_clean["state"] == state][TARGET_COLUMN].mean()
                    delta_crop = pred_kg - crop_avg
                    delta_state = pred_kg - state_avg

                    def stat_card(title, value, delta, caption):
                        arrow = "↑" if delta >= 0 else "↓"
                        css_cls = "up" if delta >= 0 else "down"
                        return f'''
                        <div class="stat-card">
                            <div class="title">{title}</div>
                            <div class="value">{value:,.0f} kg/ha</div>
                            <div class="delta {css_cls}">{arrow} {delta:,.0f} kg/ha</div>
                            <div class="caption">{caption}</div>
                        </div>
                        '''

                    s1, s2, s3 = st.columns(3)
                    with s1:
                        st.markdown(
                            stat_card(f"🌾 Avg Yield — {crop}", crop_avg, delta_crop, "vs predicted"),
                            unsafe_allow_html=True,
                        )
                    with s2:
                        st.markdown(
                            stat_card(f"🌿 Avg Yield — {state}", state_avg, delta_state, "vs predicted"),
                            unsafe_allow_html=True,
                        )
                    with s3:
                        diff_caption = "Higher than state avg" if delta_state >= 0 else "Lower than state avg"
                        arrow = "↑" if delta_state >= 0 else "↓"
                        css_cls = "up" if delta_state >= 0 else "down"
                        st.markdown(
                            f'''
                            <div class="stat-card">
                                <div class="title">📊 Difference ({state} Avg)</div>
                                <div class="value">{delta_state:,.0f} kg/ha</div>
                                <div class="delta {css_cls}">{arrow} {diff_caption}</div>
                            </div>
                            ''',
                            unsafe_allow_html=True,
                        )

                    st.write("")
                    st.markdown("#### 📌 Data-driven Observations")

                    crop_higher = pred_kg >= crop_avg
                    state_higher = pred_kg >= state_avg

                    st.markdown(
                        f'''<div class="obs-box {"positive" if crop_higher else "negative"}">
                            <div class="obs-icon">{"↑" if crop_higher else "↓"}</div>
                            <div class="obs-text">Predicted yield is <b>{"higher" if crop_higher else "lower"}</b>
                            than the historical average for <b>{crop}</b> ({crop_avg:,.0f} kg/ha).</div>
                        </div>''',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'''<div class="obs-box {"positive" if state_higher else "negative"}">
                            <div class="obs-icon">{"↑" if state_higher else "↓"}</div>
                            <div class="obs-text">Predicted yield is <b>{"higher" if state_higher else "lower"}</b>
                            than the historical average for <b>{state}</b> ({state_avg:,.0f} kg/ha).</div>
                        </div>''',
                        unsafe_allow_html=True,
                    )

                    rain_min, rain_max = df["rainfall_mm"].min(), df["rainfall_mm"].max()
                    if rainfall < rain_min or rainfall > rain_max:
                        st.markdown(
                            '''<div class="obs-box neutral">
                                <div class="obs-icon">⚠️</div>
                                <div class="obs-text">Entered rainfall is outside the historical range
                                seen in the training data.</div>
                            </div>''',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        '<div class="info-note"><b>Note:</b> This is a statistical estimate from historical '
                        'data, not guaranteed agricultural advice.</div>',
                        unsafe_allow_html=True,
                    )


# ============================================================
# PAGE 3: DATA INSIGHTS
# ============================================================
elif page == "📊 Data Insights":
    st.markdown("# 📊 Data Insights")

    st.subheader("Dataset Overview")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Rows", f"{df.shape[0]:,}")
    o2.metric("Columns", df.shape[1])
    o3.metric("Missing Values", int(df.isnull().sum().sum()))
    o4.metric("Duplicate Rows", int(df.duplicated().sum()))

    o5, o6, o7 = st.columns(3)
    o5.metric("Unique Crops", df["crop"].nunique())
    o6.metric("Unique States", df["state"].nunique())
    o7.metric("Year Range", f"{int(df['year'].min())}–{int(df['year'].max())}")

    with st.expander("Missing values per column"):
        st.dataframe(df.isnull().sum().rename("Missing Count"))

    st.markdown("---")
    st.subheader("Model Comparison")
    st.dataframe(
        comparison_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2 Score": "{:.3f}"}),
        use_container_width=True,
    )
    st.caption(f"✅ Best model selected: **{best_model_name}**")

    st.markdown("---")
    st.subheader("Distributions & Relationships")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df_clean, x="crop", color_discrete_sequence=["#2e8b57"], title="Crop Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df_clean, x="state", color_discrete_sequence=["#3b82c4"], title="State Distribution")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(df_clean, x=TARGET_COLUMN, nbins=40, color_discrete_sequence=["#7b2fd6"], title="Yield Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.scatter(df_clean, x="rainfall_mm", y=TARGET_COLUMN, opacity=0.4,
                          color_discrete_sequence=["#e08b1d"], title="Rainfall vs Yield")
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        fig = px.scatter(df_clean, x="temperature_C", y=TARGET_COLUMN, opacity=0.4,
                          color_discrete_sequence=["#d64545"], title="Temperature vs Yield")
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        fig = px.scatter(df_clean, x="humidity_pct", y=TARGET_COLUMN, opacity=0.4,
                          color_discrete_sequence=["#2e8b57"], title="Humidity vs Yield")
        st.plotly_chart(fig, use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        fig = px.scatter(df_clean, x="soil_pH", y=TARGET_COLUMN, opacity=0.4,
                          color_discrete_sequence=["#3b82c4"], title="Soil pH vs Yield")
        st.plotly_chart(fig, use_container_width=True)
    with c8:
        corr = df_clean[NUMERIC_FEATURES + [TARGET_COLUMN]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Greens", title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 4: CROP ANALYSIS
# ============================================================
elif page == "🌾 Crop Analysis":
    st.markdown("# 🌾 Crop Analysis")

    crop_choice = st.selectbox("Select a crop", sorted(df_clean["crop"].unique()))
    crop_df = df_clean[df_clean["crop"] == crop_choice]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average Yield", f"{crop_df[TARGET_COLUMN].mean():,.0f} kg/ha")
    c2.metric("Minimum Yield", f"{crop_df[TARGET_COLUMN].min():,.0f} kg/ha")
    c3.metric("Maximum Yield", f"{crop_df[TARGET_COLUMN].max():,.0f} kg/ha")
    c4.metric("Number of Records", f"{len(crop_df):,}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader(f"Yield Trend — {crop_choice}")
        trend = crop_df.groupby("year")[TARGET_COLUMN].mean().reset_index()
        fig = px.line(trend, x="year", y=TARGET_COLUMN, markers=True, color_discrete_sequence=["#2e8b57"])
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader(f"State-wise Yield — {crop_choice}")
        state_yield = crop_df.groupby("state")[TARGET_COLUMN].mean().sort_values().reset_index()
        fig = px.bar(state_yield, x=TARGET_COLUMN, y="state", orientation="h", color_discrete_sequence=["#3b82c4"])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Yield Distribution — {crop_choice}")
    fig = px.histogram(crop_df, x=TARGET_COLUMN, nbins=30, color_discrete_sequence=["#7b2fd6"])
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 5: ABOUT PROJECT
# ============================================================
elif page == "ℹ️ About Project":
    st.markdown("# ℹ️ About This Project")
    st.write(
        """
        **Crop Yield Prediction** is a machine-learning dashboard that estimates
        crop yield (`yield_kg_ha`) from soil, weather, crop and state/year data.

        **Dataset:** `FAO_Crop_data.csv` — ~18,000 records across Indian states
        (2020–2024), with the following fields:
        """
    )
    st.code(", ".join(REQUIRED_COLUMNS), language="text")

    st.write(
        f"""
        **Modeling approach**
        - Preprocessing: median imputation + scaling for numeric features,
          most-frequent imputation + one-hot encoding for categorical features
          (`state`, `crop`), built with `ColumnTransformer` + `Pipeline`.
        - Models compared: Linear Regression, Decision Tree, Random Forest,
          Gradient Boosting — evaluated on a held-out 20% test split using
          MAE, RMSE and R².
        - **Selected model for predictions: {best_model_name}**
          (R² = {comparison_df.iloc[0]['R2 Score']:.3f} on test data).

        **Important limitations**
        - Predictions are statistical estimates based on historical patterns
          in this dataset — not guaranteed farming advice.
        - Only the fields present in the source dataset are used; no external
          columns (district, soil type, irrigation, fertilizer, market price,
          etc.) were invented or assumed.
        """
    )
