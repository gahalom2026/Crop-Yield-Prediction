"""
Crop Yield Prediction Dashboard
--------------------------------
A professional Streamlit application that trains a regression pipeline on
FAO_Crop_data.csv and lets a user predict crop yield (yield_kg_ha) from
soil, weather, crop, location and year features.

Run with:
    streamlit run app.py

Dataset columns used (EXACTLY as in FAO_Crop_data.csv, nothing invented):
    year, state, district, rainfall_mm, temperature_C, humidity_pct,
    soil_N_kg_ha, soil_P_kg_ha, soil_K_kg_ha, soil_pH, crop, yield_kg_ha
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
CATEGORICAL_FEATURES = ["state", "district", "crop"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "yield_kg_ha"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

CANDIDATE_DATA_PATHS = [
    "FAO_Crop_data.csv",
    "data/FAO_Crop_data.csv",
    os.path.join(os.path.dirname(__file__), "FAO_Crop_data.csv"),
    os.path.join(os.path.dirname(__file__), "data", "FAO_Crop_data.csv"),
]

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
    .obs-box {
        background-color: #f2f9f4;
        border-left: 4px solid #1e7c3f;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        color: #163d27 !important;
    }
    .obs-box, .obs-box p, .obs-box span {
        color: #163d27 !important;
    }
    .obs-box strong {
        color: #0f5c2c !important;
    }
    h1, h2, h3 { color: #163d27; }
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
    ["🏠 Dashboard", "🌾 Predict Yield", "📊 Data Insights", "🌿 Crop Analysis", "ℹ️ About Project"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### DATASET OVERVIEW")
st.sidebar.write(f"**Total Records:** {len(df):,}")
st.sidebar.write(f"**Total Crops:** {df['crop'].nunique()}")
st.sidebar.write(f"**States:** {df['state'].nunique()}")
st.sidebar.write(f"**Years:** {int(df['year'].min())} – {int(df['year'].max())}")
st.sidebar.markdown("---")
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
elif page == "🌾 Predict Yield":
    st.markdown("# 🌾 Predict Yield")
    st.caption("Fill in field conditions to get a predicted yield")

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("📋 Input Parameters")

        # State and District live OUTSIDE the form on purpose: widgets inside a
        # st.form only trigger a rerun when the form is submitted, so a District
        # dropdown built there would not react to a State change until submit.
        # Placing them here means District updates immediately, and it is built
        # directly from the CSV (state -> its own district values only).
        STATE_PLACEHOLDER = "Choose state"
        DISTRICT_PLACEHOLDER = "Choose district"
        CROP_PLACEHOLDER = "Select crop"

        sd1, sd2 = st.columns(2)
        with sd1:
            state_options = [STATE_PLACEHOLDER] + sorted(df["state"].dropna().unique())
            state = st.selectbox("State", state_options, index=0, key="predict_state")
        with sd2:
            if state == STATE_PLACEHOLDER:
                district = st.selectbox("District", [DISTRICT_PLACEHOLDER], index=0, key="predict_district")
            else:
                available_districts = sorted(
                    df.loc[df["state"] == state, "district"].dropna().unique()
                )
                district = st.selectbox("District", available_districts, key="predict_district")

        with st.form("predict_form"):
            f1, f2 = st.columns(2)
            with f1:
                year_options = sorted(set(df["year"].unique().tolist() + [int(df["year"].max()) + 1]))
                year = st.selectbox("Year", year_options, index=len(year_options) - 2)
                crop = st.selectbox("Crop", [CROP_PLACEHOLDER] + sorted(df["crop"].unique()), index=0)
                rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=float(df["rainfall_mm"].median()))

            with f2:
                temperature = st.number_input("Temperature (°C)", value=float(df["temperature_C"].median()))
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=float(df["humidity_pct"].median()))
                n = st.number_input("Soil Nitrogen (kg/ha)", min_value=0.0, value=float(df["soil_N_kg_ha"].median()))
                p = st.number_input("Soil Phosphorus (kg/ha)", min_value=0.0, value=float(df["soil_P_kg_ha"].median()))
                k = st.number_input("Soil Potassium (kg/ha)", min_value=0.0, value=float(df["soil_K_kg_ha"].median()))

            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=float(df["soil_pH"].median()))
            submitted = st.form_submit_button("🔍 Predict Yield")

    with right:
        st.subheader("🌾 Predicted Yield")
        if submitted:
            if state == STATE_PLACEHOLDER or district == DISTRICT_PLACEHOLDER or crop == CROP_PLACEHOLDER:
                st.error("❌ Please choose a State, District and Crop before predicting.")
                st.stop()

            input_dict = {
                "year": year, "rainfall_mm": rainfall, "temperature_C": temperature,
                "humidity_pct": humidity, "soil_N_kg_ha": n, "soil_P_kg_ha": p,
                "soil_K_kg_ha": k, "soil_pH": ph, "state": state, "district": district,
                "crop": crop,
            }
            try:
                pred_kg = make_prediction(best_pipeline, input_dict)
            except Exception as e:
                st.error(f"❌ An error occurred while making the prediction: {e}")
                st.stop()

            pred_tons = pred_kg / 1000

            st.markdown(
                f'<div class="metric-card"><div class="big">{pred_kg:,.0f} kg/ha</div>'
                f'<div class="label">≈ {pred_tons:,.2f} tons/hectare</div></div>',
                unsafe_allow_html=True,
            )

            crop_avg = df_clean[df_clean["crop"] == crop][TARGET_COLUMN].mean()
            state_avg = df_clean[df_clean["state"] == state][TARGET_COLUMN].mean()

            st.write("")
            m1, m2 = st.columns(2)
            m1.metric(f"Avg Yield — {crop}", f"{crop_avg:,.0f} kg/ha",
                      delta=f"{pred_kg - crop_avg:,.0f} kg/ha")
            m2.metric(f"Avg Yield — {state}", f"{state_avg:,.0f} kg/ha",
                      delta=f"{pred_kg - state_avg:,.0f} kg/ha")

            st.markdown("#### 📌 Data-driven Observations")
            obs = []
            obs.append(
                f"Predicted yield is **{'higher' if pred_kg >= crop_avg else 'lower'}** "
                f"than the historical average for **{crop}** ({crop_avg:,.0f} kg/ha)."
            )
            obs.append(
                f"Predicted yield is **{'higher' if pred_kg >= state_avg else 'lower'}** "
                f"than the historical average for **{state}** ({state_avg:,.0f} kg/ha)."
            )
            rain_min, rain_max = df["rainfall_mm"].min(), df["rainfall_mm"].max()
            if rainfall < rain_min or rainfall > rain_max:
                obs.append("⚠️ Entered rainfall is outside the historical range seen in training data.")
            for o in obs:
                st.markdown(f'<div class="obs-box">{o}</div>', unsafe_allow_html=True)

            st.caption(
                "Note: this is a statistical estimate from historical data, not guaranteed "
                "agricultural advice."
            )
        else:
            st.info("Fill in the form and click **Predict Yield** to see the result here.")


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
elif page == "🌿 Crop Analysis":
    st.markdown("# 🌿 Crop Analysis")

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
        crop yield (`yield_kg_ha`) from soil, weather, crop and location data.

        **Dataset:** `FAO_Crop_data.csv` — ~18,000 records across Indian states
        and districts (2020–2024), with the following fields:
        """
    )
    st.code(", ".join(REQUIRED_COLUMNS), language="text")

    st.write(
        f"""
        **Modeling approach**
        - Preprocessing: median imputation + scaling for numeric features,
          most-frequent imputation + one-hot encoding for categorical features
          (`state`, `district`, `crop`), built with `ColumnTransformer` + `Pipeline`.
        - Models compared: Linear Regression, Decision Tree, Random Forest,
          Gradient Boosting — evaluated on a held-out 20% test split using
          MAE, RMSE and R².
        - **Selected model for predictions: {best_model_name}**
          (R² = {comparison_df.iloc[0]['R2 Score']:.3f} on test data).

        **Important limitations**
        - Predictions are statistical estimates based on historical patterns
          in this dataset — not guaranteed farming advice.
        - Only the fields present in the source dataset are used; no external
          columns (soil type, irrigation, fertilizer, market price, etc.)
          were invented or assumed.
        """
    )
