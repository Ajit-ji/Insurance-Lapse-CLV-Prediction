
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Insurance Lapse & CLV Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 32px;
        font-weight: bold;
    }

    .section-title {
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# 3. TITLE
# ==========================================================

st.markdown(
    '<div class="main-title">📊 Insurance Lapse & CLV Analytics</div>',
    unsafe_allow_html=True
)

st.write(
    "Explore insurance customer risk scores, survival analysis, "
    "customer segmentation, and premium-based value."
)

st.caption(
    "Note: Model predictions and value estimates are analytical "
    "outputs and require further validation before operational use."
)


# ==========================================================
# 4. LOAD DATA
# ==========================================================

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Possible dataset locations
possible_paths = [
    PROJECT_DIR / "outputs" / "audit" / "integrated_rsf_customer_results.csv",
    PROJECT_DIR.parent / "outputs" / "audit" / "integrated_rsf_customer_results.csv",
]

DATA_PATH = None

for path in possible_paths:
    if path.exists():
        DATA_PATH = path
        break


if DATA_PATH is None:

    st.error(
        "Integrated customer results CSV was not found."
    )

    st.write("Checked the following locations:")

    for path in possible_paths:
        st.code(str(path))

    st.info(
        "Please verify that integrated_rsf_customer_results.csv "
        "exists inside the outputs/audit folder."
    )

    st.stop()


# Read dataset
try:

    df = pd.read_csv(DATA_PATH)

except Exception as error:

    st.error("An error occurred while loading the dataset.")
    st.exception(error)
    st.stop()


if df.empty:

    st.warning("The dataset is empty.")
    st.stop()


st.success(
    f"Dataset loaded successfully: {DATA_PATH.name}"
)


# ==========================================================
# 5. SIDEBAR
# ==========================================================

st.sidebar.header("Dashboard Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard Overview",
        "Customer Results",
        "Risk Analysis",
        "Customer Segmentation",
        "Premium Value Analysis",
        "Model Performance",
        "Raw Customer Prediction"
    ]
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def find_columns(keywords):
    """
    Find dataset columns that contain any of the provided keywords.
    """
    return [
        column
        for column in df.columns
        if any(
            keyword.lower() in column.lower()
            for keyword in keywords
        )
    ]


def numeric_columns():
    """
    Return numeric columns from the dataset.
    """
    return df.select_dtypes(
        include=np.number
    ).columns.tolist()


def safe_numeric_series(column_name):
    """
    Convert a column into numeric values and remove missing values.
    """
    return pd.to_numeric(
        df[column_name],
        errors="coerce"
    ).dropna()



# ==========================================================
# MODEL AND PREDICTION HELPERS
# ==========================================================


def find_existing_path(relative_path):
    """Search the app folder and its parent folder for a file."""
    possible_paths = [
        PROJECT_DIR / relative_path,
        PROJECT_DIR.parent / relative_path,
    ]

    for candidate in possible_paths:
        if candidate.exists():
            return candidate

    return None


@st.cache_resource
def load_rsf_components():
    """Load the fitted RSF preprocessor and model."""
    preprocessor_path = find_existing_path(
        Path("outputs") / "models" / "rsf_preprocessor.joblib"
    )
    model_path = find_existing_path(
        Path("outputs") / "models" / "rsf_model.joblib"
    )

    if preprocessor_path is None:
        raise FileNotFoundError(
            "rsf_preprocessor.joblib was not found in outputs/models."
        )

    if model_path is None:
        raise FileNotFoundError(
            "rsf_model.joblib was not found in outputs/models."
        )

    return joblib.load(preprocessor_path), joblib.load(model_path)


def normalize_binary_value(value):
    """Convert common binary values into 0/1 for the trained pipeline."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"yes", "y", "true", "1"}:
            return 1
        if normalized in {"no", "n", "false", "0"}:
            return 0

        try:
            number = float(normalized)
            if number in {0, 1}:
                return int(number)
        except ValueError:
            return np.nan

        return np.nan

    try:
        number = float(value)
        if number in {0, 1}:
            return int(number)
    except (TypeError, ValueError):
        return np.nan

    return np.nan


def normalize_category(value, allowed_values):
    """Map missing/unseen categories to the training category Unknown."""
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip()
    return value if value in allowed_values else "Unknown"


def prepare_customer_input(
    age,
    income,
    annual_premium,
    has_children,
    home_owner,
    college_degree,
    good_credit,
    marital_status,
    county,
):
    """Build the exact nine raw columns used by the RSF preprocessor."""
    return pd.DataFrame([
        {
            "age": age,
            "income": income,
            "annual_premium": annual_premium,
            "has_children": normalize_binary_value(has_children),
            "home_owner": normalize_binary_value(home_owner),
            "college_degree": normalize_binary_value(college_degree),
            "good_credit": normalize_binary_value(good_credit),
            "marital_status": normalize_category(
                marital_status,
                {"Unknown", "Single", "Married"},
            ),
            "county": normalize_category(
                county,
                {
                    "Unknown", "Collin", "Cooke", "Dallas", "Denton",
                    "Ellis", "Grayson", "Hill", "Hunt", "Johnson",
                    "Kaufman", "Navarro", "Parker", "Rockwall", "Tarrant",
                },
            ),
        }
    ])


def get_first_existing_column(dataframe, candidates):
    """Return the first available column from a list of candidates."""
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def get_risk_category(event_risk_percentage):
    """Return an illustrative, non-validated risk label."""
    if event_risk_percentage < 20:
        return "Lower estimated event risk"
    if event_risk_percentage < 50:
        return "Moderate estimated event risk"
    return "Higher estimated event risk"


def generate_rsf_prediction(customer_input, prediction_horizon_days):
    """
    Generate the raw RSF score and a survival-based event-risk estimate.

    Assumption: model time is measured in days. The percentage is
    calculated as 1 - predicted survival probability at the selected
    model time. It is not a calibrated probability until validated.
    """
    rsf_preprocessor, rsf_model = load_rsf_components()

    transformed_customer = rsf_preprocessor.transform(customer_input)

    if transformed_customer.shape[1] != 25:
        raise ValueError(
            "The transformed input has "
            f"{transformed_customer.shape[1]} features; expected 25."
        )

    raw_risk_score = rsf_model.predict(transformed_customer)
    raw_risk_score = float(np.asarray(raw_risk_score).reshape(-1)[0])

    if not hasattr(rsf_model, "predict_survival_function"):
        raise AttributeError(
            "The loaded model does not support survival-function prediction."
        )

    survival_functions = rsf_model.predict_survival_function(
        transformed_customer,
        return_array=True,
    )

    survival_function = np.asarray(survival_functions[0])
    model_times = np.asarray(rsf_model.unique_times_)

    if len(model_times) == 0 or len(survival_function) == 0:
        raise ValueError("The RSF model does not contain survival time points.")

    # Choose the closest model time that is not greater than the requested
    # horizon. If the horizon is before the first model time, use the first.
    time_index = np.searchsorted(
        model_times,
        prediction_horizon_days,
        side="right",
    ) - 1

    time_index = int(np.clip(time_index, 0, len(model_times) - 1))

    selected_time = float(model_times[time_index])
    survival_probability = float(survival_function[time_index])
    survival_probability = float(np.clip(survival_probability, 0, 1))

    event_risk_percentage = float(
        np.clip((1 - survival_probability) * 100, 0, 100)
    )

    return {
        "raw_risk_score": raw_risk_score,
        "survival_probability": survival_probability,
        "event_risk_percentage": event_risk_percentage,
        "selected_time": selected_time,
        "transformed_shape": transformed_customer.shape,
    }


# ==========================================================
# 6. DASHBOARD OVERVIEW
# ==========================================================

if page == "Dashboard Overview":

    st.header("Dashboard Overview")

    total_customers = len(df)
    total_columns = len(df.columns)
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:

        st.metric(
            "Total Columns",
            f"{total_columns:,}"
        )

    with col3:

        st.metric(
            "Missing Values",
            f"{missing_values:,}"
        )

    with col4:

        st.metric(
            "Duplicate Rows",
            f"{duplicate_rows:,}"
        )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    information_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Unique Values": [
            df[column].nunique()
            for column in df.columns
        ]
    })

    st.dataframe(
        information_df,
        use_container_width=True
    )

    st.subheader("Available Columns")

    st.write(df.columns.tolist())


# ==========================================================
# 7. CUSTOMER RESULTS
# ==========================================================

elif page == "Customer Results":

    st.header("Customer Results")

    st.write(
        f"Total available customer records: {len(df):,}"
    )

    # Search by customer ID if available
    customer_id_columns = find_columns(
        ["customer_id", "customerid"]
    )

    filtered_df = df.copy()

    if customer_id_columns:

        customer_id_column = customer_id_columns[0]

        search_text = st.text_input(
            "Search Customer ID",
            placeholder="Enter customer ID"
        )

        if search_text.strip():

            filtered_df = filtered_df[
                filtered_df[customer_id_column]
                .astype(str)
                .str.contains(
                    search_text.strip(),
                    case=False,
                    na=False
                )
            ]

    rows_to_display = st.slider(
        "Number of rows to display",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )

    st.write(
        f"Matching records: {len(filtered_df):,}"
    )

    st.dataframe(
        filtered_df.head(rows_to_display),
        use_container_width=True
    )

    st.subheader("Download Customer Results")

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Filtered Results",
        data=csv_data,
        file_name="filtered_customer_results.csv",
        mime="text/csv"
    )


# ==========================================================
# 8. RISK ANALYSIS
# ==========================================================

elif page == "Risk Analysis":

    st.header("Risk Score Analysis")

    risk_columns = find_columns(
        ["risk"]
    )

    if not risk_columns:

        st.warning(
            "No column containing 'risk' was found."
        )

        st.write("Available columns:")

        st.write(df.columns.tolist())

    else:

        selected_risk_column = st.selectbox(
            "Select Risk Score Column",
            risk_columns
        )

        risk_data = safe_numeric_series(
            selected_risk_column
        )

        if risk_data.empty:

            st.warning(
                "The selected risk column does not contain "
                "usable numeric values."
            )

        else:

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Minimum Risk Score",
                    f"{risk_data.min():.4f}"
                )

            with col2:

                st.metric(
                    "Average Risk Score",
                    f"{risk_data.mean():.4f}"
                )

            with col3:

                st.metric(
                    "Median Risk Score",
                    f"{risk_data.median():.4f}"
                )

            with col4:

                st.metric(
                    "Maximum Risk Score",
                    f"{risk_data.max():.4f}"
                )

            st.subheader("Risk Score Distribution")

            histogram_data = pd.DataFrame({
                "Risk Score": risk_data
            })

            st.bar_chart(
                histogram_data["Risk Score"].value_counts(
                    bins=20
                ).sort_index()
            )

            st.subheader("Risk Score Summary")

            risk_summary = risk_data.describe().to_frame(
                name="Value"
            )

            st.dataframe(
                risk_summary,
                use_container_width=True
            )


# ==========================================================
# 9. CUSTOMER SEGMENTATION
# ==========================================================

elif page == "Customer Segmentation":

    st.header("Customer Segmentation")

    segment_columns = find_columns(
        ["segment", "category"]
    )

    if not segment_columns:

        st.warning(
            "No segment or category column was found."
        )

        st.write("Available columns:")

        st.write(df.columns.tolist())

    else:

        selected_segment_column = st.selectbox(
            "Select Customer Segment Column",
            segment_columns
        )

        segment_series = (
            df[selected_segment_column]
            .fillna("Unknown")
            .astype(str)
        )

        segment_counts = (
            segment_series
            .value_counts()
        )

        st.subheader("Customers by Segment")

        st.bar_chart(segment_counts)

        segment_summary = (
            segment_counts
            .rename_axis("Segment")
            .reset_index(name="Customer_Count")
        )

        segment_summary["Percentage"] = (
            segment_summary["Customer_Count"]
            / len(df)
            * 100
        ).round(2)

        st.subheader("Segment Summary")

        st.dataframe(
            segment_summary,
            use_container_width=True
        )

        st.subheader("Filter Customers by Segment")

        segment_options = [
            "All"
        ] + sorted(
            segment_series.unique().tolist()
        )

        selected_segment = st.selectbox(
            "Choose Segment",
            segment_options
        )

        if selected_segment == "All":

            filtered_customers = df.copy()

        else:

            filtered_customers = df[
                segment_series == selected_segment
            ]

        st.write(
            f"Customers in selection: "
            f"{len(filtered_customers):,}"
        )

        st.dataframe(
            filtered_customers.head(100),
            use_container_width=True
        )

        segment_csv = filtered_customers.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Selected Segment",
            data=segment_csv,
            file_name="selected_customer_segment.csv",
            mime="text/csv"
        )


# ==========================================================
# 10. PREMIUM-BASED VALUE ANALYSIS
# ==========================================================

elif page == "Premium Value Analysis":

    st.header("Premium-Based Customer Value")

    value_columns = find_columns(
        ["value", "clv", "premium"]
    )

    if not value_columns:

        st.warning(
            "No value, CLV, or premium-related column was found."
        )

        st.write("Available columns:")

        st.write(df.columns.tolist())

    else:

        selected_value_column = st.selectbox(
            "Select Value Column",
            value_columns
        )

        value_data = safe_numeric_series(
            selected_value_column
        )

        if value_data.empty:

            st.warning(
                "The selected column does not contain "
                "usable numeric values."
            )

        else:

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Minimum Value",
                    f"{value_data.min():,.2f}"
                )

            with col2:

                st.metric(
                    "Average Value",
                    f"{value_data.mean():,.2f}"
                )

            with col3:

                st.metric(
                    "Median Value",
                    f"{value_data.median():,.2f}"
                )

            with col4:

                st.metric(
                    "Total Value",
                    f"{value_data.sum():,.2f}"
                )

            st.subheader("Value Distribution")

            st.bar_chart(
                value_data.value_counts(
                    bins=20
                ).sort_index()
            )

            st.subheader("Value Summary")

            value_summary = value_data.describe().to_frame(
                name="Value"
            )

            st.dataframe(
                value_summary,
                use_container_width=True
            )

            st.info(
                "The current value metric is a premium-based "
                "value proxy, not a complete profit-based CLV."
            )


# ==========================================================
# 11. MODEL PERFORMANCE
# ==========================================================

elif page == "Model Performance":

    st.header("Model Performance")

    st.subheader("Random Survival Forest Evaluation")

    # Use saved evaluation file when available
    evaluation_path = (
        PROJECT_DIR
        / "outputs"
        / "audit"
        / "final_model_evaluation.csv"
    )

    if evaluation_path.exists():

        evaluation_df = pd.read_csv(
            evaluation_path
        )

        st.dataframe(
            evaluation_df,
            use_container_width=True
        )

        st.download_button(
            label="Download Model Evaluation",
            data=evaluation_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="final_model_evaluation.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "The final_model_evaluation.csv file was not found."
        )

        st.write(
            "Run the model evaluation notebook and save "
            "the evaluation CSV before using this page."
        )

    st.subheader("Reported Model Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "RSF Test C-index",
            "0.5749"
        )

    with col2:

        st.metric(
            "RSF Brier Score",
            "0.0524"
        )

    with col3:

        st.metric(
            "Baseline Brier Score",
            "0.0536"
        )

    st.subheader("Interpretation")

    st.write(
        """
        The Random Survival Forest model achieved a test C-index
        of approximately 0.5749.

        Its Integrated Brier Score was 0.0524, compared with
        0.0536 for the baseline model.

        The lower Brier Score indicates a modest improvement
        over the baseline for this evaluation dataset.

        These results should not be interpreted as proof of
        highly accurate customer lapse prediction.
        """
    )

    st.info(
        "The model metrics are based on the current evaluation "
        "run and should be updated if the model is retrained."
    )
# ==========================================================
# 12. RAW CUSTOMER PREDICTION
# ==========================================================

elif page == "Raw Customer Prediction":
    st.header("Raw Customer Prediction")

    st.write(
        "Generate an RSF prediction using either manually entered "
        "customer information or a real customer selected from the dataset."
    )

    st.info(
        "The percentage shown below is a survival-based event-risk estimate "
        "at a selected time horizon. It is not a calibrated probability "
        "until calibration and external validation are completed."
    )

    prediction_mode = st.radio(
        "Choose Input Method",
        [
            "Manual Customer Input",
            "Select Customer from Dataset",
        ],
        horizontal=True,
    )

    horizon_options = {
        "6 months": 182,
        "1 year": 365,
        "2 years": 730,
        "3 years": 1095,
    }

    selected_horizon_label = st.selectbox(
        "Prediction Horizon",
        list(horizon_options.keys()),
        index=1,
        help="The model time unit is assumed to be days.",
    )

    prediction_horizon_days = horizon_options[selected_horizon_label]

    raw_customer_df = None
    estimated_duration_years = None
    generate_prediction = False

    # ------------------------------------------------------
    # 12A. SELECT A REAL CUSTOMER FROM THE DATASET
    # ------------------------------------------------------

    if prediction_mode == "Select Customer from Dataset":
        st.subheader("Select an Existing Customer")

        customer_id_column = get_first_existing_column(
            df,
            ["customer_id", "customerid", "Customer_ID", "CustomerID"],
        )

        required_columns = [
            "age",
            "income",
            "annual_premium",
            "has_children",
            "home_owner",
            "college_degree",
            "good_credit",
            "marital_status",
            "county",
        ]

        missing_columns = [
            column for column in required_columns if column not in df.columns
        ]

        if customer_id_column is None:
            st.error("A customer ID column was not found in the dataset.")
        elif missing_columns:
            st.error(
                "The dataset is missing required columns: "
                + ", ".join(missing_columns)
            )
        else:
            customer_ids = (
                df[customer_id_column]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

            if not customer_ids:
                st.warning("No customer IDs are available.")
            else:
                selected_customer_id = st.selectbox(
                    "Customer ID",
                    customer_ids,
                )

                matching_rows = df[
                    df[customer_id_column].astype(str) == selected_customer_id
                ]

                selected_customer_row = matching_rows.iloc[0]

                st.subheader("Selected Customer Record")
                st.dataframe(
                    matching_rows.head(1),
                    use_container_width=True,
                )

                raw_customer_df = prepare_customer_input(
                    age=selected_customer_row["age"],
                    income=selected_customer_row["income"],
                    annual_premium=selected_customer_row["annual_premium"],
                    has_children=selected_customer_row["has_children"],
                    home_owner=selected_customer_row["home_owner"],
                    college_degree=selected_customer_row["college_degree"],
                    good_credit=selected_customer_row["good_credit"],
                    marital_status=selected_customer_row["marital_status"],
                    county=selected_customer_row["county"],
                )

                duration_column = get_first_existing_column(
                    df,
                    ["duration_years", "duration_year", "duration_days"],
                )

                if duration_column is not None:
                    estimated_duration_years = selected_customer_row[
                        duration_column
                    ]

                st.subheader("Normalized Model Input")
                st.dataframe(raw_customer_df, use_container_width=True)

                generate_prediction = st.button(
                    "Generate Prediction for Selected Customer",
                    type="primary",
                )

    # ------------------------------------------------------
    # 12B. MANUAL CUSTOMER INPUT
    # ------------------------------------------------------

    else:
        st.subheader("Enter New Customer Information")

        with st.form("manual_customer_prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input(
                    "Age",
                    min_value=18,
                    max_value=100,
                    value=30,
                    step=1,
                )

                income = st.number_input(
                    "Annual Income",
                    min_value=0.0,
                    value=500000.0,
                    step=10000.0,
                )

                annual_premium = st.number_input(
                    "Annual Premium",
                    min_value=0.0,
                    value=25000.0,
                    step=1000.0,
                )

                has_children = st.selectbox(
                    "Has Children",
                    ["Yes", "No"],
                )

                home_owner = st.selectbox(
                    "Home Owner",
                    ["Yes", "No"],
                )

            with col2:
                college_degree = st.selectbox(
                    "College Degree",
                    ["Yes", "No"],
                )

                good_credit = st.selectbox(
                    "Good Credit",
                    ["Yes", "No"],
                )

                marital_status = st.selectbox(
                    "Marital Status",
                    ["Unknown", "Single", "Married"],
                )

                county = st.selectbox(
                    "County",
                    [
                        "Unknown",
                        "Collin",
                        "Cooke",
                        "Dallas",
                        "Denton",
                        "Ellis",
                        "Grayson",
                        "Hill",
                        "Hunt",
                        "Johnson",
                        "Kaufman",
                        "Navarro",
                        "Parker",
                        "Rockwall",
                        "Tarrant",
                    ],
                )

            estimated_duration_years = st.number_input(
                "Duration Reference (Years)",
                min_value=0.0,
                value=1.0,
                step=0.5,
                help="Reference only; duration is not used as an RSF input.",
            )

            generate_prediction = st.form_submit_button(
                "Generate Manual Prediction",
                type="primary",
            )

        if generate_prediction:
            raw_customer_df = prepare_customer_input(
                age=age,
                income=income,
                annual_premium=annual_premium,
                has_children=has_children,
                home_owner=home_owner,
                college_degree=college_degree,
                good_credit=good_credit,
                marital_status=marital_status,
                county=county,
            )

            st.subheader("Manual Model Input")
            st.dataframe(raw_customer_df, use_container_width=True)

    # ------------------------------------------------------
    # 12C. GENERATE AND DISPLAY HUMAN-READABLE PREDICTION
    # ------------------------------------------------------

    if generate_prediction and raw_customer_df is not None:
        try:
            with st.spinner("Applying preprocessing and generating prediction..."):
                prediction_result = generate_rsf_prediction(
                    raw_customer_df,
                    prediction_horizon_days=prediction_horizon_days,
                )

            raw_risk_score = prediction_result["raw_risk_score"]
            survival_probability = prediction_result["survival_probability"]
            event_risk_percentage = prediction_result["event_risk_percentage"]
            selected_time = prediction_result["selected_time"]
            transformed_shape = prediction_result["transformed_shape"]
            risk_category = get_risk_category(event_risk_percentage)

            st.success("Prediction generated successfully.")

            st.subheader("Human-Readable Prediction")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Estimated Event Risk",
                    f"{event_risk_percentage:.2f}%",
                )

            with col2:
                st.metric(
                    "Estimated Survival",
                    f"{survival_probability * 100:.2f}%",
                )

            with col3:
                st.metric(
                    "Model Time Used",
                    f"{selected_time:.0f} days",
                )

            st.subheader("Risk Interpretation")
            st.write(f"**Category:** {risk_category}")

            st.progress(
                int(round(event_risk_percentage)),
                text=f"Estimated event risk: {event_risk_percentage:.2f}%",
            )

            st.caption(
                "The category thresholds are illustrative and have not been "
                "validated as business decision thresholds."
            )

            st.subheader("Prediction Summary")

            summary_df = raw_customer_df.copy()
            summary_df["prediction_horizon"] = selected_horizon_label
            summary_df["estimated_event_risk_percentage"] = round(
                event_risk_percentage, 2
            )
            summary_df["estimated_survival_percentage"] = round(
                survival_probability * 100, 2
            )

            if estimated_duration_years is not None:
                summary_df["duration_reference"] = estimated_duration_years

            st.dataframe(summary_df, use_container_width=True)

            with st.expander("Technical Model Details"):
                st.write(f"Raw RSF risk score: {raw_risk_score:.6f}")
                st.write(f"Transformed feature count: {transformed_shape[1]}")
                st.write(f"Selected model time: {selected_time:.0f} days")
                st.write(
                    "Event risk formula: (1 - survival probability) × 100"
                )

            st.warning(
                "This percentage is a model-based survival estimate. "
                "It is not a guaranteed or calibrated real-world probability. "
                "The event definition and calibration must be validated before "
                "operational use."
            )

        except FileNotFoundError as error:
            st.error("Required model files were not found.")
            st.exception(error)

        except Exception as error:
            st.error("An error occurred while generating the prediction.")
            st.exception(error)
            st.info(
                "Check that rsf_model.joblib and rsf_preprocessor.joblib "
                "were created from the same training run and that the "
                "raw input columns match the training data."
            )

# ==========================================================
# 13. FOOTER
# ==========================================================

st.divider()

st.caption(
    "Insurance Lapse & CLV Analytics | "
    "Random Survival Forest | "
    "Premium-Based Customer Value Proxy"
)

