import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DataSentinel AI",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    [data-testid="stMetric"] {
        background-color: #1b2330;
        border: 1px solid #303846;
        padding: 18px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🛡️ DataSentinel AI")
st.subheader("Intelligent Dataset Auditing & Impact Analysis")

st.write(
    "Investigate dataset quality, detect suspicious patterns, "
    "and understand potential risks before training machine learning models."
)

st.divider()


# --------------------------------------------------
# DATA UPLOAD
# --------------------------------------------------

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV dataset",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to begin your dataset audit.")

    st.markdown("### What DataSentinel investigates")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🔍 Data quality")
        st.write("Missing values, duplicate records, and inconsistent columns.")

    with col2:
        st.markdown("#### 📊 Statistical risks")
        st.write("Potential outliers, unusual distributions, and class imbalance.")

    with col3:
        st.markdown("#### 🧪 Model impact")
        st.write("Compare model performance under controlled data interventions.")

    st.stop()


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:
    uploaded_file.seek(0)

    df = pd.read_csv(uploaded_file)

except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()


if df.empty:
    st.error("The uploaded dataset contains no records.")
    st.stop()

if len(df.columns) == 0:
    st.error("The uploaded dataset contains no columns.")
    st.stop()


st.success("Dataset loaded successfully.")

st.markdown("### Dataset overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", f"{len(df):,}")

with col2:
    st.metric("Columns", f"{len(df.columns):,}")

with col3:
    st.metric(
        "Missing cells",
        f"{int(df.isna().sum().sum()):,}",
    )

with col4:
    st.metric(
        "Duplicate rows",
        f"{int(df.duplicated().sum()):,}",
    )


# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------

with st.expander("Preview dataset", expanded=True):
    st.dataframe(df.head(100), use_container_width=True)


# --------------------------------------------------
# DATA QUALITY AUDIT
# --------------------------------------------------

st.divider()
st.header("1. Dataset Quality Audit")

missing = df.isna().sum()

missing_report = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": missing.values,
    "Missing Percentage": (
        missing.values / len(df) * 100
    ).round(2),
    "Data Type": df.dtypes.astype(str).values,
})

duplicates = int(df.duplicated().sum())

quality_col1, quality_col2 = st.columns(2)

with quality_col1:
    st.markdown("#### Missing values")
    st.dataframe(
        missing_report.sort_values(
            "Missing Values",
            ascending=False,
        ),
        use_container_width=True,
    )

with quality_col2:
    st.markdown("#### Data types")
    dtype_report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Unique Values": [
            df[column].nunique(dropna=True)
            for column in df.columns
        ],
    })

    st.dataframe(dtype_report, use_container_width=True)


# --------------------------------------------------
# DUPLICATE ANALYSIS
# --------------------------------------------------

st.markdown("#### Duplicate record analysis")

if duplicates:
    st.warning(
        f"Found {duplicates} duplicated rows. "
        "Investigate whether these represent genuine repeated observations."
    )
else:
    st.success("No completely duplicated rows were detected.")


# --------------------------------------------------
# OUTLIER ANALYSIS
# --------------------------------------------------

st.divider()
st.header("2. Statistical Outlier Investigation")

numeric_df = df.select_dtypes(include=np.number)

if numeric_df.shape[1] == 0:
    st.info("No numeric columns are available for outlier analysis.")

else:
    selected_column = st.selectbox(
        "Select a numeric column",
        numeric_df.columns,
    )

    values = numeric_df[selected_column].dropna()

    if len(values) < 4:
        st.info("At least four non-missing values are required.")

    else:
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outlier_mask = (
                (values < q1) | (values > q3)
            )
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (
                (values < lower_bound)
                | (values > upper_bound)
            )

        outlier_count = int(outlier_mask.sum())

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Potential outliers", outlier_count)

        with col2:
            st.metric(
                "Outlier percentage",
                f"{outlier_count / len(values) * 100:.2f}%",
            )

        with col3:
            st.metric(
                "Distinct values",
                f"{values.nunique():,}",
            )

        fig = px.box(
            df,
            y=selected_column,
            title=f"Distribution of {selected_column}",
            points="outliers",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Outliers are statistical flags, not necessarily errors. "
            "Domain knowledge is required before removing observations."
        )


# --------------------------------------------------
# CLASS IMBALANCE
# --------------------------------------------------

st.divider()
st.header("3. Target Distribution Investigation")

target_column = st.selectbox(
    "Choose a target column for classification analysis",
    ["None"] + list(df.columns),
)

if target_column != "None":

    target = df[target_column].dropna()

    class_counts = target.value_counts()

    if len(class_counts) < 2:
        st.warning(
            "The selected target has fewer than two observed classes."
        )

    else:
        fig = px.bar(
            x=class_counts.index.astype(str),
            y=class_counts.values,
            labels={
                "x": target_column,
                "y": "Number of records",
            },
            title="Target class distribution",
        )

        st.plotly_chart(fig, use_container_width=True)

        minority_count = int(class_counts.min())
        majority_count = int(class_counts.max())

        imbalance_ratio = (
            majority_count / minority_count
            if minority_count > 0
            else float("inf")
        )

        st.metric(
            "Majority-to-minority ratio",
            f"{imbalance_ratio:.2f}:1",
        )

        if imbalance_ratio >= 3:
            st.warning(
                "Potential class imbalance detected. "
                "Consider stratified evaluation and class-specific metrics."
            )
        else:
            st.success(
                "No severe imbalance detected using the 3:1 screening threshold."
            )


# --------------------------------------------------
# CONTROLLED MODEL EXPERIMENT
# --------------------------------------------------

st.divider()
st.header("4. Data Impact Laboratory")

st.write(
    "Compare a baseline model with a model trained after median/mode "
    "imputation and duplicate-row removal. This is a preliminary experiment, "
    "not proof that the cleaning intervention is better."
)

st.warning(
    "Choose a classification target. The target is excluded from the "
    "imputation process, and evaluation uses a held-out test set."
)

if target_column != "None":

    experiment_df = df.dropna(subset=[target_column]).copy()

    if experiment_df[target_column].nunique() < 2:
        st.warning("The target must contain at least two classes.")

    elif experiment_df[target_column].nunique() > 50:
        st.warning(
            "This preliminary experiment is limited to targets with "
            "50 or fewer classes."
        )

    elif len(experiment_df) < 30:
        st.warning(
            "Use at least 30 labelled rows for this preliminary experiment."
        )

    else:
        if st.button("Run controlled experiment"):

            with st.spinner("Training and evaluating models..."):

                try:
                    y = experiment_df[target_column]

                    X = experiment_df.drop(
                        columns=[target_column]
                    )

                    # Convert categorical predictors to numeric indicators.
                    X = pd.get_dummies(
                        X,
                        dummy_na=True,
                        dtype=float,
                    )

                    # Remove columns that contain no usable information.
                    X = X.replace([np.inf, -np.inf], np.nan)

                    X = X.loc[:, X.notna().any()]

                    if X.shape[1] == 0:
                        raise ValueError(
                            "No usable predictor columns were found."
                        )

                    # Remove classes with too few examples for stratified splitting.
                    class_counts = y.value_counts()

                    eligible_classes = class_counts[
                        class_counts >= 2
                    ].index

                    eligible_mask = y.isin(eligible_classes)

                    X = X.loc[eligible_mask]
                    y = y.loc[eligible_mask]

                    if y.nunique() < 2:
                        raise ValueError(
                            "At least two classes with sufficient examples "
                            "are required."
                        )

                    class_counts = y.value_counts()

                    if class_counts.min() < 2:
                        raise ValueError(
                            "Each retained class needs at least two examples."
                        )

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.25,
                        random_state=42,
                        stratify=y,
                    )

                    # Baseline: impute using training data only.
                    baseline_imputer = SimpleImputer(
                        strategy="median"
                    )

                    X_train_base = baseline_imputer.fit_transform(
                        X_train
                    )

                    X_test_base = baseline_imputer.transform(
                        X_test
                    )

                    baseline_model = RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        class_weight="balanced",
                    )

                    baseline_model.fit(
                        X_train_base,
                        y_train,
                    )

                    baseline_predictions = baseline_model.predict(
                        X_test_base
                    )

                    baseline_accuracy = accuracy_score(
                        y_test,
                        baseline_predictions,
                    )

                    # Intervention: remove duplicate feature rows
                    # from the training set, then impute.
                    train_frame = X_train.copy()
                    train_frame["_target_"] = y_train

                    train_frame = train_frame.drop_duplicates()

                    y_train_clean = train_frame.pop("_target_")
                    X_train_clean = train_frame

                    if y_train_clean.nunique() < 2:
                        raise ValueError(
                            "Removing duplicate rows left fewer than "
                            "two training classes."
                        )

                    cleaned_imputer = SimpleImputer(
                        strategy="median"
                    )

                    X_train_clean_imputed = cleaned_imputer.fit_transform(
                        X_train_clean
                    )

                    X_test_clean_imputed = cleaned_imputer.transform(
                        X_test
                    )

                    cleaned_model = RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        class_weight="balanced",
                    )

                    cleaned_model.fit(
                        X_train_clean_imputed,
                        y_train_clean,
                    )

                    cleaned_predictions = cleaned_model.predict(
                        X_test_clean_imputed
                    )

                    cleaned_accuracy = accuracy_score(
                        y_test,
                        cleaned_predictions,
                    )

                    st.session_state["impact_results"] = {
                        "baseline_accuracy": baseline_accuracy,
                        "intervention_accuracy": cleaned_accuracy,
                        "test_rows": len(y_test),
                        "train_rows_before": len(y_train),
                        "train_rows_after": len(y_train_clean),
                    }

                except Exception as error:
                    st.error(
                        f"The experiment could not be completed: {error}"
                    )

        if "impact_results" in st.session_state:

            results = st.session_state["impact_results"]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Baseline accuracy",
                    f"{results['baseline_accuracy']:.3f}",
                )

            with col2:
                st.metric(
                    "Intervention accuracy",
                    f"{results['intervention_accuracy']:.3f}",
                    delta=(
                        results["intervention_accuracy"]
                        - results["baseline_accuracy"]
                    ),
                )

            with col3:
                st.metric(
                    "Held-out test rows",
                    results["test_rows"],
                )

            st.caption(
                "A single split and accuracy score are insufficient to "
                "establish a reliable improvement. Repeated evaluation, "
                "leakage checks, and additional metrics are needed."
            )


# --------------------------------------------------
# DOWNLOAD AUDIT REPORT
# --------------------------------------------------

st.divider()
st.header("5. Export Audit Report")

report = {
    "audit_timestamp": datetime.now().isoformat(timespec="seconds"),
    "dataset_rows": len(df),
    "dataset_columns": len(df.columns),
    "missing_cells": int(df.isna().sum().sum()),
    "duplicate_rows": duplicates,
    "columns_with_missing_values": int(
        (df.isna().sum() > 0).sum()
    ),
}

report_df = pd.DataFrame(
    list(report.items()),
    columns=["Metric", "Value"],
)

st.dataframe(report_df, use_container_width=True)

csv_report = report_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download audit report (CSV)",
    data=csv_report,
    file_name="datasentinel_audit_report.csv",
    mime="text/csv",
)

st.success("DataSentinel AI audit is complete.")