import io
import json
import warnings
from datetime import datetime
from risk_engine import calculate_dataset_risk

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_selection import mutual_info_classif


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DataSentinel AI | Version 2",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
    }

    [data-testid="stMetric"] {
        background-color: #1b2330;
        border: 1px solid #303846;
        padding: 15px;
        border-radius: 12px;
    }

    h1, h2, h3 {
        color: #e8eef7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def prepare_column_names(df):
    result = df.copy()
    result.columns = [
        str(column).strip()
        for column in result.columns
    ]

    if result.columns.duplicated().any():
        raise ValueError(
            "The dataset has duplicate column names. "
            "Rename them before continuing."
        )

    return result


def find_identifier_candidates(df):
    findings = []

    for column in df.columns:
        series = df[column].dropna()

        if len(series) < 2:
            continue

        unique_ratio = series.nunique() / len(series)
        name = str(column).lower()

        name_suggests_id = any(
            token in name
            for token in [
                "id",
                "identifier",
                "uuid",
                "record_number",
                "serial_number",
            ]
        )

        if name_suggests_id and unique_ratio >= 0.5:
            findings.append({
                "Column": column,
                "Unique ratio": round(unique_ratio, 3),
                "Reason": (
                    "Column name suggests an identifier and "
                    "the column has many distinct values."
                ),
                "Confidence": "Candidate only",
            })

        elif (
            unique_ratio >= 0.98
            and len(series) >= 20
            and not pd.api.types.is_float_dtype(series)
        ):
            findings.append({
                "Column": column,
                "Unique ratio": round(unique_ratio, 3),
                "Reason": (
                    "Very high uniqueness may indicate a record "
                    "identifier or nearly unique field."
                ),
                "Confidence": "Candidate only",
            })

    return pd.DataFrame(findings)


def profile_dataset(df):
    rows, columns = df.shape

    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    missing_pct = (
        missing_cells / (rows * columns) * 100
        if rows * columns
        else 0
    )

    constant_columns = [
        col for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]

    return {
        "Rows": rows,
        "Columns": columns,
        "Missing cells": missing_cells,
        "Missing percentage": round(missing_pct, 2),
        "Duplicate rows": duplicate_rows,
        "Constant columns": len(constant_columns),
    }


def build_preprocessor(X):
    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        col for col in X.columns
        if col not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ])

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def evaluate_classifier(X, y, seeds):
    results = []

    for seed in seeds:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=seed,
                stratify=y,
            )

            model = Pipeline([
                ("preprocessor", build_preprocessor(X_train)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=150,
                        random_state=seed,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ])

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            results.append({
                "Seed": seed,
                "Accuracy": accuracy_score(
                    y_test,
                    predictions,
                ),
                "Balanced accuracy": balanced_accuracy_score(
                    y_test,
                    predictions,
                ),
                "Precision macro": precision_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "Recall macro": recall_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "F1 macro": f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "Train rows": len(X_train),
                "Test rows": len(X_test),
            })

        except Exception as error:
            warnings.warn(
                f"Seed {seed} failed: {error}"
            )

    return pd.DataFrame(results)


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ DataSentinel AI")
st.subheader(
    "Version 2 — Dataset Risk Intelligence & Impact Laboratory"
)

st.write(
    "Audit data quality, investigate suspicious predictors, "
    "and experimentally evaluate the impact of dataset interventions."
)

st.caption(
    "Research prototype. Statistical warnings are hypotheses to investigate, "
    "not automatic proof of data defects or causal relationships."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("DataSentinel Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV dataset",
    type=["csv"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Research configuration")

number_of_seeds = st.sidebar.slider(
    "Evaluation repetitions",
    min_value=3,
    max_value=10,
    value=5,
)

st.sidebar.caption(
    "Each repetition uses a different random train/test split. "
    "These scores are useful for sensitivity analysis but are not "
    "independent estimates of generalisation performance."
)


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔍 Risk intelligence")
        st.write(
            "Investigate missingness, duplicates, suspicious identifiers, "
            "and possible target leakage."
        )

    with col2:
        st.markdown("### 🧪 Impact laboratory")
        st.write(
            "Compare baseline performance with controlled data interventions."
        )

    with col3:
        st.markdown("### 📊 Research evidence")
        st.write(
            "Review multiple metrics and repeated experiment results."
        )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:
    uploaded_file.seek(0)

    df = pd.read_csv(uploaded_file)
    df = prepare_column_names(df)

except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()


if df.empty:
    st.error("The uploaded dataset has no rows.")
    st.stop()

if len(df.columns) < 1:
    st.error("The dataset has no columns.")
    st.stop()

st.success("Dataset loaded successfully.")
# =========================================================
# DATASET RISK SCORE — VERSION 3
# =========================================================

st.markdown("---")
st.header("Dataset Risk Intelligence")

try:
    risk_result = calculate_dataset_risk(df)

    # Display the overall risk score
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Dataset Risk Score",
            f"{risk_result['risk_score']}/100"
        )

    with col2:
        st.metric(
            "Dataset Risk Level",
            risk_result["risk_level"]
        )

    # Display supporting statistics
    st.subheader("Risk Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric(
            "Missing Cells",
            risk_result["missing_cells"]
        )

    with summary_col2:
        st.metric(
            "Duplicate Rows",
            risk_result["duplicate_rows"]
        )

    with summary_col3:
        st.metric(
            "Missing Data",
            f"{risk_result['missing_percentage']:.2f}%"
        )

    # Display detected risks and recommendations
    st.subheader("Detected Risks and Recommendations")

    st.dataframe(
        risk_result["issues"],
        width="stretch",
        hide_index=True
    )

    st.caption(
        "The risk score is a heuristic screening indicator, "
        "not a validated probability of model failure. "
        "Review each warning in the context of your dataset."
    )

except Exception as error:
    st.error(f"Dataset risk analysis failed: {error}")

# =========================================================
# END OF DATASET RISK SCORE
# =========================================================

# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("1. Dataset Overview")

profile = profile_dataset(df)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Rows", f"{profile['Rows']:,}")

with c2:
    st.metric("Columns", f"{profile['Columns']:,}")

with c3:
    st.metric("Missing cells", f"{profile['Missing cells']:,}")

with c4:
    st.metric("Duplicate rows", f"{profile['Duplicate rows']:,}")

with c5:
    st.metric("Missing data", f"{profile['Missing percentage']:.2f}%")


with st.expander("Preview uploaded dataset", expanded=True):
    st.dataframe(
        df.head(100),
        width="stretch",
    )


# ============================================================
# DATA QUALITY
# ============================================================

st.divider()
st.header("2. Data Quality Investigation")

missing_report = pd.DataFrame({
    "Column": df.columns,
    "Data type": df.dtypes.astype(str).values,
    "Missing count": df.isna().sum().values,
    "Missing percentage": (
        df.isna().mean().values * 100
    ).round(2),
    "Unique values": [
        df[col].nunique(dropna=True)
        for col in df.columns
    ],
})

st.markdown("#### Column-level quality report")

st.dataframe(
    missing_report.sort_values(
        "Missing percentage",
        ascending=False,
    ),
    width="stretch",
)

if profile["Duplicate rows"] > 0:
    st.warning(
        f"{profile['Duplicate rows']} completely duplicated rows found. "
        "Check whether they represent legitimate repeated observations."
    )
else:
    st.success("No completely duplicated rows found.")


constant_columns = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]

if constant_columns:
    st.warning(
        "Potential constant columns: "
        + ", ".join(map(str, constant_columns))
    )


# ============================================================
# OUTLIERS
# ============================================================

st.divider()
st.header("3. Statistical Outlier Investigation")

numeric_df = df.select_dtypes(include=np.number)

if numeric_df.empty:
    st.info("No numeric columns available.")

else:
    selected_numeric = st.selectbox(
        "Choose a numeric column",
        numeric_df.columns,
    )

    values = numeric_df[selected_numeric].dropna()

    if len(values) >= 4:
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            mask = (
                (values < q1) | (values > q3)
            )
        else:
            mask = (
                (values < q1 - 1.5 * iqr)
                | (values > q3 + 1.5 * iqr)
            )

        outlier_count = int(mask.sum())

        a, b = st.columns(2)

        with a:
            st.metric(
                "Potential outliers",
                outlier_count,
            )

        with b:
            st.metric(
                "Flagged percentage",
                f"{outlier_count / len(values) * 100:.2f}%",
            )

        fig = px.box(
            df,
            y=selected_numeric,
            points="outliers",
            title=f"Distribution: {selected_numeric}",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.caption(
            "IQR outlier detection is a screening method. "
            "An unusual value is not necessarily incorrect."
        )

    else:
        st.info(
            "At least four non-missing values are required."
        )


# ============================================================
# TARGET SELECTION
# ============================================================

st.divider()
st.header("4. Target & Feature Risk Intelligence")

target_column = st.selectbox(
    "Select the classification target",
    ["None"] + list(df.columns),
)

identifier_candidates = find_identifier_candidates(df)

st.markdown("#### Potential identifier columns")

if identifier_candidates.empty:
    st.info("No obvious identifier candidates detected.")
else:
    st.dataframe(
        identifier_candidates,
        width="stretch",
    )

    st.caption(
        "Identifier detection is heuristic. Review each candidate manually."
    )


if target_column != "None":

    labelled = df.dropna(
        subset=[target_column]
    ).copy()

    st.markdown("#### Target distribution")

    class_counts = labelled[target_column].value_counts()

    if len(class_counts) < 2:
        st.warning(
            "The selected target has fewer than two classes."
        )

    else:
        fig = px.bar(
            x=class_counts.index.astype(str),
            y=class_counts.values,
            labels={
                "x": target_column,
                "y": "Records",
            },
            title="Target class distribution",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        ratio = (
            class_counts.max() / class_counts.min()
            if class_counts.min() > 0
            else np.inf
        )

        st.metric(
            "Majority/minority ratio",
            f"{ratio:.2f}:1",
        )

        if ratio >= 3:
            st.warning(
                "Potential class imbalance detected. "
                "Consider class-specific metrics."
            )

    st.markdown("#### Possible target leakage screening")

    st.write(
        "The following screening identifies predictors with very high "
        "association to the target. Strong association can be legitimate "
        "and does not by itself establish leakage."
    )

    predictors = labelled.drop(
        columns=[target_column]
    )

    leakage_findings = []

    target_cardinality = labelled[target_column].nunique()

    for column in predictors.columns:

        series = predictors[column]

        if series.nunique(dropna=True) < 2:
            continue

        # A predictor identical to the target is a strong warning.
        paired = pd.DataFrame({
            "feature": series,
            "target": labelled[target_column],
        }).dropna()

        if len(paired) < 5:
            continue

        if paired["feature"].astype(str).equals(
            paired["target"].astype(str)
        ):
            leakage_findings.append({
                "Feature": column,
                "Finding": (
                    "Feature values match target values on "
                    "observed rows."
                ),
                "Severity": "High review priority",
            })

    if leakage_findings:
        st.dataframe(
            pd.DataFrame(leakage_findings),
            width="stretch",
        )
    else:
        st.info(
            "No exact target-copy predictors detected by this screening. "
            "This does not rule out indirect or temporal leakage."
        )


# ============================================================
# EXPERIMENT LABORATORY
# ============================================================

st.divider()
st.header("5. Data Impact Laboratory")

st.write(
    "Compare baseline evaluation with an intervention that removes "
    "duplicate training records. Both approaches use the same held-out "
    "test partition for each seed."
)

st.warning(
    "This experiment evaluates classification only. It does not establish "
    "causality, and duplicate removal is not always appropriate."
)


if target_column == "None":

    st.info("Select a target column above to enable experiments.")

else:

    run_experiment = st.button(
        "Run repeated impact experiment",
        type="primary",
    )

    if run_experiment:

        labelled = df.dropna(
            subset=[target_column]
        ).copy()

        y_raw = labelled[target_column].astype(str)

        if y_raw.nunique() < 2:
            st.error("The target must contain at least two classes.")

        elif y_raw.value_counts().min() < 4:
            st.error(
                "Each class needs at least four labelled observations "
                "for this preliminary repeated-split experiment."
            )

        else:

            X_raw = labelled.drop(
                columns=[target_column]
            ).copy()

            # Avoid obvious identifier leakage.
            candidate_ids = set(
                identifier_candidates["Column"]
                if not identifier_candidates.empty
                else []
            )

            removed_ids = [
                col for col in X_raw.columns
                if col in candidate_ids
            ]

            X_raw = X_raw.drop(
                columns=removed_ids,
                errors="ignore",
            )

            # Drop columns with no observed values.
            X_raw = X_raw.dropna(
                axis=1,
                how="all",
            )

            # Prevent unsupported objects and infinite numeric values.
            for col in X_raw.select_dtypes(include=np.number):
                X_raw[col] = X_raw[col].replace(
                    [np.inf, -np.inf],
                    np.nan,
                )

            if X_raw.shape[1] == 0:
                st.error(
                    "No usable predictors remain after removing "
                    "all-missing and candidate identifier columns."
                )

            else:
                seeds = list(
                    range(42, 42 + number_of_seeds)
                )

                baseline_results = []
                intervention_results = []

                progress = st.progress(0)

                for index, seed in enumerate(seeds):

                    try:
                        X_train, X_test, y_train, y_test = (
                            train_test_split(
                                X_raw,
                                y_raw,
                                test_size=0.25,
                                random_state=seed,
                                stratify=y_raw,
                            )
                        )

                        baseline_model = Pipeline([
                            (
                                "preprocessor",
                                build_preprocessor(X_train),
                            ),
                            (
                                "classifier",
                                RandomForestClassifier(
                                    n_estimators=150,
                                    random_state=seed,
                                    class_weight="balanced",
                                    n_jobs=-1,
                                ),
                            ),
                        ])

                        baseline_model.fit(
                            X_train,
                            y_train,
                        )

                        baseline_predictions = (
                            baseline_model.predict(X_test)
                        )

                        baseline_results.append({
                            "Seed": seed,
                            "Accuracy": accuracy_score(
                                y_test,
                                baseline_predictions,
                            ),
                            "Balanced accuracy": (
                                balanced_accuracy_score(
                                    y_test,
                                    baseline_predictions,
                                )
                            ),
                            "Precision macro": precision_score(
                                y_test,
                                baseline_predictions,
                                average="macro",
                                zero_division=0,
                            ),
                            "Recall macro": recall_score(
                                y_test,
                                baseline_predictions,
                                average="macro",
                                zero_division=0,
                            ),
                            "F1 macro": f1_score(
                                y_test,
                                baseline_predictions,
                                average="macro",
                                zero_division=0,
                            ),
                        })

                        # Intervention: remove duplicate feature rows
                        # from the training set only.
                        training_frame = X_train.copy()

                        training_frame["_target_"] = (
                            y_train.values
                        )

                        training_frame = (
                            training_frame.drop_duplicates()
                        )

                        y_train_intervention = (
                            training_frame.pop("_target_")
                        )

                        X_train_intervention = training_frame

                        if y_train_intervention.nunique() < 2:
                            raise ValueError(
                                "Intervention removed too many "
                                "training examples from a class."
                            )

                        intervention_model = Pipeline([
                            (
                                "preprocessor",
                                build_preprocessor(
                                    X_train_intervention
                                ),
                            ),
                            (
                                "classifier",
                                RandomForestClassifier(
                                    n_estimators=150,
                                    random_state=seed,
                                    class_weight="balanced",
                                    n_jobs=-1,
                                ),
                            ),
                        ])

                        intervention_model.fit(
                            X_train_intervention,
                            y_train_intervention,
                        )

                        intervention_predictions = (
                            intervention_model.predict(X_test)
                        )

                        intervention_results.append({
                            "Seed": seed,
                            "Accuracy": accuracy_score(
                                y_test,
                                intervention_predictions,
                            ),
                            "Balanced accuracy": (
                                balanced_accuracy_score(
                                    y_test,
                                    intervention_predictions,
                                )
                            ),
                            "Precision macro": precision_score(
                                y_test,
                                intervention_predictions,
                                average="macro",
                                zero_division=0,
                            ),
                            "Recall macro": recall_score(
                                y_test,
                                intervention_predictions,
                                average="macro",
                                zero_division=0,
                            ),
                            "F1 macro": f1_score(
                                y_test,
                                intervention_predictions,
                                average="macro",
                                zero_division=0,
                            ),
                            "Training rows before": len(X_train),
                            "Training rows after": len(
                                X_train_intervention
                            ),
                        })

                    except Exception as error:
                        st.warning(
                            f"Seed {seed} failed: {error}"
                        )

                    progress.progress(
                        (index + 1) / len(seeds)
                    )

                baseline_df = pd.DataFrame(
                    baseline_results
                )

                intervention_df = pd.DataFrame(
                    intervention_results
                )

                if (
                    baseline_df.empty
                    or intervention_df.empty
                ):
                    st.error(
                        "No successful experiments completed. "
                        "Check class counts and feature types."
                    )

                else:
                    metrics = [
                        "Accuracy",
                        "Balanced accuracy",
                        "Precision macro",
                        "Recall macro",
                        "F1 macro",
                    ]

                    comparison = []

                    for metric in metrics:

                        baseline_mean = (
                            baseline_df[metric].mean()
                        )

                        intervention_mean = (
                            intervention_df[metric].mean()
                        )

                        comparison.append({
                            "Metric": metric,
                            "Baseline mean": baseline_mean,
                            "Intervention mean": intervention_mean,
                            "Difference": (
                                intervention_mean - baseline_mean
                            ),
                        })

                    comparison_df = pd.DataFrame(
                        comparison
                    )

                    st.session_state["ds_baseline"] = (
                        baseline_df
                    )

                    st.session_state["ds_intervention"] = (
                        intervention_df
                    )

                    st.session_state["ds_comparison"] = (
                        comparison_df
                    )

                    st.success(
                        "Repeated impact experiment completed."
                    )


if "ds_comparison" in st.session_state:

    st.markdown("#### Experimental comparison")

    comparison_df = st.session_state["ds_comparison"]

    st.dataframe(
        comparison_df.style.format({
            "Baseline mean": "{:.4f}",
            "Intervention mean": "{:.4f}",
            "Difference": "{:+.4f}",
        }),
        width="stretch",
    )

    fig = px.bar(
        comparison_df,
        x="Metric",
        y="Difference",
        title="Measured change after intervention",
    )

    fig.add_hline(y=0)

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.markdown("#### Per-seed results")

    st.write("Baseline")

    st.dataframe(
        st.session_state["ds_baseline"],
        width="stretch",
    )

    st.write("Intervention")

    st.dataframe(
        st.session_state["ds_intervention"],
        width="stretch",
    )

    st.caption(
        "A positive difference means the intervention performed better "
        "on that metric in these runs. Results may depend on the split, "
        "dataset, target definition, and evaluation protocol."
    )


# ============================================================
# REPORT EXPORT
# ============================================================

st.divider()
st.header("6. Research Report Export")

report = {
    "project": "DataSentinel AI",
    "version": "2.0",
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "dataset_rows": int(df.shape[0]),
    "dataset_columns": int(df.shape[1]),
    "missing_cells": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "target_column": (
        target_column
        if target_column != "None"
        else None
    ),
    "identifier_candidates": (
        identifier_candidates.to_dict(orient="records")
    ),
}

if "ds_comparison" in st.session_state:

    report["experiment_comparison"] = (
        st.session_state["ds_comparison"].to_dict(
            orient="records"
        )
    )

    report["baseline_results"] = (
        st.session_state["ds_baseline"].to_dict(
            orient="records"
        )
    )

    report["intervention_results"] = (
        st.session_state["ds_intervention"].to_dict(
            orient="records"
        )
    )

report_json = json.dumps(
    report,
    indent=2,
    default=str,
)

st.download_button(
    "Download research report (JSON)",
    data=report_json.encode("utf-8"),
    file_name="datasentinel_research_report.json",
    mime="application/json",
)

st.download_button(
    "Download quality report (CSV)",
    data=missing_report.to_csv(index=False).encode("utf-8"),
    file_name="datasentinel_quality_report.csv",
    mime="text/csv",
)

st.success("DataSentinel AI Version 2 is ready.")
