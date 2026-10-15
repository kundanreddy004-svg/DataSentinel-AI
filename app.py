import io
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from risk_engine import calculate_dataset_risk
from drift_detector import detect_dataset_drift
from leakage_detector import detect_target_leakage
from pdf_report import generate_audit_pdf


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
    .stApp { background-color: #0e1117; }
    [data-testid="stMetric"] {
        background-color: #1b2330;
        border: 1px solid #303846;
        padding: 15px;
        border-radius: 12px;
    }
    h1, h2, h3 { color: #e8eef7; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def prepare_column_names(df):
    result = df.copy()
    result.columns = [str(column).strip() for column in result.columns]
    if result.columns.duplicated().any():
        raise ValueError(
            "The dataset has duplicate column names. Rename them before continuing."
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
            for token in (
                "id", "identifier", "uuid", "record_number", "serial_number"
            )
        )

        if name_suggests_id and unique_ratio >= 0.5:
            findings.append({
                "Column": column,
                "Unique ratio": round(unique_ratio, 3),
                "Reason": (
                    "Column name suggests an identifier and the column has "
                    "many distinct values."
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
                    "Very high uniqueness may indicate a record identifier "
                    "or nearly unique field."
                ),
                "Confidence": "Candidate only",
            })

    return pd.DataFrame(
        findings,
        columns=["Column", "Unique ratio", "Reason", "Confidence"],
    )


def profile_dataset(df):
    rows, columns = df.shape
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    missing_pct = (
        missing_cells / (rows * columns) * 100
        if rows * columns else 0
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
        "Constant column names": constant_columns,
    }


def build_preprocessor(X):
    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical_columns = [
        col for col in X.columns if col not in numeric_columns
    ]

    transformers = []
    if numeric_columns:
        transformers.append((
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            numeric_columns,
        ))

    if categorical_columns:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )),
            ]),
            categorical_columns,
        ))

    if not transformers:
        raise ValueError("No usable feature columns remain.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_classifier(seed, X_train):
    return Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])


def safe_records(frame):
    if not isinstance(frame, pd.DataFrame):
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


# ============================================================
# HEADER AND SIDEBAR
# ============================================================

st.title("🛡️ DataSentinel AI")
st.subheader("Version 2 — Dataset Risk Intelligence & Impact Laboratory")
st.write(
    "Audit data quality, investigate suspicious predictors, and "
    "experimentally evaluate dataset interventions."
)
st.caption(
    "Research prototype. Statistical warnings are investigation signals, "
    "not proof of defects or causal relationships."
)
st.divider()

st.sidebar.title("DataSentinel Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.markdown("### Research configuration")
number_of_seeds = st.sidebar.slider(
    "Evaluation repetitions", min_value=3, max_value=10, value=5
)
st.sidebar.caption(
    "Repeated random splits are useful for sensitivity analysis, but are "
    "not independent estimates of generalisation performance."
)


# ============================================================
# DRIFT DETECTION
# ============================================================

st.divider()
st.header("📊 Dataset Drift Detection")
st.write(
    "Compare a reference dataset with a current dataset to identify "
    "potential changes in numerical and categorical distributions."
)
st.info(
    "Use comparable datasets from different time periods or pipelines. "
    "Drift flags are investigation signals, not proof of a defect."
)

reference_file = st.file_uploader(
    "Upload Reference Dataset (baseline)",
    type=["csv"],
    key="drift_reference_file",
)
current_file = st.file_uploader(
    "Upload Current Dataset (to compare)",
    type=["csv"],
    key="drift_current_file",
)

if reference_file is not None and current_file is not None:
    try:
        reference_file.seek(0)
        current_file.seek(0)
        reference_df = prepare_column_names(pd.read_csv(reference_file))
        current_df = prepare_column_names(pd.read_csv(current_file))

        if reference_df.empty or current_df.empty:
            st.error("Both CSV files must contain at least one data row.")
        else:
            left, right = st.columns(2)
            with left:
                st.subheader("Reference dataset")
                st.metric("Rows", f"{len(reference_df):,}")
                st.metric("Columns", f"{len(reference_df.columns):,}")
                with st.expander("Preview reference data"):
                    st.dataframe(reference_df.head(10), width="stretch")
            with right:
                st.subheader("Current dataset")
                st.metric("Rows", f"{len(current_df):,}")
                st.metric("Columns", f"{len(current_df.columns):,}")
                with st.expander("Preview current data"):
                    st.dataframe(current_df.head(10), width="stretch")

            shared_columns = [
                c for c in reference_df.columns if c in current_df.columns
            ]
            reference_only = [
                c for c in reference_df.columns if c not in current_df.columns
            ]
            current_only = [
                c for c in current_df.columns if c not in reference_df.columns
            ]
            st.caption(
                f"Shared columns: {len(shared_columns)} | "
                f"Reference-only: {len(reference_only)} | "
                f"Current-only: {len(current_only)}"
            )

            if not shared_columns:
                st.error(
                    "The datasets have no matching column names. Rename columns "
                    "so they can be compared."
                )
            elif st.button("🔍 Detect Dataset Drift", key="detect_dataset_drift_button"):
                with st.spinner("Comparing dataset distributions..."):
                    drift_result = detect_dataset_drift(
                        reference_df, current_df
                    )
                st.session_state["ds_drift_result"] = drift_result
                st.session_state["ds_drift_reference_name"] = reference_file.name
                st.session_state["ds_drift_current_name"] = current_file.name
                st.session_state["ds_drift_reference_only"] = reference_only
                st.session_state["ds_drift_current_only"] = current_only

            drift_result_display = st.session_state.get("ds_drift_result")
            if isinstance(drift_result_display, dict):
                drift_report_display = drift_result_display.get(
                    "report", pd.DataFrame()
                )
                st.subheader("Drift Detection Results")
                if drift_report_display.empty:
                    st.warning(
                        "The detector did not produce a report. Check data types "
                        "and shared columns."
                    )
                else:
                    if "Potential drift" in drift_report_display.columns:
                        flags = drift_report_display[
                            drift_report_display["Potential drift"]
                            .astype(str).str.lower().eq("review recommended")
                        ]
                        no_flags = int(
                            drift_report_display["Potential drift"]
                            .astype(str).str.lower().eq("not detected").sum()
                        )
                    else:
                        flags = pd.DataFrame()
                        no_flags = 0

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Columns reported", len(drift_report_display))
                    m2.metric("Potential drift", len(flags))
                    m3.metric("Not flagged", no_flags)
                    if not flags.empty:
                        st.warning(
                            "Potential distribution changes were detected. "
                            "Review flagged columns and investigate their cause."
                        )
                    else:
                        st.success(
                            "No columns were flagged by the configured screening "
                            "tests. This does not prove equivalence."
                        )
                    st.dataframe(
                        drift_report_display,
                        width="stretch",
                        hide_index=True,
                    )
                    st.download_button(
                        "Download drift report (CSV)",
                        data=drift_report_display.to_csv(index=False).encode("utf-8"),
                        file_name="datasentinel_drift_report.csv",
                        mime="text/csv",
                        key="download_drift_report_csv",
                    )

                ref_only_saved = st.session_state.get(
                    "ds_drift_reference_only", reference_only
                )
                cur_only_saved = st.session_state.get(
                    "ds_drift_current_only", current_only
                )
                if ref_only_saved:
                    st.warning(
                        "Columns only in reference dataset: "
                        + ", ".join(map(str, ref_only_saved))
                    )
                if cur_only_saved:
                    st.warning(
                        "Columns only in current dataset: "
                        + ", ".join(map(str, cur_only_saved))
                    )
                st.caption(
                    "Statistical tests can be sensitive to sample size, missing "
                    "values, and multiple comparisons."
                )
    except Exception as error:
        st.error(f"Unable to compare the datasets: {error}")
else:
    st.caption(
        "Upload both CSV files to preview the datasets and enable drift detection."
    )


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔍 Risk intelligence")
        st.write("Investigate missingness, duplicates, identifiers, and possible leakage.")
    with col2:
        st.markdown("### 🧪 Impact laboratory")
        st.write("Compare baseline performance with controlled data interventions.")
    with col3:
        st.markdown("### 📄 Audit reports")
        st.write("Export available audit findings to a PDF report.")
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:
    uploaded_file.seek(0)
    df = prepare_column_names(pd.read_csv(uploaded_file))
except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()

if df.empty or len(df.columns) < 1:
    st.error("The uploaded dataset must contain rows and columns.")
    st.stop()

st.success("Dataset loaded successfully.")
st.session_state["ds_dataset_name"] = uploaded_file.name


# ============================================================
# RISK INTELLIGENCE
# ============================================================

st.divider()
st.header("1. Dataset Risk Intelligence")

risk_result = None
try:
    risk_result = calculate_dataset_risk(df)
    st.session_state["ds_risk_result"] = risk_result

    c1, c2 = st.columns(2)
    c1.metric("Dataset Risk Score", f"{risk_result['risk_score']}/100")
    c2.metric("Dataset Risk Level", str(risk_result["risk_level"]))

    a, b, c = st.columns(3)
    a.metric("Missing Cells", risk_result["missing_cells"])
    b.metric("Duplicate Rows", risk_result["duplicate_rows"])
    b_pct = risk_result.get("missing_percentage", None)
    if b_pct is None:
        missing_pct = (
            df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
            if df.size else 0
        )
    else:
        missing_pct = b_pct
    c.metric("Missing Data", f"{missing_pct:.2f}%")

    st.subheader("Detected Risks and Recommendations")
    issues = risk_result.get("issues")
    if isinstance(issues, pd.DataFrame):
        st.dataframe(issues, width="stretch", hide_index=True)
    elif issues:
        st.write(issues)
    else:
        st.info("No issue details were returned by the risk engine.")

    st.caption(
        "The risk score is a heuristic screening indicator, not a validated "
        "probability of model failure."
    )
except Exception as error:
    st.session_state.pop("ds_risk_result", None)
    st.error(f"Dataset risk analysis failed: {error}")


# ============================================================
# AI-POWERED AUDIT EXPLANATIONS
# ============================================================

st.divider()
st.header("AI-Powered Audit Explanations")
st.caption(
    "Rule-based explanations translate audit findings into practical next steps. "
    "These are screening signals, not proof of defects or causal relationships."
)

with st.expander("Dataset Risk Explanation", expanded=True):
    saved_risk = st.session_state.get("ds_risk_result")
    if isinstance(saved_risk, dict):
        score = saved_risk.get("risk_score", "N/A")
        level = saved_risk.get("risk_level", "Unknown")
        missing_cells = int(saved_risk.get("missing_cells", int(df.isna().sum().sum())))
        duplicate_rows = int(saved_risk.get("duplicate_rows", int(df.duplicated().sum())))
        st.write(
            f"The dataset contains {len(df):,} rows and {len(df.columns):,} columns. "
            f"Its reported risk score is {score}/100, with a risk level of '{level}'. "
            f"The analysis identified {missing_cells:,} missing cells and {duplicate_rows:,} duplicate rows."
        )
        st.markdown("**Recommended Actions**")
        actions = []
        if missing_cells:
            actions.append("Investigate missingness by column and determine whether imputation or removal is justified.")
        if duplicate_rows:
            actions.append("Review duplicate records before removing them; confirm whether they are valid repeated observations.")
        issues = saved_risk.get("issues")
        if isinstance(issues, pd.DataFrame) and not issues.empty:
            actions.append("Review the detailed risk findings below and prioritize issues that could affect downstream analysis or model performance.")
        if not actions:
            actions.append("No missing cells or duplicate rows were detected by these checks; review the detailed findings and validate domain-specific constraints.")
        for action in actions:
            st.markdown(f"- {action}")
    else:
        st.info("Dataset risk results are not available yet.")

with st.expander("Missing Values Explanation"):
    missing_count = int(df.isna().sum().sum())
    missing_pct = (missing_count / df.size * 100) if df.size else 0.0
    st.write(f"The dataset has {missing_count:,} missing cells ({missing_pct:.2f}% of all cells).")
    if missing_count:
        st.markdown("**Recommended Actions**")
        st.markdown("- Inspect missingness by column and, where relevant, by target class or time period.")
        st.markdown("- Choose deletion or imputation based on the data-generating process; fit imputation only on training data during model evaluation.")
    else:
        st.success("No missing values were detected in the uploaded dataset.")

with st.expander("Duplicate Rows Explanation"):
    duplicate_count = int(df.duplicated().sum())
    st.write(f"The dataset contains {duplicate_count:,} completely duplicated rows.")
    if duplicate_count:
        st.markdown("**Recommended Actions**")
        st.markdown("- Inspect duplicate records and confirm whether they represent accidental repeats or legitimate repeated measurements.")
        st.markdown("- If deduplication is justified, document the rule and apply it consistently before splitting data for model evaluation.")
    else:
        st.success("No completely duplicated rows were detected.")

with st.expander("Target Leakage Explanation"):
    leakage_report = st.session_state.get("ds_leakage_report")
    if isinstance(leakage_report, pd.DataFrame) and not leakage_report.empty:
        st.write("Target-leakage screening has returned findings. A flagged predictor may encode the target directly or indirectly; confirm its meaning and when it becomes available in the real workflow.")
        st.dataframe(leakage_report, width="stretch", hide_index=True)
        st.markdown("**Recommended Actions**")
        st.markdown("- Investigate predictors marked as exact copies or perfect mappings of the target.")
        st.markdown("- Remove a predictor only when domain review confirms it leaks unavailable target information; rerun evaluation with a leakage-safe pipeline.")
    elif isinstance(leakage_report, pd.DataFrame):
        st.success("The leakage detector did not flag any predictors in its returned report.")
    else:
        st.info("Run target-leakage detection in the dashboard to generate this explanation.")

with st.expander("Dataset Drift Explanation"):
    drift_result = st.session_state.get("ds_drift_result")
    if isinstance(drift_result, dict):
        drift_report = drift_result.get("report")
        if isinstance(drift_report, pd.DataFrame) and not drift_report.empty:
            st.write("The drift detector compared the reference and current datasets. Flagged distribution changes are investigation signals; they do not automatically mean that model performance has degraded.")
            st.dataframe(drift_report, width="stretch", hide_index=True)
            st.markdown("**Recommended Actions**")
            st.markdown("- Verify that the reference and current datasets use consistent schemas, units, collection methods, and time windows.")
            st.markdown("- Investigate flagged features, check for upstream pipeline changes, and evaluate model performance on appropriately labeled recent data.")
        else:
            st.write("A drift comparison has been run, but no detailed feature report is available to display.")
        ref_only = drift_result.get("reference_only_columns", [])
        cur_only = drift_result.get("current_only_columns", [])
        if ref_only or cur_only:
            st.warning(f"Schema differences — reference-only columns: {ref_only or 'None'}; current-only columns: {cur_only or 'None'}.")
    else:
        st.info("Run Dataset Drift Detection in the dashboard to generate this explanation.")


# ============================================================
# DATASET OVERVIEW AND QUALITY
# ============================================================

st.header("2. Dataset Overview")
profile = profile_dataset(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{profile['Rows']:,}")
c2.metric("Columns", f"{profile['Columns']:,}")
c3.metric("Missing cells", f"{profile['Missing cells']:,}")
c4.metric("Duplicate rows", f"{profile['Duplicate rows']:,}")
c5.metric("Missing data", f"{profile['Missing percentage']:.2f}%")

with st.expander("Preview uploaded dataset", expanded=True):
    st.dataframe(df.head(100), width="stretch")

st.divider()
st.header("3. Data Quality Investigation")
missing_report = pd.DataFrame({
    "Column": df.columns,
    "Data type": df.dtypes.astype(str).values,
    "Missing count": df.isna().sum().values,
    "Missing percentage": (df.isna().mean().values * 100).round(2),
    "Unique values": [df[col].nunique(dropna=True) for col in df.columns],
})
st.session_state["ds_missing_report"] = missing_report
st.dataframe(
    missing_report.sort_values("Missing percentage", ascending=False),
    width="stretch",
    hide_index=True,
)

if profile["Duplicate rows"] > 0:
    st.warning(
        f"{profile['Duplicate rows']} completely duplicated rows found. "
        "Check whether they are legitimate repeated observations."
    )
else:
    st.success("No completely duplicated rows found.")

constant_columns = profile["Constant column names"]
if constant_columns:
    st.warning("Potential constant columns: " + ", ".join(map(str, constant_columns)))


# ============================================================
# OUTLIER INVESTIGATION
# ============================================================

st.divider()
st.header("4. Statistical Outlier Investigation")
numeric_df = df.select_dtypes(include=np.number)

if numeric_df.empty:
    st.info("No numeric columns available.")
else:
    selected_numeric = st.selectbox("Choose a numeric column", numeric_df.columns)
    values = numeric_df[selected_numeric].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(values) >= 4:
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_mask = (values < q1) | (values > q3)
        else:
            outlier_mask = (
                (values < q1 - 1.5 * iqr)
                | (values > q3 + 1.5 * iqr)
            )
        outlier_count = int(outlier_mask.sum())
        a, b = st.columns(2)
        a.metric("Potential outliers", outlier_count)
        b.metric(
            "Flagged percentage",
            f"{outlier_count / len(values) * 100:.2f}%",
        )
        fig = px.box(
            df, y=selected_numeric, points="outliers",
            title=f"Distribution: {selected_numeric}",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "IQR outlier detection is a screening method. An unusual value is "
            "not necessarily incorrect."
        )
    else:
        st.info("At least four non-missing numeric values are required.")


# ============================================================
# TARGET AND LEAKAGE SCREENING
# ============================================================

st.divider()
st.header("5. Target & Feature Risk Intelligence")
target_column = st.selectbox(
    "Select the classification target",
    ["None"] + list(df.columns),
)
st.session_state["ds_target_column"] = (
    target_column if target_column != "None" else None
)

identifier_candidates = find_identifier_candidates(df)
st.subheader("Potential identifier columns")
if identifier_candidates.empty:
    st.info("No obvious identifier candidates detected.")
else:
    st.dataframe(
        identifier_candidates, width="stretch", hide_index=True
    )
    st.caption("Identifier detection is heuristic. Review each candidate manually.")
st.session_state["ds_identifier_candidates"] = identifier_candidates

if target_column != "None":
    labelled = df.dropna(subset=[target_column]).copy()
    class_counts = labelled[target_column].value_counts()
    st.subheader("Target distribution")
    if len(class_counts) < 2:
        st.warning("The selected target has fewer than two classes.")
    else:
        fig = px.bar(
            x=class_counts.index.astype(str),
            y=class_counts.values,
            labels={"x": target_column, "y": "Records"},
            title="Target class distribution",
        )
        st.plotly_chart(fig, width="stretch")
        ratio = (
            class_counts.max() / class_counts.min()
            if class_counts.min() > 0 else np.inf
        )
        st.metric("Majority/minority ratio", f"{ratio:.2f}:1")
        if ratio >= 3:
            st.warning(
                "Potential class imbalance detected. Consider class-specific metrics."
            )

    st.subheader("Possible target leakage screening")
    try:
        leakage_report = detect_target_leakage(
            labelled, target_column=target_column
        )
        st.session_state["ds_leakage_report"] = leakage_report
        if leakage_report.empty:
            st.info("The leakage detector returned no predictor findings.")
        else:
            st.dataframe(
                leakage_report, width="stretch", hide_index=True
            )
            if "Status" in leakage_report.columns:
                flagged = leakage_report[
                    leakage_report["Status"].astype(str).str.contains(
                        "review", case=False, na=False
                    )
                ]
                if not flagged.empty:
                    st.warning(
                        f"{len(flagged)} predictor(s) require manual leakage review."
                    )
        st.caption(
            "Automated screening identifies potential risks. Strong association "
            "does not by itself prove leakage."
        )
    except Exception as error:
        st.session_state.pop("ds_leakage_report", None)
        st.error(f"Leakage detection failed: {error}")
else:
    st.session_state.pop("ds_leakage_report", None)
    st.info("Select a target column to run target distribution and leakage screening.")


# ============================================================
# EXPERIMENT LABORATORY
# ============================================================

st.divider()
st.header("6. Data Impact Laboratory")
st.write(
    "Compare baseline evaluation with an intervention that removes duplicate "
    "feature-and-target rows from the training set only."
)
st.warning(
    "This is a preliminary experiment. Duplicate removal is not always "
    "appropriate and results do not establish causality."
)

if target_column == "None":
    st.info("Select a target column above to enable experiments.")
else:
    if st.button("Run repeated impact experiment", type="primary"):
        labelled = df.dropna(subset=[target_column]).copy()
        y_raw = labelled[target_column].astype(str)
        if y_raw.nunique() < 2:
            st.error("The target must contain at least two classes.")
        elif y_raw.value_counts().min() < 4:
            st.error(
                "Each class needs at least four labelled observations for this "
                "preliminary repeated-split experiment."
            )
        else:
            X_raw = labelled.drop(columns=[target_column]).copy()
            candidate_ids = set(
                identifier_candidates["Column"].tolist()
                if not identifier_candidates.empty else []
            )
            removed_ids = [col for col in X_raw.columns if col in candidate_ids]
            X_raw = X_raw.drop(columns=removed_ids, errors="ignore")
            X_raw = X_raw.dropna(axis=1, how="all")
            for col in X_raw.select_dtypes(include=np.number).columns:
                X_raw[col] = X_raw[col].replace([np.inf, -np.inf], np.nan)

            if X_raw.shape[1] == 0:
                st.error(
                    "No usable predictors remain after removing all-missing "
                    "and candidate identifier columns."
                )
            else:
                seeds = list(range(42, 42 + number_of_seeds))
                baseline_results = []
                intervention_results = []
                progress = st.progress(0)

                for index, seed in enumerate(seeds):
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_raw, y_raw, test_size=0.25,
                            random_state=seed, stratify=y_raw,
                        )

                        baseline_model = make_classifier(seed, X_train)
                        baseline_model.fit(X_train, y_train)
                        base_pred = baseline_model.predict(X_test)
                        baseline_results.append({
                            "Seed": seed,
                            "Accuracy": accuracy_score(y_test, base_pred),
                            "Balanced accuracy": balanced_accuracy_score(y_test, base_pred),
                            "Precision macro": precision_score(
                                y_test, base_pred, average="macro", zero_division=0
                            ),
                            "Recall macro": recall_score(
                                y_test, base_pred, average="macro", zero_division=0
                            ),
                            "F1 macro": f1_score(
                                y_test, base_pred, average="macro", zero_division=0
                            ),
                        })

                        train_frame = X_train.copy()
                        train_frame["_ds_target_"] = y_train.to_numpy()
                        train_frame = train_frame.drop_duplicates()
                        y_train_intervention = train_frame.pop("_ds_target_")
                        X_train_intervention = train_frame

                        if y_train_intervention.nunique() < 2:
                            raise ValueError(
                                "Intervention removed too many examples from a class."
                            )

                        intervention_model = make_classifier(
                            seed, X_train_intervention
                        )
                        intervention_model.fit(
                            X_train_intervention, y_train_intervention
                        )
                        intervention_pred = intervention_model.predict(X_test)
                        intervention_results.append({
                            "Seed": seed,
                            "Accuracy": accuracy_score(y_test, intervention_pred),
                            "Balanced accuracy": balanced_accuracy_score(
                                y_test, intervention_pred
                            ),
                            "Precision macro": precision_score(
                                y_test, intervention_pred,
                                average="macro", zero_division=0
                            ),
                            "Recall macro": recall_score(
                                y_test, intervention_pred,
                                average="macro", zero_division=0
                            ),
                            "F1 macro": f1_score(
                                y_test, intervention_pred,
                                average="macro", zero_division=0
                            ),
                            "Training rows before": len(X_train),
                            "Training rows after": len(X_train_intervention),
                        })
                    except Exception as error:
                        warnings.warn(f"Seed {seed} failed: {error}")
                        st.warning(f"Seed {seed} failed: {error}")
                    progress.progress((index + 1) / len(seeds))

                baseline_df = pd.DataFrame(baseline_results)
                intervention_df = pd.DataFrame(intervention_results)
                if baseline_df.empty or intervention_df.empty:
                    st.error(
                        "No successful experiments completed. Check class counts "
                        "and feature types."
                    )
                else:
                    metrics = [
                        "Accuracy", "Balanced accuracy", "Precision macro",
                        "Recall macro", "F1 macro",
                    ]
                    comparison_df = pd.DataFrame([
                        {
                            "Metric": metric,
                            "Baseline mean": baseline_df[metric].mean(),
                            "Intervention mean": intervention_df[metric].mean(),
                            "Difference": (
                                intervention_df[metric].mean()
                                - baseline_df[metric].mean()
                            ),
                        }
                        for metric in metrics
                    ])
                    st.session_state["ds_baseline"] = baseline_df
                    st.session_state["ds_intervention"] = intervention_df
                    st.session_state["ds_comparison"] = comparison_df
                    st.success("Repeated impact experiment completed.")

if "ds_comparison" in st.session_state:
    st.subheader("Experimental comparison")
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
        comparison_df, x="Metric", y="Difference",
        title="Measured change after intervention",
    )
    fig.add_hline(y=0)
    st.plotly_chart(fig, width="stretch")
    st.markdown("#### Per-seed results")
    st.write("Baseline")
    st.dataframe(st.session_state["ds_baseline"], width="stretch")
    st.write("Intervention")
    st.dataframe(st.session_state["ds_intervention"], width="stretch")
    st.caption(
        "Positive differences indicate better performance after the intervention "
        "in these runs. Results depend on the data and evaluation protocol."
    )


# ============================================================
# JSON, CSV, AND PDF REPORT EXPORT
# ============================================================

st.divider()
st.header("7. Research Report Export")

report = {
    "project": "DataSentinel AI",
    "version": "2.0",
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "dataset_name": uploaded_file.name,
    "dataset_rows": int(df.shape[0]),
    "dataset_columns": int(df.shape[1]),
    "missing_cells": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "target_column": target_column if target_column != "None" else None,
    "identifier_candidates": safe_records(identifier_candidates),
}

current_risk = st.session_state.get("ds_risk_result")
if isinstance(current_risk, dict):
    report["risk_score"] = current_risk.get("risk_score")
    report["risk_level"] = current_risk.get("risk_level")

current_leakage = st.session_state.get("ds_leakage_report")
if isinstance(current_leakage, pd.DataFrame):
    report["leakage_findings"] = safe_records(current_leakage)

current_drift = st.session_state.get("ds_drift_result")
if isinstance(current_drift, dict):
    current_drift_report = current_drift.get("report")
    report["drift_report"] = safe_records(current_drift_report)
    report["drift_reference_dataset"] = st.session_state.get(
        "ds_drift_reference_name"
    )
    report["drift_current_dataset"] = st.session_state.get(
        "ds_drift_current_name"
    )
    report["reference_only_columns"] = current_drift.get(
        "reference_only_columns",
        st.session_state.get("ds_drift_reference_only", []),
    )
    report["current_only_columns"] = current_drift.get(
        "current_only_columns",
        st.session_state.get("ds_drift_current_only", []),
    )

if "ds_comparison" in st.session_state:
    report["experiment_comparison"] = safe_records(
        st.session_state["ds_comparison"]
    )
    report["baseline_results"] = safe_records(
        st.session_state.get("ds_baseline", pd.DataFrame())
    )
    report["intervention_results"] = safe_records(
        st.session_state.get("ds_intervention", pd.DataFrame())
    )

st.download_button(
    "Download research report (JSON)",
    data=json.dumps(report, indent=2, default=str).encode("utf-8"),
    file_name="datasentinel_research_report.json",
    mime="application/json",
)
st.download_button(
    "Download quality report (CSV)",
    data=missing_report.to_csv(index=False).encode("utf-8"),
    file_name="datasentinel_quality_report.csv",
    mime="text/csv",
)

st.subheader("Complete PDF Audit Report")
try:
    pdf_bytes = generate_audit_pdf(
        risk_result=current_risk,
        leakage_report=(
            current_leakage
            if isinstance(current_leakage, pd.DataFrame)
            else pd.DataFrame()
        ),
        drift_result=current_drift if isinstance(current_drift, dict) else None,
        dataset_name=uploaded_file.name,
    )
    st.download_button(
        "📄 Download PDF Audit Report",
        data=pdf_bytes,
        file_name="datasentinel_audit_report.pdf",
        mime="application/pdf",
        key="download_pdf_audit_report",
    )
    st.caption(
        "The PDF uses the latest analysis results available in this session. "
        "Run drift detection and target leakage screening to include those results."
    )
except Exception as error:
    st.error(f"Unable to generate the PDF audit report: {error}")

st.success("DataSentinel AI Version 2 is ready.")
