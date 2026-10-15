import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency


def _is_identifier(column_name, series=None):
    """
    Identify likely identifier columns using their names.

    Important:
    Numeric uniqueness alone is NOT sufficient to classify
    a feature as an identifier. This prevents columns such as
    Age, Monthly_Income, and Website_Visits from being excluded
    from drift analysis just because their values are unique.
    """
    name = str(column_name).strip().lower()
    normalized = name.replace("-", "_").replace(" ", "_")

    identifier_names = {
        "id",
        "customer_id",
        "user_id",
        "record_id",
        "employee_id",
        "account_id",
        "transaction_id",
        "product_id",
        "order_id",
        "session_id",
        "customer_number",
        "account_number",
        "record_number",
        "serial_number",
        "uuid",
        "guid",
    }

    if normalized in identifier_names:
        return True

    if normalized.endswith("_id") or normalized.startswith("id_"):
        return True

    if normalized.endswith("_uuid") or normalized.endswith("_guid"):
        return True

    return False


def detect_dataset_drift(
    reference_df,
    current_df,
    numeric_threshold=0.05,
    categorical_threshold=0.05,
):
    """
    Compare reference and current datasets for potential
    numerical and categorical distribution changes.

    Numeric columns:
        Two-sample Kolmogorov-Smirnov test.

    Categorical columns:
        Chi-square test of independence between dataset source
        and category.

    Identifier columns:
        Reported but not statistically tested.

    Missing values:
        Numeric missing values are excluded from the KS test.
        Categorical missing values are treated as a category.

    Returns:
        report:
            DataFrame containing the drift analysis.

        reference_only_columns:
            Columns present only in the reference dataset.

        current_only_columns:
            Columns present only in the current dataset.

        numeric_threshold:
            Significance threshold for numerical comparisons.

        categorical_threshold:
            Significance threshold for categorical comparisons.

    Note:
        Statistical test results are screening indicators.
        They do not prove that a change is harmful.
    """

    # ---------------------------------------------------------
    # INPUT VALIDATION
    # ---------------------------------------------------------

    if not isinstance(reference_df, pd.DataFrame):
        raise TypeError(
            "Reference dataset must be a pandas DataFrame."
        )

    if not isinstance(current_df, pd.DataFrame):
        raise TypeError(
            "Current dataset must be a pandas DataFrame."
        )

    if reference_df.empty or current_df.empty:
        raise ValueError(
            "Both datasets must contain at least one row."
        )

    if not 0 < numeric_threshold < 1:
        raise ValueError(
            "numeric_threshold must be between 0 and 1."
        )

    if not 0 < categorical_threshold < 1:
        raise ValueError(
            "categorical_threshold must be between 0 and 1."
        )

    if not reference_df.columns.is_unique:
        raise ValueError(
            "Reference dataset contains duplicate column names."
        )

    if not current_df.columns.is_unique:
        raise ValueError(
            "Current dataset contains duplicate column names."
        )

    # ---------------------------------------------------------
    # FIND COMMON AND DIFFERENT COLUMNS
    # ---------------------------------------------------------

    common_columns = [
        column
        for column in reference_df.columns
        if column in current_df.columns
    ]

    if not common_columns:
        raise ValueError(
            "The datasets have no columns in common."
        )

    reference_only_columns = [
        column
        for column in reference_df.columns
        if column not in current_df.columns
    ]

    current_only_columns = [
        column
        for column in current_df.columns
        if column not in reference_df.columns
    ]

    results = []

    # ---------------------------------------------------------
    # ANALYSE EACH COMMON COLUMN
    # ---------------------------------------------------------

    for column in common_columns:

        reference_series = reference_df[column]
        current_series = current_df[column]

        # -----------------------------------------------------
        # IDENTIFIER SCREENING
        # -----------------------------------------------------

        if _is_identifier(column):

            results.append({
                "Column": column,
                "Type": "Identifier",
                "Reference rows used": len(reference_series),
                "Current rows used": len(current_series),
                "Statistic": np.nan,
                "P-value": np.nan,
                "Potential drift": "Not tested",
                "Interpretation": (
                    "Column name indicates a likely identifier. "
                    "Distribution testing was skipped. "
                    "Verify the classification manually."
                ),
            })

            continue

        # -----------------------------------------------------
        # NUMERIC COLUMN DETECTION
        # -----------------------------------------------------

        reference_numeric = pd.api.types.is_numeric_dtype(
            reference_series
        )

        current_numeric = pd.api.types.is_numeric_dtype(
            current_series
        )

        # -----------------------------------------------------
        # NUMERICAL DRIFT DETECTION
        # -----------------------------------------------------

        if reference_numeric and current_numeric:

            reference_values = (
                pd.to_numeric(
                    reference_series,
                    errors="coerce",
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            current_values = (
                pd.to_numeric(
                    current_series,
                    errors="coerce",
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if (
                len(reference_values) < 2
                or len(current_values) < 2
            ):

                results.append({
                    "Column": column,
                    "Type": "Numeric",
                    "Reference rows used": len(reference_values),
                    "Current rows used": len(current_values),
                    "Statistic": np.nan,
                    "P-value": np.nan,
                    "Potential drift": "Insufficient data",
                    "Interpretation": (
                        "At least two valid numerical values "
                        "per dataset are required."
                    ),
                })

                continue

            statistic, p_value = ks_2samp(
                reference_values,
                current_values,
            )

            drift_detected = p_value < numeric_threshold

            results.append({
                "Column": column,
                "Type": "Numeric",
                "Reference rows used": len(reference_values),
                "Current rows used": len(current_values),
                "Statistic": round(float(statistic), 4),
                "P-value": round(float(p_value), 6),
                "Potential drift": (
                    "Review recommended"
                    if drift_detected
                    else "Not detected"
                ),
                "Interpretation": (
                    "Numerical distributions differ at the "
                    "configured significance threshold."
                    if drift_detected
                    else
                    "No statistically significant difference "
                    "was detected at the configured threshold."
                ),
            })

        # -----------------------------------------------------
        # CATEGORICAL DRIFT DETECTION
        # -----------------------------------------------------

        else:

            reference_values = (
                reference_series
                .astype("string")
                .fillna("<MISSING>")
            )

            current_values = (
                current_series
                .astype("string")
                .fillna("<MISSING>")
            )

            categories = sorted(
                set(reference_values.tolist())
                | set(current_values.tolist())
            )

            reference_counts = (
                reference_values
                .value_counts()
                .reindex(categories, fill_value=0)
            )

            current_counts = (
                current_values
                .value_counts()
                .reindex(categories, fill_value=0)
            )

            contingency = np.vstack([
                reference_counts.to_numpy(),
                current_counts.to_numpy(),
            ])

            # -------------------------------------------------
            # HANDLE SINGLE-CATEGORY COLUMNS
            # -------------------------------------------------

            if len(categories) < 2:

                results.append({
                    "Column": column,
                    "Type": "Categorical",
                    "Reference rows used": len(reference_values),
                    "Current rows used": len(current_values),
                    "Statistic": 0.0,
                    "P-value": 1.0,
                    "Potential drift": "Not detected",
                    "Interpretation": (
                        "Only one category was observed across "
                        "both datasets."
                    ),
                })

                continue

            # -------------------------------------------------
            # CHI-SQUARE TEST
            # -------------------------------------------------

            active_columns = contingency.sum(axis=0) > 0

            contingency = contingency[:, active_columns]

            if contingency.shape[1] < 2:

                statistic = 0.0
                p_value = 1.0

            else:

                try:

                    statistic, p_value, _, _ = (
                        chi2_contingency(
                            contingency,
                            correction=False,
                        )
                    )

                except ValueError:

                    results.append({
                        "Column": column,
                        "Type": "Categorical",
                        "Reference rows used": len(reference_values),
                        "Current rows used": len(current_values),
                        "Statistic": np.nan,
                        "P-value": np.nan,
                        "Potential drift": "Insufficient data",
                        "Interpretation": (
                            "The chi-square test could not be "
                            "calculated for this category table."
                        ),
                    })

                    continue

            drift_detected = (
                p_value < categorical_threshold
            )

            results.append({
                "Column": column,
                "Type": "Categorical",
                "Reference rows used": len(reference_values),
                "Current rows used": len(current_values),
                "Statistic": round(float(statistic), 4),
                "P-value": round(float(p_value), 6),
                "Potential drift": (
                    "Review recommended"
                    if drift_detected
                    else "Not detected"
                ),
                "Interpretation": (
                    "Category proportions differ at the "
                    "configured significance threshold."
                    if drift_detected
                    else
                    "No statistically significant difference "
                    "was detected at the configured threshold."
                ),
            })

    # ---------------------------------------------------------
    # BUILD FINAL REPORT
    # ---------------------------------------------------------

    report = pd.DataFrame(results)

    return {
        "report": report,
        "reference_only_columns": reference_only_columns,
        "current_only_columns": current_only_columns,
        "numeric_threshold": numeric_threshold,
        "categorical_threshold": categorical_threshold,
    }