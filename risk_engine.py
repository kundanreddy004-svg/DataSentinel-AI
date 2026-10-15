import pandas as pd
import numpy as np


def calculate_dataset_risk(df, target_column=None):
    """
    Calculate a transparent, heuristic dataset risk score.

    This score is a screening indicator, not a validated
    probability of dataset failure.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("The dataset has no columns.")

    total_rows, total_columns = df.shape
    total_cells = total_rows * total_columns

    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    missing_percentage = (
        missing_cells / total_cells * 100
        if total_cells else 0
    )

    duplicate_percentage = (
        duplicate_rows / total_rows * 100
        if total_rows else 0
    )

    issues = []
    score = 0.0

    # 1. Missing-value risk
    if missing_percentage > 0:
        contribution = min(missing_percentage * 2, 25)
        score += contribution

        issues.append({
            "Issue": "Missing values",
            "Severity": (
                "High" if missing_percentage >= 10
                else "Medium" if missing_percentage >= 2
                else "Low"
            ),
            "Evidence": (
                f"{missing_cells} missing cells "
                f"({missing_percentage:.2f}%)."
            ),
            "Score contribution": round(contribution, 2),
            "Recommendation": (
                "Investigate why values are missing before "
                "choosing an imputation or removal strategy."
            ),
        })

    # 2. Duplicate-row risk
    if duplicate_rows > 0:
        contribution = min(duplicate_percentage * 1.5, 20)
        score += contribution

        issues.append({
            "Issue": "Duplicate rows",
            "Severity": (
                "High" if duplicate_percentage >= 10
                else "Medium" if duplicate_percentage >= 2
                else "Low"
            ),
            "Evidence": (
                f"{duplicate_rows} duplicate rows "
                f"({duplicate_percentage:.2f}%)."
            ),
            "Score contribution": round(contribution, 2),
            "Recommendation": (
                "Check whether repeated rows are legitimate "
                "observations or accidental duplicates."
            ),
        })

    # 3. Constant-column risk
    constant_columns = [
        col for col in df.columns
        if df[col].nunique(dropna=True) <= 1
    ]

    if constant_columns:
        contribution = min(5 * len(constant_columns), 15)
        score += contribution

        issues.append({
            "Issue": "Constant or all-missing columns",
            "Severity": "Medium",
            "Evidence": (
                "Columns: "
                + ", ".join(map(str, constant_columns))
            ),
            "Score contribution": round(contribution, 2),
            "Recommendation": (
                "Check whether these columns provide useful "
                "information for the intended analysis."
            ),
        })

    # 4. Potential identifier columns
    identifier_candidates = []

    for col in df.columns:
        name = str(col).lower().replace("-", "_").replace(" ", "_")

        identifier_name = (
            name == "id"
            or name.endswith("_id")
            or name.startswith("id_")
            or name in {"uuid", "record_number", "record_no"}
        )

        unique_ratio = df[col].nunique(dropna=True) / max(total_rows, 1)

        if identifier_name and unique_ratio >= 0.5:
            identifier_candidates.append(col)

    if identifier_candidates:
        contribution = min(5 * len(identifier_candidates), 15)
        score += contribution

        issues.append({
            "Issue": "Potential identifier predictors",
            "Severity": "Medium",
            "Evidence": (
                "Candidate columns: "
                + ", ".join(map(str, identifier_candidates))
            ),
            "Score contribution": round(contribution, 2),
            "Recommendation": (
                "Review these columns before model training. "
                "An identifier is not automatically harmful "
                "or a source of data leakage."
            ),
        })

    # 5. Target-column checks
    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' was not found."
            )

        target_missing = int(df[target_column].isna().sum())

        if target_missing:
            contribution = min(
                target_missing / total_rows * 100,
                20
            )
            score += contribution

            issues.append({
                "Issue": "Missing target values",
                "Severity": "High",
                "Evidence": (
                    f"{target_missing} rows have a missing "
                    "target value."
                ),
                "Score contribution": round(contribution, 2),
                "Recommendation": (
                    "Investigate missing labels. Exclude or "
                    "resolve them appropriately before supervised "
                    "model evaluation."
                ),
            })

    score = round(min(score, 100), 2)

    if score < 15:
        risk_level = "Low"
    elif score < 35:
        risk_level = "Moderate"
    elif score < 60:
        risk_level = "High"
    else:
        risk_level = "Very High"

    if not issues:
        issues.append({
            "Issue": "No configured risk indicators detected",
            "Severity": "Informational",
            "Evidence": (
                "The implemented screening rules did not "
                "identify any of their configured warning conditions."
            ),
            "Score contribution": 0,
            "Recommendation": (
                "Continue with domain-specific checks and "
                "model validation."
            ),
        })

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "rows": total_rows,
        "columns": total_columns,
        "missing_cells": missing_cells,
        "missing_percentage": round(missing_percentage, 2),
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": round(duplicate_percentage, 2),
        "issues": pd.DataFrame(issues),
    }