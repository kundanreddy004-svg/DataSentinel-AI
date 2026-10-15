"""
DataSentinel AI — Automated Audit Explanations
Provides understandable explanations and recommended actions
for data-quality, leakage, and drift findings.
"""

import pandas as pd


def explain_dataset_risk(risk_result):
    """Explain the dataset risk score and its contributing issues."""

    if not isinstance(risk_result, dict):
        return {
            "title": "Dataset Risk Analysis",
            "severity": "Unknown",
            "explanation": "A valid risk analysis result was not provided.",
            "recommendations": [
                "Run the dataset risk analysis before requesting an explanation."
            ],
        }

    score = risk_result.get("risk_score")
    level = risk_result.get("risk_level", "Unknown")
    rows = risk_result.get("rows", 0)
    columns = risk_result.get("columns", 0)
    missing_cells = risk_result.get("missing_cells", 0)
    duplicate_rows = risk_result.get("duplicate_rows", 0)

    if score is None:
        score_text = "unavailable"
    else:
        try:
            score_text = f"{float(score):.2f}/100"
        except (TypeError, ValueError):
            score_text = str(score)

    explanation = (
        f"The dataset contains {rows} rows and {columns} columns. "
        f"Its reported risk score is {score_text}, with a risk level "
        f"of '{level}'. The analysis identified {missing_cells} missing "
        f"cells and {duplicate_rows} duplicate rows."
    )

    recommendations = []

    if missing_cells:
        recommendations.append(
            "Inspect missing values column by column. Choose an appropriate "
            "imputation or row-removal strategy based on the meaning of the "
            "data and the reason values are missing."
        )

    if duplicate_rows:
        recommendations.append(
            "Inspect duplicate records and determine whether they are genuine "
            "repeated observations or accidental duplicates before removing them."
        )

    issues = risk_result.get("issues")

    if isinstance(issues, pd.DataFrame) and not issues.empty:
        recommendations.append(
            "Review the detailed risk findings and prioritize issues that could "
            "affect downstream analysis or model performance."
        )
    elif isinstance(issues, (list, tuple)) and issues:
        recommendations.append(
            "Review the individual risk findings and address the most important "
            "issues first."
        )

    if not recommendations:
        recommendations.append(
            "No missing cells or duplicate rows were reported by these checks. "
            "Continue reviewing schema, data validity, leakage, and drift."
        )

    return {
        "title": "Dataset Risk Analysis",
        "severity": str(level),
        "explanation": explanation,
        "recommendations": recommendations,
    }


def explain_missing_values(df):
    """Explain missing values in each affected column."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        return {
            "title": "Missing Values",
            "severity": "Low",
            "explanation": "No missing values were found in the dataset.",
            "recommendations": [
                "Continue checking data validity, unexpected values, "
                "and consistency between columns."
            ],
            "details": [],
        }

    details = []

    for column, count in missing.items():
        percentage = (count / len(df) * 100) if len(df) else 0

        details.append({
            "column": str(column),
            "missing_count": int(count),
            "missing_percentage": round(percentage, 2),
        })

    total_missing = int(missing.sum())

    explanation = (
        f"{total_missing} missing cells were found across "
        f"{len(missing)} columns. Missing values can reduce the number "
        "of usable observations or affect model training, depending on "
        "how they are handled."
    )

    recommendations = [
        "Review the missing-value percentage and meaning of each affected column.",
        "Consider median imputation for suitable numeric features or mode "
        "imputation for suitable categorical features.",
        "Consider a missing-value indicator when the fact that a value is missing "
        "may itself be informative.",
        "Avoid automatically deleting rows without checking how much data "
        "would be lost and whether the missingness is systematic.",
    ]

    return {
        "title": "Missing Values",
        "severity": "Needs review",
        "explanation": explanation,
        "recommendations": recommendations,
        "details": details,
    }


def explain_duplicates(df):
    """Explain duplicate rows and suggest an appropriate review."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count == 0:
        return {
            "title": "Duplicate Rows",
            "severity": "Low",
            "explanation": "No exact duplicate rows were detected.",
            "recommendations": [
                "Check for partial duplicates using relevant business keys "
                "if duplicate records are still suspected."
            ],
        }

    percentage = (
        duplicate_count / len(df) * 100
        if len(df)
        else 0
    )

    return {
        "title": "Duplicate Rows",
        "severity": "Needs review",
        "explanation": (
            f"The dataset contains {duplicate_count} exact duplicate rows, "
            f"representing {percentage:.2f}% of all rows. Duplicates may "
            "overrepresent repeated observations in analysis or model training."
        ),
        "recommendations": [
            "Inspect duplicate records before making changes.",
            "Determine whether repeated records are legitimate observations.",
            "If duplicates are accidental, remove them using a documented rule.",
            "Re-run the audit after cleaning to confirm the result.",
        ],
    }


def explain_leakage(leakage_report):
    """Explain target-leakage findings returned by the leakage detector."""

    if not isinstance(leakage_report, pd.DataFrame):
        return {
            "title": "Target Leakage",
            "severity": "Unknown",
            "explanation": "A valid leakage report was not provided.",
            "recommendations": [
                "Run target-leakage detection and provide its report."
            ],
        }

    if leakage_report.empty:
        return {
            "title": "Target Leakage",
            "severity": "Low",
            "explanation": (
                "The leakage detector did not report any suspicious predictors. "
                "This does not guarantee that the dataset is free from leakage."
            ),
            "recommendations": [
                "Review feature definitions and confirm that predictors would "
                "be available at the time of real-world prediction."
            ],
        }

    status_column = (
        "Status" if "Status" in leakage_report.columns else None
    )

    if status_column:
        statuses = (
            leakage_report[status_column]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        suspicious = leakage_report[
            statuses.str.contains(
                "leak|suspicious|warning|high risk|exact target",
                regex=True,
            )
        ]
    else:
        suspicious = leakage_report.iloc[0:0]

    if suspicious.empty:
        return {
            "title": "Target Leakage",
            "severity": "Needs review",
            "explanation": (
                f"The leakage detector returned {len(leakage_report)} "
                "report rows, but no clearly suspicious status could be "
                "identified from the available Status column."
            ),
            "recommendations": [
                "Inspect the full leakage report and the detector's criteria.",
                "Verify that predictors do not contain information derived "
                "from the target."
            ],
        }

    predictors = []

    if "Predictor" in suspicious.columns:
        predictors = suspicious["Predictor"].astype(str).tolist()

    return {
        "title": "Target Leakage",
        "severity": "High — investigate",
        "explanation": (
            f"The detector marked {len(suspicious)} report entries as "
            "potentially suspicious. Target leakage can make evaluation "
            "metrics look better than real-world performance."
        ),
        "recommendations": [
            (
                "Review these predictors: " + ", ".join(predictors)
                if predictors
                else "Review the suspicious entries in the leakage report."
            ),
            "Confirm whether each suspicious feature is available at prediction time.",
            "Remove confirmed leakage sources and repeat model evaluation.",
        ],
    }


def explain_drift(drift_result):
    """Explain distribution drift findings returned by the drift detector."""

    if not isinstance(drift_result, dict):
        return {
            "title": "Dataset Drift",
            "severity": "Unknown",
            "explanation": "A valid drift result was not provided.",
            "recommendations": [
                "Run drift detection on a reference dataset and a current dataset."
            ],
        }

    report = drift_result.get("report")

    if not isinstance(report, pd.DataFrame):
        return {
            "title": "Dataset Drift",
            "severity": "Unknown",
            "explanation": (
                "The drift result does not contain the expected report table."
            ),
            "recommendations": [
                "Check the output format returned by detect_dataset_drift()."
            ],
        }

    reference_only = drift_result.get("reference_only_columns", [])
    current_only = drift_result.get("current_only_columns", [])

    if report.empty:
        explanation = (
            "The drift report contains no feature-level results. "
            "Check whether both datasets contain comparable columns and "
            "whether the detector completed successfully."
        )

        severity = "Needs review"
        recommendations = [
            "Verify that reference and current datasets use compatible schemas.",
            "Review the detector output before concluding that no drift exists.",
        ]

    else:
        drift_columns = [
            column for column in report.columns
            if any(
                word in str(column).lower()
                for word in ("drift", "changed", "status", "significant")
            )
        ]

        drift_count = None

        for column in drift_columns:
            values = report[column]

            if pd.api.types.is_bool_dtype(values):
                drift_count = int(values.fillna(False).sum())
                break

        explanation = (
            f"The detector returned results for {len(report)} feature(s). "
            "Distribution differences can indicate changes in incoming data, "
            "but do not by themselves prove that model performance has declined."
        )

        if drift_count is not None:
            explanation += f" The selected drift indicator marked {drift_count} feature(s)."

        severity = "Review recommended"

        recommendations = [
            "Inspect the detector's feature-level results and drift thresholds.",
            "Check whether changes reflect legitimate changes in the population "
            "or problems in data collection.",
            "If a predictive model is deployed, evaluate its performance using "
            "recent labeled data when available.",
        ]

    if reference_only:
        explanation += (
            f" {len(reference_only)} column(s) occur only in the reference dataset."
        )
        recommendations.append(
            "Investigate reference-only columns and determine whether their "
            "removal from the current dataset is expected."
        )

    if current_only:
        explanation += (
            f" {len(current_only)} column(s) occur only in the current dataset."
        )
        recommendations.append(
            "Investigate newly introduced columns and update the data pipeline "
            "or schema documentation where appropriate."
        )

    return {
        "title": "Dataset Drift",
        "severity": severity,
        "explanation": explanation,
        "recommendations": recommendations,
    }


def format_explanation(explanation):
    """Format an explanation dictionary as readable plain text."""

    if not isinstance(explanation, dict):
        return str(explanation)

    lines = [
        explanation.get("title", "Audit Explanation"),
        f"Severity: {explanation.get('severity', 'Unknown')}",
        "",
        explanation.get("explanation", ""),
        "",
        "Recommended actions:",
    ]

    for recommendation in explanation.get("recommendations", []):
        lines.append(f"- {recommendation}")

    details = explanation.get("details", [])

    if details:
        lines.extend(["", "Affected columns:"])

        for item in details:
            lines.append(
                f"- {item['column']}: {item['missing_count']} missing cells "
                f"({item['missing_percentage']}%)"
            )

    return "\n".join(lines)