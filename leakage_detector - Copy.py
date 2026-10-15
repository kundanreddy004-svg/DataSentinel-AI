import pandas as pd
import numpy as np


def detect_target_leakage(df, target_column):
    """
    Screen predictors for potential target leakage.

    Exact target copies are strong warning signals.
    Statistical associations are screening indicators only;
    they do not prove leakage.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The dataset is empty.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    target = df[target_column]
    results = []

    if target.isna().all():
        raise ValueError("The target column contains only missing values.")

    for column in df.columns:
        if column == target_column:
            continue

        feature = df[column]

        # Ignore columns that contain no usable values.
        valid = feature.notna() & target.notna()

        if valid.sum() == 0:
            continue

        x = feature[valid].reset_index(drop=True)
        y = target[valid].reset_index(drop=True)

        # Exact equality on rows where both columns have values.
        exact_match = bool((x.astype(str) == y.astype(str)).all())

        # A predictor may have a different representation while
        # still mapping perfectly to the target.
        unique_x = x.nunique()
        unique_y = y.nunique()

        perfect_mapping = False

        if unique_x > 1 and unique_y > 1:
            mapping = pd.DataFrame({
                "feature": x.astype(str),
                "target": y.astype(str)
            })

            targets_per_feature = (
                mapping.groupby("feature")["target"].nunique()
            )

            features_per_target = (
                mapping.groupby("target")["feature"].nunique()
            )

            feature_determines_target = (
                targets_per_feature.max() == 1
            )

            target_determines_feature = (
                features_per_target.max() == 1
            )

            perfect_mapping = bool(
                feature_determines_target
                and target_determines_feature
            )

        reasons = []

        if exact_match:
            reasons.append("Exact target copy")

        if perfect_mapping:
            reasons.append("Perfect one-to-one target mapping")

        if exact_match:
            status = "High-priority review"
        elif perfect_mapping:
            status = "Investigate"
        else:
            status = "No direct warning detected"

        results.append({
            "Predictor": column,
            "Rows compared": int(valid.sum()),
            "Exact target copy": exact_match,
            "Perfect mapping": perfect_mapping,
            "Status": status,
            "Reason": (
                "; ".join(reasons)
                if reasons
                else "No exact-copy or one-to-one mapping warning"
            ),
        })

    report = pd.DataFrame(results)

    if not report.empty:
        report = report.sort_values(
            by=["Exact target copy", "Perfect mapping"],
            ascending=False
        ).reset_index(drop=True)

    return report