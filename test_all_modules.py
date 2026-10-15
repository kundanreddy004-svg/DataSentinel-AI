import pandas as pd
import pytest

from drift_detector import detect_dataset_drift
from leakage_detector import detect_target_leakage
from risk_engine import calculate_dataset_risk


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Student_ID": range(1, 11),
        "Study_Hours": [2, 3, 2, 4, 5, 6, 3, 4, 7, 5],
        "Attendance": [60, 65, 62, 70, 75, 80, 68, 72, 85, 78],
        "Previous_Score": [45, 50, 48, 60, 65, 75, 55, 62, 88, 70],
        "Final_Result": [
            "Fail", "Fail", "Fail", "Pass", "Pass",
            "Pass", "Fail", "Pass", "Pass", "Pass"
        ],
    })


def test_drift_detector_detects_numeric_drift():
    reference_df = pd.DataFrame({
        "Study_Hours": [1, 2, 1, 2, 3, 2, 1, 3, 2, 1]
    })

    current_df = pd.DataFrame({
        "Study_Hours": [20, 22, 21, 23, 24, 22, 21, 25, 23, 24]
    })

    result = detect_dataset_drift(reference_df, current_df)

    report = result["report"]

    assert not report.empty
    assert "Study_Hours" in report["Column"].values
    assert (
        report.loc[
            report["Column"] == "Study_Hours",
            "Potential drift"
        ].iloc[0] == "Review recommended"
    )


def test_drift_detector_reports_schema_differences():
    reference_df = pd.DataFrame({
        "Age": [20, 21, 22],
        "Region": ["North", "South", "East"],
    })

    current_df = pd.DataFrame({
        "Age": [23, 24, 25],
        "Subscription": ["Basic", "Premium", "Basic"],
    })

    result = detect_dataset_drift(reference_df, current_df)

    assert result["reference_only_columns"] == ["Region"]
    assert result["current_only_columns"] == ["Subscription"]


def test_drift_detector_rejects_empty_dataset():
    reference_df = pd.DataFrame()
    current_df = pd.DataFrame({"Age": [20, 21]})

    with pytest.raises(ValueError):
        detect_dataset_drift(reference_df, current_df)


def test_leakage_detector_detects_exact_target_copy():
    df = pd.DataFrame({
        "Study_Hours": [2, 5, 7, 3, 8, 4],
        "Copied_Result": [
            "Fail", "Pass", "Pass",
            "Fail", "Pass", "Pass"
        ],
        "Final_Result": [
            "Fail", "Pass", "Pass",
            "Fail", "Pass", "Pass"
        ],
    })

    report = detect_target_leakage(
        df,
        target_column="Final_Result"
    )

    copied_row = report.loc[
        report["Predictor"] == "Copied_Result"
    ]

    assert not copied_row.empty
    assert bool(copied_row.iloc[0]["Exact target copy"])


def test_leakage_detector_rejects_missing_target():
    df = pd.DataFrame({
        "Study_Hours": [2, 5, 7],
        "Final_Result": ["Fail", "Pass", "Pass"],
    })

    with pytest.raises(ValueError):
        detect_target_leakage(
            df,
            target_column="Missing_Target"
        )


def test_risk_engine_returns_valid_score(sample_df):
    result = calculate_dataset_risk(
        sample_df,
        target_column="Final_Result"
    )

    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] in {
        "Low", "Moderate", "High", "Critical"
    }
    assert result["rows"] == len(sample_df)
    assert result["columns"] == len(sample_df.columns)


def test_risk_engine_detects_missing_values():
    df = pd.DataFrame({
        "Study_Hours": [2, None, 5, 7],
        "Attendance": [60, 70, 80, 90],
        "Final_Result": ["Fail", "Fail", "Pass", "Pass"],
    })

    result = calculate_dataset_risk(
        df,
        target_column="Final_Result"
    )

    assert result["missing_cells"] == 1
    