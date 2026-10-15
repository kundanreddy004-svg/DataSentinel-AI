import pandas as pd

from drift_detector import detect_dataset_drift


# Reference dataset: original data
reference_df = pd.DataFrame({
    "Study_Hours": [2, 3, 2, 3, 4, 2, 3, 4, 2, 3],
    "Attendance": [60, 65, 62, 68, 64, 61, 66, 63, 67, 65],
    "Final_Result": [
        "Fail", "Fail", "Fail", "Fail", "Pass",
        "Fail", "Fail", "Pass", "Fail", "Fail"
    ]
})


# Current dataset: deliberately changed distributions
current_df = pd.DataFrame({
    "Study_Hours": [8, 9, 10, 8, 9, 10, 8, 9, 10, 8],
    "Attendance": [90, 95, 98, 92, 96, 99, 91, 94, 97, 93],
    "Final_Result": [
        "Pass", "Pass", "Pass", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Pass", "Pass"
    ]
})


print("\n" + "=" * 55)
print("       DATASENTINEL AI - DRIFT DETECTOR TEST")
print("=" * 55)

try:
    result = detect_dataset_drift(
        reference_df,
        current_df
    )

    report = result["report"]

    print("\nDRIFT DETECTION REPORT")
    print("-" * 55)
    print(report.to_string(index=False))

    print("\nCOLUMN COMPARISON")
    print("-" * 55)

    print(
        "Columns only in reference dataset:",
        result["reference_only_columns"]
    )

    print(
        "Columns only in current dataset:",
        result["current_only_columns"]
    )

    flagged = report[
        report["Potential drift"] == "Review recommended"
    ]

    if not flagged.empty:
        print("\nSUCCESS: Potential drift was detected.")
        print("Columns requiring review:", flagged["Column"].tolist())
    else:
        print("\nWARNING: No potential drift was flagged.")

    print("\nDrift detector execution completed.")

except Exception as error:
    print("\nTEST FAILED")
    print("Error:", error)
    raise