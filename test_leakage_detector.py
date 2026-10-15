import pandas as pd
from leakage_detector import detect_target_leakage


def main():
    # Test 1: Use the existing sample dataset
    df = pd.read_csv("sample_data.csv")

    report = detect_target_leakage(
        df,
        target_column="Final_Result"
    )

    print("\n" + "=" * 60)
    print("       DATASENTINEL AI — LEAKAGE DETECTOR")
    print("=" * 60)

    print("\nTest 1: Sample dataset")
    print(report.to_string(index=False))

    # Test 2: Create an artificial exact-copy example
    artificial_df = pd.DataFrame({
        "Study_Hours": [2, 5, 7, 3, 8, 4],
        "Copied_Result": [
            "Fail", "Pass", "Pass",
            "Fail", "Pass", "Pass"
        ],
        "Final_Result": [
            "Fail", "Pass", "Pass",
            "Fail", "Pass", "Pass"
        ]
    })

    artificial_report = detect_target_leakage(
        artificial_df,
        target_column="Final_Result"
    )

    print("\n" + "-" * 60)
    print("Test 2: Artificial exact-copy example")
    print(artificial_report.to_string(index=False))

    # Verify that the exact copy was detected
    copied_row = artificial_report[
        artificial_report["Predictor"] == "Copied_Result"
    ]

    assert not copied_row.empty
    assert bool(copied_row.iloc[0]["Exact target copy"])

    # Test 3: Invalid target should produce a clear error
    try:
        detect_target_leakage(
            artificial_df,
            target_column="Missing_Target"
        )
    except ValueError:
        print("\nTest 3: Invalid target handled correctly.")
    else:
        raise AssertionError("Invalid target was not rejected.")

    print("\nSUCCESS: All leakage detector tests passed.")


if __name__ == "__main__":
    main()