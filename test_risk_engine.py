import pandas as pd
from risk_engine import calculate_dataset_risk


def main():
    # Load the existing sample dataset
    df = pd.read_csv("sample_data.csv")

    # Calculate the dataset risk score
    result = calculate_dataset_risk(
        df,
        target_column="Final_Result"
    )

    print("\n" + "=" * 45)
    print("       DATASENTINEL AI — RISK ENGINE")
    print("=" * 45)

    print(f"Dataset rows       : {result['rows']}")
    print(f"Dataset columns    : {result['columns']}")
    print(f"Missing cells      : {result['missing_cells']}")
    print(f"Duplicate rows     : {result['duplicate_rows']}")

    print("-" * 45)
    print(f"Risk score         : {result['risk_score']}/100")
    print(f"Risk level         : {result['risk_level']}")
    print("-" * 45)

    print("\nDetected issues:")
    print(result["issues"].to_string(index=False))

    # Basic verification checks
    assert result["rows"] == 13
    assert result["columns"] == 5
    assert result["missing_cells"] == 1
    assert result["duplicate_rows"] == 1
    assert 0 <= result["risk_score"] <= 100

    print("\nSUCCESS: All basic verification checks passed.")


if __name__ == "__main__":
    main()