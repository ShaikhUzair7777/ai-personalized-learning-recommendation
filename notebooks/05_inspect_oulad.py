import pandas as pd
from pathlib import Path

# OULAD raw data directory
DATA_DIR = Path("data/raw/oulad")

files = [
    "assessments.csv",
    "courses.csv",
    "studentAssessment.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
    "vle.csv"
]

print("=" * 80)
print("OULAD DATASET INSPECTION")
print("=" * 80)

for filename in files:

    filepath = DATA_DIR / filename

    print("\n" + "=" * 80)
    print(f"FILE: {filename}")
    print("=" * 80)

    df = pd.read_csv(filepath)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    print(list(df.columns))

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nFirst 5 Rows:")
    print(df.head())

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)