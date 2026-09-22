import pandas as pd
from pathlib import Path

# ============================================================
# Coursera Dataset - Initial Inspection
# ============================================================

DATA_PATH = Path("data/raw/coursera/Coursera.csv")

print("=" * 70)
print("AI PERSONALIZED LEARNING RECOMMENDATION SYSTEM")
print("COURsera DATASET INSPECTION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Check file
# ------------------------------------------------------------

if not DATA_PATH.exists():
    print("\nERROR: Dataset file not found!")
    print(f"Expected location: {DATA_PATH}")
    raise SystemExit

print(f"\nDataset found: {DATA_PATH}")
print(f"File size: {DATA_PATH.stat().st_size / (1024 * 1024):.2f} MB")

# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

print("\nLoading dataset...")

try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
except UnicodeDecodeError:
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding="latin1")

print("Dataset loaded successfully!")

# ------------------------------------------------------------
# 3. Dataset dimensions
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("1. DATASET DIMENSIONS")
print("=" * 70)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

# ------------------------------------------------------------
# 4. Column names
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# ------------------------------------------------------------
# 5. Data types
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. DATA TYPES")
print("=" * 70)

print(df.dtypes)

# ------------------------------------------------------------
# 6. Missing values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

for column, count in missing.items():
    percentage = (count / len(df)) * 100
    print(f"{column}: {count:,} ({percentage:.2f}%)")

# ------------------------------------------------------------
# 7. Duplicate rows
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. DUPLICATE ROWS")
print("=" * 70)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates:,}")

# ------------------------------------------------------------
# 8. First 5 records
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("6. FIRST 5 RECORDS")
print("=" * 70)

print(df.head().to_string())

# ------------------------------------------------------------
# 9. Last 5 records
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("7. LAST 5 RECORDS")
print("=" * 70)

print(df.tail().to_string())

# ------------------------------------------------------------
# 10. Unique values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("8. UNIQUE VALUES")
print("=" * 70)

for column in df.columns:
    print(f"{column}: {df[column].nunique(dropna=True):,}")

# ------------------------------------------------------------
# 11. Basic statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("9. NUMERICAL SUMMARY")
print("=" * 70)

print(df.describe(include="all").transpose().to_string())

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)