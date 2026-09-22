import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# Coursera Dataset - Data Cleaning
# ============================================================

RAW_PATH = Path("data/raw/coursera/Coursera.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "coursera_clean.csv"

print("=" * 70)
print("AI PERSONALIZED LEARNING RECOMMENDATION SYSTEM")
print("COURsera DATA CLEANING")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("\nLoading dataset...")

try:
    df = pd.read_csv(RAW_PATH, encoding="utf-8")
except UnicodeDecodeError:
    try:
        df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(RAW_PATH, encoding="latin1")

print(f"Original rows: {len(df):,}")
print(f"Original columns: {len(df.columns)}")

# ------------------------------------------------------------
# 2. Standardize column names
# ------------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nStandardized columns:")
print(list(df.columns))

# ------------------------------------------------------------
# 3. Remove exact duplicate rows
# ------------------------------------------------------------

duplicates_before = df.duplicated().sum()

print(f"\nDuplicate rows found: {duplicates_before:,}")

df = df.drop_duplicates().copy()

print(f"Rows after duplicate removal: {len(df):,}")

# ------------------------------------------------------------
# 4. Clean text columns
# ------------------------------------------------------------

text_columns = [
    "course_name",
    "university",
    "difficulty_level",
    "course_url",
    "course_description",
    "skills"
]

for column in text_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.replace("\t", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

# ------------------------------------------------------------
# 5. Clean course rating
# ------------------------------------------------------------

df["course_rating"] = pd.to_numeric(
    df["course_rating"],
    errors="coerce"
)

# ------------------------------------------------------------
# 6. Remove rows missing essential information
# ------------------------------------------------------------

required_columns = [
    "course_name",
    "course_description",
    "skills"
]

before_required_filter = len(df)

df = df.dropna(
    subset=required_columns
).copy()

removed_required = before_required_filter - len(df)

print(
    f"\nRows removed due to missing essential data: "
    f"{removed_required:,}"
)

# ------------------------------------------------------------
# 7. Remove duplicate course URLs
# ------------------------------------------------------------

url_duplicates = df["course_url"].duplicated().sum()

print(f"Duplicate course URLs: {url_duplicates:,}")

df = df.drop_duplicates(
    subset=["course_url"]
).copy()

# ------------------------------------------------------------
# 8. Reset index
# ------------------------------------------------------------

df = df.reset_index(drop=True)

# ------------------------------------------------------------
# 9. Create combined text for recommendation model
# ------------------------------------------------------------

df["combined_text"] = (
    df["course_name"].fillna("") + " " +
    df["course_description"].fillna("") + " " +
    df["skills"].fillna("")
)

# ------------------------------------------------------------
# 10. Create processed directory
# ------------------------------------------------------------

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# 11. Save cleaned dataset
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)

# ------------------------------------------------------------
# 12. Final report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print(f"Final rows    : {len(df):,}")
print(f"Final columns : {len(df.columns)}")
print(f"Output file   : {OUTPUT_PATH}")

print("\nFinal columns:")
for column in df.columns:
    print(f"- {column}")

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 3 cleaned records:")
print(
    df[
        [
            "course_name",
            "university",
            "difficulty_level",
            "course_rating",
            "skills"
        ]
    ].head(3).to_string()
)

print("\n" + "=" * 70)
print("READY FOR FEATURE ENGINEERING")
print("=" * 70)