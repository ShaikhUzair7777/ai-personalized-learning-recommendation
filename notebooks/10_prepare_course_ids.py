import pandas as pd
from pathlib import Path

print("=" * 80)
print("PREPARING COURSE IDENTIFIERS")
print("=" * 80)

INPUT_PATH = Path("data/processed/coursera_clean.csv")
OUTPUT_PATH = Path("data/processed/courses_ready.csv")

print("\nLoading Coursera dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Courses loaded: {len(df):,}")

# ---------------------------------------------------------
# Create stable course ID
# ---------------------------------------------------------

print("\nCreating course IDs...")

df.insert(
    0,
    "course_id",
    [f"C{i:04d}" for i in range(1, len(df) + 1)]
)

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

print("\nShape:")
print(df.shape)

print("\nUnique course IDs:")
print(df["course_id"].nunique())

print("\nDuplicate course IDs:")
print(df["course_id"].duplicated().sum())

print("\nMissing course IDs:")
print(df["course_id"].isnull().sum())

print("\nFirst 10 course IDs:")
print(
    df[
        ["course_id", "course_name", "university"]
    ].head(10).to_string(index=False)
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 80)
print("COURSE ID PREPARATION COMPLETE")
print("=" * 80)

print(f"\nSaved to: {OUTPUT_PATH}")
print(f"Final courses: {len(df):,}")