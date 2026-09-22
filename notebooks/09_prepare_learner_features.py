import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("PREPARING LEARNER FEATURES FOR RECOMMENDATION")
print("=" * 80)

# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

INPUT_PATH = Path("data/processed/oulad_learner_features.csv")
OUTPUT_PATH = Path("data/processed/learner_features_ready.csv")

# ---------------------------------------------------------
# 2. Load finalized OULAD learner data
# ---------------------------------------------------------

print("\nLoading OULAD learner features...")

df = pd.read_csv(INPUT_PATH)

print(f"Records loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns)}")

# ---------------------------------------------------------
# 3. Create engagement score
# ---------------------------------------------------------

print("\nCreating engagement indicators...")

df["engagement_score"] = (
    0.40 * np.log1p(df["total_clicks"])
    + 0.30 * np.log1p(df["unique_resources"])
    + 0.30 * np.log1p(df["active_days"])
)

# ---------------------------------------------------------
# 4. Create performance indicators
# ---------------------------------------------------------

print("Creating performance indicators...")

df["performance_score"] = df["average_score"]

def performance_level(score):
    if score >= 80:
        return "High"
    elif score >= 60:
        return "Medium"
    else:
        return "Low"

df["performance_level"] = df["performance_score"].apply(
    performance_level
)

# ---------------------------------------------------------
# 5. Create engagement levels
# ---------------------------------------------------------

print("Creating engagement levels...")

low_threshold = df["engagement_score"].quantile(0.25)
high_threshold = df["engagement_score"].quantile(0.75)

def engagement_level(score):
    if score >= high_threshold:
        return "High"
    elif score >= low_threshold:
        return "Medium"
    else:
        return "Low"

df["engagement_level"] = df["engagement_score"].apply(
    engagement_level
)

# ---------------------------------------------------------
# 6. Normalize numerical features
# ---------------------------------------------------------

print("Normalizing numerical learner features...")

numeric_columns = [
    "average_score",
    "assessment_count",
    "total_clicks",
    "unique_resources",
    "active_days",
    "studied_credits",
    "num_of_prev_attempts"
]

for column in numeric_columns:

    min_value = df[column].min()
    max_value = df[column].max()

    if max_value != min_value:
        df[column + "_normalized"] = (
            (df[column] - min_value)
            / (max_value - min_value)
        )
    else:
        df[column + "_normalized"] = 0.0

# ---------------------------------------------------------
# 7. Validation
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

print("\nShape:")
print(df.shape)

print("\nUnique learners:")
print(df["id_student"].nunique())

print("\nDuplicate learner IDs:")
print(df["id_student"].duplicated().sum())

print("\nMissing values:")
print(df.isnull().sum())

print("\nPerformance distribution:")
print(df["performance_level"].value_counts())

print("\nEngagement distribution:")
print(df["engagement_level"].value_counts())

# ---------------------------------------------------------
# 8. Save
# ---------------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 80)
print("LEARNER FEATURE PREPARATION COMPLETE")
print("=" * 80)

print(f"\nSaved to: {OUTPUT_PATH}")
print(f"Final learner records: {len(df):,}")