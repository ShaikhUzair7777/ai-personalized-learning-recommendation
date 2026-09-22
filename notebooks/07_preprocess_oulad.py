import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path("data/raw/oulad")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("OULAD PREPROCESSING")
print("=" * 80)


# ============================================================
# 1. LOAD STUDENT INFORMATION
# ============================================================

print("\nLoading student information...")

student_info = pd.read_csv(
    RAW_DIR / "studentInfo.csv"
)

print(f"Student records loaded: {len(student_info):,}")


# ============================================================
# 2. LOAD ASSESSMENT DATA
# ============================================================

print("Loading assessment data...")

student_assessment = pd.read_csv(
    RAW_DIR / "studentAssessment.csv"
)

print(
    f"Assessment records loaded: "
    f"{len(student_assessment):,}"
)


# ============================================================
# 3. CREATE PERFORMANCE FEATURES
# ============================================================

print("\nCreating performance features...")

# Calculate learner-level assessment statistics.
#
# Missing scores are automatically ignored when calculating
# the mean. We do NOT replace missing scores with zero.

performance = (
    student_assessment
    .groupby("id_student")
    .agg(
        average_score=("score", "mean"),
        assessment_count=("id_assessment", "count")
    )
    .reset_index()
)

print(
    f"Students with assessment information: "
    f"{len(performance):,}"
)


# ============================================================
# 4. LOAD VLE ACTIVITY
# ============================================================

print("\nLoading VLE activity...")

student_vle = pd.read_csv(
    RAW_DIR / "studentVle.csv"
)

print(
    f"VLE interaction records loaded: "
    f"{len(student_vle):,}"
)


# ============================================================
# 5. CREATE ENGAGEMENT FEATURES
# ============================================================

print("\nCreating engagement features...")

engagement = (
    student_vle
    .groupby("id_student")
    .agg(
        total_clicks=("sum_click", "sum"),
        unique_resources=("id_site", "nunique"),
        active_days=("date", "nunique")
    )
    .reset_index()
)

print(
    f"Students with VLE activity: "
    f"{len(engagement):,}"
)


# ============================================================
# 6. SELECT PROFILE FEATURES
# ============================================================

print("\nSelecting learner profile features...")

profile = student_info[
    [
        "id_student",
        "highest_education",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits"
    ]
].copy()


# ============================================================
# 7. MERGE PROFILE + PERFORMANCE
# ============================================================

print("\nCombining learner information...")

learner_features = profile.merge(
    performance,
    on="id_student",
    how="left"
)

# Add learning engagement.
learner_features = learner_features.merge(
    engagement,
    on="id_student",
    how="left"
)


# ============================================================
# 8. HANDLE MISSING VALUES
# ============================================================

print("\nHandling missing values...")

# No assessment record means there was no observed
# assessment activity in the available data.
#
# We keep the learner instead of deleting the record.

learner_features["average_score"] = (
    learner_features["average_score"].fillna(0)
)

learner_features["assessment_count"] = (
    learner_features["assessment_count"].fillna(0)
)

# No VLE activity means there was no observed VLE activity.

learner_features["total_clicks"] = (
    learner_features["total_clicks"].fillna(0)
)

learner_features["unique_resources"] = (
    learner_features["unique_resources"].fillna(0)
)

learner_features["active_days"] = (
    learner_features["active_days"].fillna(0)
)


# ============================================================
# 9. DATA TYPE CLEANUP
# ============================================================

print("Cleaning data types...")

learner_features["assessment_count"] = (
    learner_features["assessment_count"].astype(int)
)

learner_features["total_clicks"] = (
    learner_features["total_clicks"].astype(int)
)

learner_features["unique_resources"] = (
    learner_features["unique_resources"].astype(int)
)

learner_features["active_days"] = (
    learner_features["active_days"].astype(int)
)


# ============================================================
# 10. VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

print("\nShape:")
print(learner_features.shape)

print("\nColumns:")
print(list(learner_features.columns))

print("\nMissing values:")
print(learner_features.isna().sum())

print("\nDuplicate learner IDs:")
print(
    learner_features["id_student"].duplicated().sum()
)

print("\nFirst 10 rows:")
print(learner_features.head(10))


# ============================================================
# 11. SAVE PROCESSED DATASET
# ============================================================

output_file = (
    PROCESSED_DIR /
    "oulad_learner_features.csv"
)

learner_features.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 80)
print("OULAD PREPROCESSING COMPLETE")
print("=" * 80)

print(f"\nSaved to: {output_file}")
print(
    f"Final learner records: "
    f"{len(learner_features):,}"
)