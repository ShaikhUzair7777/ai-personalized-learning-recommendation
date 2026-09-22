import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/oulad")
PROCESSED_DIR = Path("data/processed")

print("=" * 80)
print("FINALIZING OULAD LEARNER FEATURES")
print("=" * 80)

# ---------------------------------------------------------
# 1. LOAD STUDENT INFORMATION
# ---------------------------------------------------------

print("\nLoading student information...")

student_info = pd.read_csv(
    RAW_DIR / "studentInfo.csv"
)

# Keep only the profile attributes needed by our project.
profile = student_info[
    [
        "id_student",
        "highest_education",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits"
    ]
].copy()

# ---------------------------------------------------------
# 2. AGGREGATE PROFILE TO ONE ROW PER STUDENT
# ---------------------------------------------------------

print("Aggregating student profiles...")

profile = (
    profile
    .groupby("id_student")
    .agg(
        highest_education=("highest_education", "first"),
        age_band=("age_band", "first"),
        num_of_prev_attempts=("num_of_prev_attempts", "max"),
        studied_credits=("studied_credits", "max")
    )
    .reset_index()
)

print(
    f"Unique students after profile aggregation: "
    f"{profile['id_student'].nunique():,}"
)

# ---------------------------------------------------------
# 3. LOAD ASSESSMENT DATA
# ---------------------------------------------------------

print("\nLoading assessment data...")

student_assessment = pd.read_csv(
    RAW_DIR / "studentAssessment.csv"
)

performance = (
    student_assessment
    .groupby("id_student")
    .agg(
        average_score=("score", "mean"),
        assessment_count=("id_assessment", "count")
    )
    .reset_index()
)

# ---------------------------------------------------------
# 4. LOAD VLE ACTIVITY
# ---------------------------------------------------------

print("Loading VLE activity...")

student_vle = pd.read_csv(
    RAW_DIR / "studentVle.csv"
)

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

# ---------------------------------------------------------
# 5. MERGE ALL FEATURES
# ---------------------------------------------------------

print("\nCombining learner features...")

learner_features = profile.merge(
    performance,
    on="id_student",
    how="left"
)

learner_features = learner_features.merge(
    engagement,
    on="id_student",
    how="left"
)

# ---------------------------------------------------------
# 6. HANDLE MISSING OBSERVATIONS
# ---------------------------------------------------------

print("Handling missing observations...")

learner_features["average_score"] = (
    learner_features["average_score"].fillna(0)
)

learner_features["assessment_count"] = (
    learner_features["assessment_count"].fillna(0)
)

learner_features["total_clicks"] = (
    learner_features["total_clicks"].fillna(0)
)

learner_features["unique_resources"] = (
    learner_features["unique_resources"].fillna(0)
)

learner_features["active_days"] = (
    learner_features["active_days"].fillna(0)
)

# ---------------------------------------------------------
# 7. DATA TYPES
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# 8. VALIDATION
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

print("\nShape:")
print(learner_features.shape)

print("\nUnique students:")
print(learner_features["id_student"].nunique())

print("\nDuplicate student IDs:")
print(
    learner_features["id_student"].duplicated().sum()
)

print("\nMissing values:")
print(learner_features.isna().sum())

# ---------------------------------------------------------
# 9. SAVE
# ---------------------------------------------------------

output_file = (
    PROCESSED_DIR /
    "oulad_learner_features.csv"
)

learner_features.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 80)
print("FINAL OULAD DATASET SAVED")
print("=" * 80)

print(f"\nFile: {output_file}")
print(
    f"Records: {len(learner_features):,}"
)