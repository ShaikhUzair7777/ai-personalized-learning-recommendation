import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/oulad")

print("=" * 80)
print("OULAD FEATURE ANALYSIS")
print("=" * 80)

# ---------------------------------------------------------
# 1. STUDENT INFORMATION
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("1. STUDENT INFORMATION")
print("=" * 80)

student_info = pd.read_csv(DATA_DIR / "studentInfo.csv")

print("\nUnique students:")
print(student_info["id_student"].nunique())

print("\nFinal result distribution:")
print(student_info["final_result"].value_counts())

print("\nHighest education:")
print(student_info["highest_education"].value_counts())

print("\nAge band:")
print(student_info["age_band"].value_counts())

print("\nPrevious attempts:")
print(student_info["num_of_prev_attempts"].describe())

print("\nStudied credits:")
print(student_info["studied_credits"].describe())


# ---------------------------------------------------------
# 2. STUDENT ASSESSMENT PERFORMANCE
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("2. STUDENT ASSESSMENT PERFORMANCE")
print("=" * 80)

student_assessment = pd.read_csv(
    DATA_DIR / "studentAssessment.csv"
)

print("\nUnique students with assessments:")
print(student_assessment["id_student"].nunique())

print("\nScore statistics:")
print(student_assessment["score"].describe())

print("\nMissing scores:")
print(student_assessment["score"].isna().sum())

print("\nAssessment types:")
assessments = pd.read_csv(DATA_DIR / "assessments.csv")
print(assessments["assessment_type"].value_counts())


# ---------------------------------------------------------
# 3. LEARNING ACTIVITY
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("3. LEARNING ACTIVITY")
print("=" * 80)

student_vle = pd.read_csv(DATA_DIR / "studentVle.csv")

print("\nUnique students with VLE activity:")
print(student_vle["id_student"].nunique())

print("\nUnique learning resources:")
print(student_vle["id_site"].nunique())

print("\nTotal clicks:")
print(student_vle["sum_click"].sum())

print("\nClick statistics:")
print(student_vle["sum_click"].describe())


# ---------------------------------------------------------
# 4. ACTIVITY TYPES
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("4. VLE ACTIVITY TYPES")
print("=" * 80)

vle = pd.read_csv(DATA_DIR / "vle.csv")

print("\nActivity type distribution:")
print(vle["activity_type"].value_counts())


# ---------------------------------------------------------
# 5. RESOURCE INTERACTION
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("5. RESOURCE INTERACTION")
print("=" * 80)

resource_activity = student_vle.merge(
    vle[["id_site", "activity_type"]],
    on="id_site",
    how="left"
)

print("\nInteractions by activity type:")
print(
    resource_activity
    .groupby("activity_type")["sum_click"]
    .sum()
    .sort_values(ascending=False)
)


# ---------------------------------------------------------
# 6. STUDENT-LEVEL ENGAGEMENT
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("6. STUDENT-LEVEL ENGAGEMENT")
print("=" * 80)

student_engagement = (
    student_vle
    .groupby("id_student")
    .agg(
        total_clicks=("sum_click", "sum"),
        unique_resources=("id_site", "nunique"),
        active_days=("date", "nunique")
    )
    .reset_index()
)

print("\nStudent engagement statistics:")
print(student_engagement[
    ["total_clicks", "unique_resources", "active_days"]
].describe())


# ---------------------------------------------------------
# 7. STUDENT PERFORMANCE FEATURES
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("7. STUDENT PERFORMANCE FEATURES")
print("=" * 80)

student_performance = (
    student_assessment
    .groupby("id_student")
    .agg(
        average_score=("score", "mean"),
        assessment_count=("id_assessment", "count")
    )
    .reset_index()
)

print("\nPerformance statistics:")
print(student_performance[
    ["average_score", "assessment_count"]
].describe())


# ---------------------------------------------------------
# 8. COMBINED LEARNER FEATURES
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("8. COMBINED LEARNER FEATURES")
print("=" * 80)

learner_features = student_info[
    [
        "id_student",
        "highest_education",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits",
        "final_result"
    ]
].copy()

learner_features = learner_features.merge(
    student_performance,
    on="id_student",
    how="left"
)

learner_features = learner_features.merge(
    student_engagement,
    on="id_student",
    how="left"
)

print("\nCombined learner feature shape:")
print(learner_features.shape)

print("\nCombined learner features:")
print(learner_features.head(10))

print("\nMissing values in combined features:")
print(learner_features.isna().sum())

print("\n" + "=" * 80)
print("OULAD FEATURE ANALYSIS COMPLETE")
print("=" * 80)