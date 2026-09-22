import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("GENERATING SYNTHETIC LEARNER-COURSE INTERACTIONS")
print("=" * 80)

# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

LEARNER_PATH = Path(
    "data/processed/learner_features_ready.csv"
)

COURSE_PATH = Path(
    "data/processed/courses_ready.csv"
)

OUTPUT_PATH = Path(
    "data/processed/synthetic_interactions.csv"
)

# ---------------------------------------------------------
# 2. Experiment configuration
# ---------------------------------------------------------

RANDOM_SEED = 42

N_LEARNERS = 5000

CANDIDATE_COURSES = 50

MIN_INTERACTIONS = 5
MAX_INTERACTIONS = 10

rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------
# 3. Load datasets
# ---------------------------------------------------------

print("\nLoading learner features...")

learners = pd.read_csv(LEARNER_PATH)

print(f"Learners available: {len(learners):,}")

print("\nLoading course data...")

courses = pd.read_csv(COURSE_PATH)

print(f"Courses available: {len(courses):,}")

# ---------------------------------------------------------
# 4. Select learners
# ---------------------------------------------------------

print("\nSelecting learners for the experiment...")

learners_sample = learners.sample(
    n=min(N_LEARNERS, len(learners)),
    random_state=RANDOM_SEED
).copy()

print(
    f"Learners selected: "
    f"{len(learners_sample):,}"
)

# ---------------------------------------------------------
# 5. Prepare course difficulty
# ---------------------------------------------------------

print("\nPreparing course difficulty...")

difficulty_map = {
    "Beginner": 0.25,
    "Intermediate": 0.50,
    "Mixed": 0.50,
    "Advanced": 0.75,
    "All Levels": 0.50
}

courses["difficulty_score"] = (
    courses["difficulty_level"]
    .map(difficulty_map)
    .fillna(0.50)
)

# ---------------------------------------------------------
# 6. Prepare course rating
# ---------------------------------------------------------

print("Preparing course ratings...")

courses["rating_score"] = pd.to_numeric(
    courses["course_rating"],
    errors="coerce"
)

median_rating = courses["rating_score"].median()

courses["rating_score"] = (
    courses["rating_score"]
    .fillna(median_rating)
)

# Normalize rating

rating_min = courses["rating_score"].min()
rating_max = courses["rating_score"].max()

if rating_max != rating_min:

    courses["rating_normalized"] = (
        (courses["rating_score"] - rating_min)
        / (rating_max - rating_min)
    )

else:

    courses["rating_normalized"] = 0.5

# ---------------------------------------------------------
# 7. Learner engagement normalization
# ---------------------------------------------------------

engagement_reference = (
    learners["engagement_score"]
    .quantile(0.95)
)

# ---------------------------------------------------------
# 8. Generate interactions
# ---------------------------------------------------------

print("\nGenerating synthetic interactions...")

interaction_records = []

course_indices = np.arange(len(courses))

for learner_number, (_, learner) in enumerate(
    learners_sample.iterrows(),
    start=1
):

    # -----------------------------------------------------
    # Candidate courses
    # -----------------------------------------------------

    candidate_indices = rng.choice(
        course_indices,
        size=min(
            CANDIDATE_COURSES,
            len(courses)
        ),
        replace=False
    )

    candidate_courses = courses.iloc[
        candidate_indices
    ].copy()

    # -----------------------------------------------------
    # Learner characteristics
    # -----------------------------------------------------

    performance = float(
        learner["average_score"]
    )

    performance_normalized = np.clip(
        performance / 100.0,
        0.0,
        1.0
    )

    engagement_normalized = np.clip(
        float(learner["engagement_score"])
        / engagement_reference,
        0.0,
        1.0
    )

    # -----------------------------------------------------
    # Difficulty fit
    # -----------------------------------------------------

    difficulty_difference = abs(
        candidate_courses["difficulty_score"]
        - performance_normalized
    )

    difficulty_fit = (
        1.0 - difficulty_difference
    )

    # -----------------------------------------------------
    # Base preference score
    # -----------------------------------------------------

    base_score = (
        0.55 * difficulty_fit
        + 0.20 * candidate_courses[
            "rating_normalized"
        ]
        + 0.15 * engagement_normalized
        + 0.10 * rng.random(
            len(candidate_courses)
        )
    )

    # -----------------------------------------------------
    # Convert to relative preference
    # -----------------------------------------------------

    # Standardize within each learner's candidate set.
    score_mean = base_score.mean()
    score_std = base_score.std()

    if score_std > 0:

        relative_score = (
            (base_score - score_mean)
            / score_std
        )

    else:

        relative_score = np.zeros(
            len(candidate_courses)
        )

    # Sigmoid converts relative preference
    # into approximately 0-1 probability.

    preference_probability = (
        1 /
        (
            1 +
            np.exp(
                -1.2 * relative_score
            )
        )
    )

    candidate_courses[
        "preference_probability"
    ] = preference_probability.to_numpy()

    # -----------------------------------------------------
    # Probabilistic interaction selection
    # -----------------------------------------------------

    # Random number of interactions per learner.
    n_interactions = int(
        rng.integers(
            MIN_INTERACTIONS,
            MAX_INTERACTIONS + 1
        )
    )

    # Weighted sampling without replacement.
    probabilities = (
        candidate_courses[
            "preference_probability"
        ].to_numpy()
    )

    probabilities = (
        probabilities /
        probabilities.sum()
    )

    selected_positions = rng.choice(
        len(candidate_courses),
        size=min(
            n_interactions,
            len(candidate_courses)
        ),
        replace=False,
        p=probabilities
    )

    selected_courses = candidate_courses.iloc[
        selected_positions
    ].copy()

    # -----------------------------------------------------
    # Create interaction types
    # -----------------------------------------------------

    for _, course in selected_courses.iterrows():

        probability = float(
            course[
                "preference_probability"
            ]
        )

        random_value = rng.random()

        # Higher preference → stronger interaction,
        # but with probabilistic variation.

        if (
            probability >= 0.70
            and random_value < 0.45
        ):

            interaction_type = "completed"
            interaction_value = 1.00

        elif (
            probability >= 0.55
            and random_value < 0.50
        ):

            interaction_type = "started"
            interaction_value = 0.75

        elif (
            probability >= 0.40
            and random_value < 0.60
        ):

            interaction_type = "viewed"
            interaction_value = 0.50

        else:

            interaction_type = "clicked"
            interaction_value = 0.25

        interaction_records.append(
            {
                "user_id": int(
                    learner["id_student"]
                ),

                "course_id": course[
                    "course_id"
                ],

                "interaction_type":
                    interaction_type,

                "interaction_value":
                    interaction_value,

                "preference_score":
                    round(
                        probability,
                        4
                    )
            }
        )

    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    if learner_number % 500 == 0:

        print(
            f"Processed learners: "
            f"{learner_number:,}/"
            f"{len(learners_sample):,}"
        )

# ---------------------------------------------------------
# 9. Create DataFrame
# ---------------------------------------------------------

interactions = pd.DataFrame(
    interaction_records
)

# ---------------------------------------------------------
# 10. Validation
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("INTERACTION DATA VALIDATION")
print("=" * 80)

print("\nShape:")
print(interactions.shape)

print("\nUnique users:")
print(
    interactions["user_id"].nunique()
)

print("\nUnique courses:")
print(
    interactions["course_id"].nunique()
)

print("\nInteraction types:")
print(
    interactions[
        "interaction_type"
    ].value_counts()
)

print("\nMissing values:")
print(
    interactions.isnull().sum()
)

print("\nDuplicate user-course pairs:")
print(
    interactions[
        ["user_id", "course_id"]
    ].duplicated().sum()
)

print("\nInteractions per user:")
print(
    interactions
    .groupby("user_id")
    .size()
    .describe()
)

print("\nPreference score statistics:")
print(
    interactions[
        "preference_score"
    ].describe()
)

# ---------------------------------------------------------
# 11. Save
# ---------------------------------------------------------

interactions.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 80)
print("SYNTHETIC INTERACTION GENERATION COMPLETE")
print("=" * 80)

print(
    f"\nSaved to: {OUTPUT_PATH}"
)

print(
    f"Total interactions: "
    f"{len(interactions):,}"
)

print(
    f"Unique learners: "
    f"{interactions['user_id'].nunique():,}"
)

print(
    f"Unique courses: "
    f"{interactions['course_id'].nunique():,}"
)

print("\nIMPORTANT:")
print(
    "These interactions are SYNTHETIC and are used "
    "only for controlled recommender-system experimentation."
)