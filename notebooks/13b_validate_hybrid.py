import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# STEP 13B — HYBRID RECOMMENDER VALIDATION
# =============================================================================

print("=" * 80)
print("HYBRID RECOMMENDER VALIDATION")
print("=" * 80)


# =============================================================================
# 1. FILE PATHS
# =============================================================================

COURSE_FILE = "data/processed/courses_ready.csv"
LEARNER_FILE = "data/processed/learner_features_ready.csv"
INTERACTION_FILE = "data/processed/synthetic_interactions.csv"

TFIDF_MATRIX_FILE = "models/tfidf_matrix.npz"
TFIDF_VECTORIZER_FILE = "models/tfidf_vectorizer.joblib"

USER_COURSE_MATRIX_FILE = "models/user_course_matrix.npz"
USER_TO_INDEX_FILE = "models/user_to_index.joblib"
COURSE_TO_INDEX_FILE = "models/course_to_index.joblib"

OUTPUT_FILE = (
    "data/processed/hybrid_validation_results.csv"
)


# =============================================================================
# 2. LOAD DATA
# =============================================================================

print("\nLoading datasets...")

courses = pd.read_csv(COURSE_FILE)
learners = pd.read_csv(LEARNER_FILE)
interactions = pd.read_csv(INTERACTION_FILE)

print(f"Courses: {len(courses):,}")
print(f"Learners: {len(learners):,}")
print(f"Interactions: {len(interactions):,}")


# =============================================================================
# 3. LOAD TF-IDF CONTENT MODEL
# =============================================================================

print("\nLoading TF-IDF model...")

tfidf_matrix = load_npz(
    TFIDF_MATRIX_FILE
)

tfidf_vectorizer = joblib.load(
    TFIDF_VECTORIZER_FILE
)

print(
    f"TF-IDF matrix shape: "
    f"{tfidf_matrix.shape}"
)


# =============================================================================
# 4. LOAD COLLABORATIVE FILTERING DATA
# =============================================================================

print("\nLoading collaborative filtering data...")

user_course_matrix = load_npz(
    USER_COURSE_MATRIX_FILE
)

user_to_index = joblib.load(
    USER_TO_INDEX_FILE
)

course_to_index = joblib.load(
    COURSE_TO_INDEX_FILE
)

print(
    f"User-course matrix shape: "
    f"{user_course_matrix.shape}"
)

print(
    f"KNN course index size: "
    f"{len(course_to_index):,}"
)


# =============================================================================
# 5. COURSE INDEX MAPPING
# =============================================================================

print("\nPreparing course index mappings...")

course_id_to_row = {
    course_id: index
    for index, course_id
    in enumerate(courses["course_id"])
}

print(
    f"Complete catalog courses: "
    f"{len(course_id_to_row):,}"
)

print(
    f"KNN courses: "
    f"{len(course_to_index):,}"
)

missing_from_knn = (
    set(course_id_to_row.keys())
    - set(course_to_index.keys())
)

print(
    f"Courses without KNN interactions: "
    f"{len(missing_from_knn):,}"
)


# =============================================================================
# 6. DIFFICULTY FIT FUNCTION
# =============================================================================

def calculate_difficulty_fit(
    performance_level,
    course_difficulty
):
    """
    Calculates how suitable a course difficulty
    is for a learner's performance level.
    """

    performance_level = str(
        performance_level
    ).lower()

    course_difficulty = str(
        course_difficulty
    ).lower()

    # -------------------------------------------------------------------------
    # LOW PERFORMANCE
    # -------------------------------------------------------------------------

    if performance_level == "low":

        if course_difficulty in [
            "beginner",
            "basic"
        ]:
            return 1.00

        elif course_difficulty == "intermediate":
            return 0.65

        elif course_difficulty == "advanced":
            return 0.25

        return 0.60

    # -------------------------------------------------------------------------
    # MEDIUM PERFORMANCE
    # -------------------------------------------------------------------------

    elif performance_level == "medium":

        if course_difficulty == "intermediate":
            return 1.00

        elif course_difficulty in [
            "beginner",
            "basic"
        ]:
            return 0.75

        elif course_difficulty == "advanced":
            return 0.65

        return 0.75

    # -------------------------------------------------------------------------
    # HIGH PERFORMANCE
    # -------------------------------------------------------------------------

    elif performance_level == "high":

        if course_difficulty == "advanced":
            return 1.00

        elif course_difficulty == "intermediate":
            return 0.85

        elif course_difficulty in [
            "beginner",
            "basic"
        ]:
            return 0.55

        return 0.75

    return 0.60


# =============================================================================
# 7. CONTENT-BASED SCORES
# =============================================================================

def get_content_scores(
    user_profile
):
    """
    Generates TF-IDF cosine similarity scores
    for every course in the complete catalog.
    """

    profile_vector = (
        tfidf_vectorizer.transform(
            [user_profile]
        )
    )

    scores = cosine_similarity(
        profile_vector,
        tfidf_matrix
    ).flatten()

    return scores


# =============================================================================
# 8. COLLABORATIVE FILTERING SCORES
# =============================================================================

def get_collaborative_scores(
    user_id
):
    """
    Generates KNN collaborative scores.

    IMPORTANT:
    The KNN matrix contains 3,423 courses while
    the complete catalog contains 3,424 courses.

    Therefore, this function aligns the KNN
    scores back to the complete 3,424-course
    catalog.

    Any course that does not exist in the KNN
    interaction dataset receives a collaborative
    score of 0.
    """

    # -------------------------------------------------------------------------
    # If learner is not in collaborative dataset
    # -------------------------------------------------------------------------

    if user_id not in user_to_index:

        return np.zeros(
            len(courses)
        )

    user_index = user_to_index[
        user_id
    ]

    user_vector = user_course_matrix[
        user_index
    ]

    # -------------------------------------------------------------------------
    # Calculate learner similarity
    # -------------------------------------------------------------------------

    similarities = cosine_similarity(
        user_vector,
        user_course_matrix
    ).flatten()

    # Exclude the learner themselves.
    similarities[user_index] = 0

    # -------------------------------------------------------------------------
    # Select top 10 similar learners
    # -------------------------------------------------------------------------

    neighbor_indices = np.argsort(
        similarities
    )[-10:][::-1]

    neighbor_similarities = (
        similarities[
            neighbor_indices
        ]
    )

    # -------------------------------------------------------------------------
    # Aggregate neighbor interactions
    # -------------------------------------------------------------------------

    weighted_scores = (
        user_course_matrix[
            neighbor_indices
        ]
        .multiply(
            neighbor_similarities[
                :, np.newaxis
            ]
        )
        .sum(axis=0)
    )

    weighted_scores = np.asarray(
        weighted_scores
    ).flatten()

    # -------------------------------------------------------------------------
    # Normalize collaborative scores
    # -------------------------------------------------------------------------

    max_score = weighted_scores.max()

    if max_score > 0:

        weighted_scores = (
            weighted_scores /
            max_score
        )

    # -------------------------------------------------------------------------
    # ALIGN 3,423 KNN COURSES
    # WITH 3,424 CATALOG COURSES
    # -------------------------------------------------------------------------

    aligned_scores = np.zeros(
        len(courses)
    )

    for course_id, knn_index in (
        course_to_index.items()
    ):

        if course_id in course_id_to_row:

            catalog_index = (
                course_id_to_row[
                    course_id
                ]
            )

            aligned_scores[
                catalog_index
            ] = weighted_scores[
                knn_index
            ]

    return aligned_scores


# =============================================================================
# 9. FIND VALID HYBRID LEARNERS
# =============================================================================

print(
    "\nFinding learners available in BOTH datasets..."
)

collaborative_user_ids = set(
    user_to_index.keys()
)

learner_ids = set(
    learners["id_student"]
)

valid_user_ids = sorted(
    collaborative_user_ids.intersection(
        learner_ids
    )
)

print(
    f"Collaborative learners: "
    f"{len(collaborative_user_ids):,}"
)

print(
    f"Learners with profiles: "
    f"{len(learner_ids):,}"
)

print(
    f"Valid hybrid learners: "
    f"{len(valid_user_ids):,}"
)


# =============================================================================
# 10. SELECT TEST LEARNERS
# =============================================================================

if len(valid_user_ids) < 3:

    raise ValueError(
        "Fewer than 3 learners exist in "
        "both datasets."
    )


# Select three valid learners.
test_user_ids = valid_user_ids[:3]

print("\nSelected test learners:")

for user_id in test_user_ids:

    print(
        f"  Learner ID: {user_id}"
    )


# =============================================================================
# 11. TEST LEARNING PROFILES
# =============================================================================

test_profiles = {

    test_user_ids[0]: """
    Python Data Science Machine Learning
    Data Analysis SQL Statistics
    Data Visualization Pandas NumPy
    """,

    test_user_ids[1]: """
    Web Development HTML CSS JavaScript
    React Node.js Express MongoDB
    Frontend Backend Full Stack
    """,

    test_user_ids[2]: """
    Blockchain cryptocurrency Ethereum
    Smart Contracts Solidity Web3
    Decentralized Applications distributed systems
    """
}


# =============================================================================
# 12. RUN HYBRID VALIDATION
# =============================================================================

all_results = []


for user_id in test_user_ids:

    print("\n" + "-" * 80)

    print(
        f"VALIDATING LEARNER: {user_id}"
    )

    print("-" * 80)

    # -------------------------------------------------------------------------
    # LEARNER PROFILE
    # -------------------------------------------------------------------------

    learner_rows = learners[
        learners["id_student"] == user_id
    ]

    if learner_rows.empty:

        print(
            "Learner profile not found. Skipping."
        )

        continue

    learner = learner_rows.iloc[0]

    performance_score = float(
        learner["performance_score"]
    )

    performance_level = (
        learner["performance_level"]
    )

    engagement_score = float(
        learner["engagement_score"]
    )

    print(
        f"Performance score : "
        f"{performance_score:.2f}"
    )

    print(
        f"Performance level : "
        f"{performance_level}"
    )

    print(
        f"Engagement score  : "
        f"{engagement_score:.2f}"
    )

    # -------------------------------------------------------------------------
    # CONTENT SCORE
    # -------------------------------------------------------------------------

    content_scores = get_content_scores(
        test_profiles[user_id]
    )

    # -------------------------------------------------------------------------
    # COLLABORATIVE SCORE
    # -------------------------------------------------------------------------

    collaborative_scores = (
        get_collaborative_scores(
            user_id
        )
    )

    # -------------------------------------------------------------------------
    # DIFFICULTY FIT
    # -------------------------------------------------------------------------

    difficulty_scores = np.array([

        calculate_difficulty_fit(
            performance_level,
            difficulty
        )

        for difficulty
        in courses["difficulty_level"]

    ])

    # -------------------------------------------------------------------------
    # PERFORMANCE SCORE
    # -------------------------------------------------------------------------

    performance_normalized = np.clip(
        performance_score / 100,
        0,
        1
    )

    performance_scores = np.full(
        len(courses),
        performance_normalized
    )

    # -------------------------------------------------------------------------
    # HYBRID SCORE
    # -------------------------------------------------------------------------

    hybrid_scores = (

        0.40 * content_scores

        + 0.30 * collaborative_scores

        + 0.15 * performance_scores

        + 0.15 * difficulty_scores

    )

    # -------------------------------------------------------------------------
    # REMOVE COURSES ALREADY INTERACTED WITH
    # -------------------------------------------------------------------------

    seen_courses = set(
        interactions[
            interactions["user_id"] == user_id
        ]["course_id"]
    )

    for course_id in seen_courses:

        if course_id in course_id_to_row:

            row_index = (
                course_id_to_row[
                    course_id
                ]
            )

            hybrid_scores[
                row_index
            ] = -1

    # -------------------------------------------------------------------------
    # GET TOP 10
    # -------------------------------------------------------------------------

    top_indices = np.argsort(
        hybrid_scores
    )[-10:][::-1]

    recommendations = (
        courses.iloc[
            top_indices
        ].copy()
    )

    # -------------------------------------------------------------------------
    # ADD INDIVIDUAL SIGNALS
    # -------------------------------------------------------------------------

    recommendations[
        "content_score"
    ] = content_scores[
        top_indices
    ]

    recommendations[
        "collaborative_score"
    ] = collaborative_scores[
        top_indices
    ]

    recommendations[
        "performance_score"
    ] = performance_scores[
        top_indices
    ]

    recommendations[
        "difficulty_fit"
    ] = difficulty_scores[
        top_indices
    ]

    recommendations[
        "hybrid_score"
    ] = hybrid_scores[
        top_indices
    ]

    recommendations[
        "test_user_id"
    ] = user_id

    # -------------------------------------------------------------------------
    # STORE RESULTS
    # -------------------------------------------------------------------------

    all_results.append(
        recommendations
    )

    # -------------------------------------------------------------------------
    # DISPLAY RECOMMENDATIONS
    # -------------------------------------------------------------------------

    print(
        "\nTop 10 Hybrid Recommendations:\n"
    )

    display_columns = [

        "course_id",
        "course_name",
        "difficulty_level",
        "content_score",
        "collaborative_score",
        "performance_score",
        "difficulty_fit",
        "hybrid_score"

    ]

    print(
        recommendations[
            display_columns
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # SIGNAL STATISTICS
    # -------------------------------------------------------------------------

    print(
        "\nSignal statistics:"
    )

    print(
        f"Content score range       : "
        f"{content_scores.min():.4f} - "
        f"{content_scores.max():.4f}"
    )

    print(
        f"Collaborative score range : "
        f"{collaborative_scores.min():.4f} - "
        f"{collaborative_scores.max():.4f}"
    )

    print(
        f"Difficulty fit range      : "
        f"{difficulty_scores.min():.4f} - "
        f"{difficulty_scores.max():.4f}"
    )


# =============================================================================
# 13. SAVE RESULTS
# =============================================================================

if not all_results:

    raise RuntimeError(
        "No hybrid recommendations were generated."
    )

final_results = pd.concat(
    all_results,
    ignore_index=True
)

final_results.to_csv(
    OUTPUT_FILE,
    index=False
)


# =============================================================================
# 14. FINAL VALIDATION STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

total_recommendations = len(
    final_results
)

nonzero_collaborative = (
    final_results[
        "collaborative_score"
    ] > 0
).sum()

collaborative_coverage = (
    nonzero_collaborative /
    total_recommendations
) * 100

average_content = (
    final_results[
        "content_score"
    ].mean()
)

average_collaborative = (
    final_results[
        "collaborative_score"
    ].mean()
)

average_difficulty = (
    final_results[
        "difficulty_fit"
    ].mean()
)

average_hybrid = (
    final_results[
        "hybrid_score"
    ].mean()
)

print(
    f"\nTotal recommendations checked: "
    f"{total_recommendations}"
)

print(
    f"Recommendations with non-zero "
    f"collaborative score: "
    f"{nonzero_collaborative}"
)

print(
    f"Collaborative signal coverage: "
    f"{collaborative_coverage:.2f}%"
)

print(
    f"\nAverage content score: "
    f"{average_content:.4f}"
)

print(
    f"Average collaborative score: "
    f"{average_collaborative:.4f}"
)

print(
    f"Average difficulty fit: "
    f"{average_difficulty:.4f}"
)

print(
    f"Average hybrid score: "
    f"{average_hybrid:.4f}"
)

print(
    f"\nResults saved to:\n"
    f"{OUTPUT_FILE}"
)


# =============================================================================
# 15. COMPLETION MESSAGE
# =============================================================================

print("\n" + "=" * 80)
print("STEP 13B COMPLETE")
print("=" * 80)

print(
    "\nValidated components:"
)

print(
    "1. TF-IDF Content-Based Filtering"
)

print(
    "2. KNN Collaborative Filtering"
)

print(
    "3. Learner Performance"
)

print(
    "4. Course Difficulty Fit"
)

print(
    "5. Hybrid Score Fusion"
)

print(
    "\nHybrid weights:"
)

print(
    "Content Score       : 40%"
)

print(
    "Collaborative Score : 30%"
)

print(
    "Performance Score   : 15%"
)

print(
    "Difficulty Fit      : 15%"
)

print(
    "\nIMPORTANT:"
)

print(
    "These weights are an experimental baseline."
)

print(
    "They are not claimed to be optimal."
)

print(
    "\nThe interaction data is synthetic and "
    "should not be presented as real student behavior."
)

print("=" * 80)