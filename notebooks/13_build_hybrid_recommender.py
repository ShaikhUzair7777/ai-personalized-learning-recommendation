import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# STEP 13 — HYBRID RECOMMENDATION ENGINE
# =============================================================================

print("=" * 80)
print("HYBRID PERSONALIZED LEARNING RECOMMENDATION ENGINE")
print("=" * 80)


# =============================================================================
# 1. PATHS
# =============================================================================

COURSE_FILE = "data/processed/courses_ready.csv"
LEARNER_FILE = "data/processed/learner_features_ready.csv"
INTERACTION_FILE = "data/processed/synthetic_interactions.csv"

TFIDF_MATRIX_FILE = "models/tfidf_matrix.npz"
TFIDF_VECTORIZER_FILE = "models/tfidf_vectorizer.joblib"

USER_COURSE_MATRIX_FILE = "models/user_course_matrix.npz"
USER_TO_INDEX_FILE = "models/user_to_index.joblib"
COURSE_TO_INDEX_FILE = "models/course_to_index.joblib"

OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
# 3. LOAD CONTENT-BASED MODEL
# =============================================================================

print("\nLoading TF-IDF content model...")

tfidf_matrix = load_npz(TFIDF_MATRIX_FILE)
tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_FILE)

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")


# =============================================================================
# 4. LOAD COLLABORATIVE FILTERING DATA
# =============================================================================

print("\nLoading collaborative filtering data...")

user_course_matrix = load_npz(USER_COURSE_MATRIX_FILE)

user_to_index = joblib.load(USER_TO_INDEX_FILE)
course_to_index = joblib.load(COURSE_TO_INDEX_FILE)

print(f"User-course matrix shape: {user_course_matrix.shape}")


# =============================================================================
# 5. PREPARE COURSE DATA
# =============================================================================

print("\nPreparing course features...")

courses["difficulty_normalized"] = (
    courses["difficulty_level"]
    .astype(str)
    .str.lower()
    .map({
        "beginner": 0.25,
        "basic": 0.25,
        "intermediate": 0.50,
        "mixed": 0.60,
        "advanced": 0.85
    })
    .fillna(0.50)
)

courses["course_rating"] = pd.to_numeric(
    courses["course_rating"],
    errors="coerce"
)

rating_min = courses["course_rating"].min()
rating_max = courses["course_rating"].max()

if rating_max > rating_min:
    courses["rating_normalized"] = (
        (courses["course_rating"] - rating_min)
        / (rating_max - rating_min)
    )
else:
    courses["rating_normalized"] = 0.5

print("Course features prepared.")


# =============================================================================
# 6. CREATE COURSE INDEX LOOKUP
# =============================================================================

course_id_to_row = {
    course_id: idx
    for idx, course_id in enumerate(courses["course_id"])
}

row_to_course_id = {
    idx: course_id
    for idx, course_id in enumerate(courses["course_id"])
}


# =============================================================================
# 7. DIFFICULTY FIT FUNCTION
# =============================================================================

def calculate_difficulty_fit(performance_level, course_difficulty):
    """
    Estimates how suitable a course difficulty is for the learner.
    """

    performance_level = str(performance_level).lower()
    course_difficulty = str(course_difficulty).lower()

    # Beginner learner
    if performance_level == "low":

        if course_difficulty in ["beginner", "basic"]:
            return 1.00

        elif course_difficulty == "intermediate":
            return 0.65

        elif course_difficulty == "advanced":
            return 0.25

        else:
            return 0.60

    # Medium learner
    elif performance_level == "medium":

        if course_difficulty == "intermediate":
            return 1.00

        elif course_difficulty in ["beginner", "basic"]:
            return 0.75

        elif course_difficulty == "advanced":
            return 0.65

        else:
            return 0.75

    # High learner
    elif performance_level == "high":

        if course_difficulty == "advanced":
            return 1.00

        elif course_difficulty == "intermediate":
            return 0.85

        elif course_difficulty in ["beginner", "basic"]:
            return 0.55

        else:
            return 0.75

    return 0.60


# =============================================================================
# 8. COLLABORATIVE SCORE
# =============================================================================

def get_collaborative_scores(user_id):
    """
    Generates collaborative filtering scores for all courses
    using the user-based interaction matrix.

    Returns:
        numpy array with one score per course.
    """

    if user_id not in user_to_index:
        return np.zeros(len(courses))

    user_index = user_to_index[user_id]

    user_vector = user_course_matrix[user_index]

    # Similarity between this learner and every learner.
    similarities = cosine_similarity(
        user_vector,
        user_course_matrix
    ).flatten()

    # Do not use the learner's own interactions as neighbors.
    similarities[user_index] = 0

    # Top 10 similar learners
    top_neighbor_indices = np.argsort(
        similarities
    )[-10:][::-1]

    neighbor_similarities = similarities[top_neighbor_indices]

    # Weighted aggregation of neighbor interactions.
    weighted_scores = (
        user_course_matrix[top_neighbor_indices]
        .multiply(neighbor_similarities[:, np.newaxis])
        .sum(axis=0)
    )

    weighted_scores = np.asarray(
        weighted_scores
    ).flatten()

    # Normalize to 0–1.
    max_score = weighted_scores.max()

    if max_score > 0:
        weighted_scores = weighted_scores / max_score

    return weighted_scores


# =============================================================================
# 9. CONTENT SCORE
# =============================================================================

def get_content_scores(user_profile):
    """
    Generates TF-IDF content similarity scores for all courses.
    """

    profile_vector = tfidf_vectorizer.transform(
        [user_profile]
    )

    scores = cosine_similarity(
        profile_vector,
        tfidf_matrix
    ).flatten()

    return scores


# =============================================================================
# 10. HYBRID RECOMMENDER
# =============================================================================

def recommend_hybrid(
    user_id,
    user_profile,
    top_k=10
):

    print("\n" + "-" * 80)
    print(f"Generating recommendations for learner: {user_id}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # CONTENT SCORE
    # -------------------------------------------------------------------------

    content_scores = get_content_scores(
        user_profile
    )

    # -------------------------------------------------------------------------
    # COLLABORATIVE SCORE
    # -------------------------------------------------------------------------

    collaborative_scores = get_collaborative_scores(
        user_id
    )

    # -------------------------------------------------------------------------
    # FIND LEARNER PROFILE
    # -------------------------------------------------------------------------

    learner_rows = learners[
        learners["id_student"] == user_id
    ]

    if len(learner_rows) == 0:
        raise ValueError(
            f"Learner {user_id} not found."
        )

    learner = learner_rows.iloc[0]

    performance_score = float(
        learner["performance_score"]
    )

    performance_level = learner[
        "performance_level"
    ]

    # -------------------------------------------------------------------------
    # COURSE-LEVEL PERSONALIZATION
    # -------------------------------------------------------------------------

    difficulty_scores = []

    for _, course in courses.iterrows():

        score = calculate_difficulty_fit(
            performance_level,
            course["difficulty_level"]
        )

        difficulty_scores.append(score)

    difficulty_scores = np.array(
        difficulty_scores
    )

    # -------------------------------------------------------------------------
    # NORMALIZE PERFORMANCE
    # -------------------------------------------------------------------------

    performance_score_normalized = np.clip(
        performance_score / 100,
        0,
        1
    )

    performance_scores = np.full(
        len(courses),
        performance_score_normalized
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

            row_index = course_id_to_row[
                course_id
            ]

            hybrid_scores[row_index] = -1

    # -------------------------------------------------------------------------
    # TOP K
    # -------------------------------------------------------------------------

    top_indices = np.argsort(
        hybrid_scores
    )[-top_k:][::-1]

    recommendations = courses.iloc[
        top_indices
    ].copy()

    recommendations[
        "content_score"
    ] = content_scores[top_indices]

    recommendations[
        "collaborative_score"
    ] = collaborative_scores[top_indices]

    recommendations[
        "performance_score"
    ] = performance_scores[top_indices]

    recommendations[
        "difficulty_fit"
    ] = difficulty_scores[top_indices]

    recommendations[
        "hybrid_score"
    ] = hybrid_scores[top_indices]

    recommendations = recommendations[
        [
            "course_id",
            "course_name",
            "university",
            "difficulty_level",
            "course_rating",
            "content_score",
            "collaborative_score",
            "performance_score",
            "difficulty_fit",
            "hybrid_score",
            "course_url"
        ]
    ]

    return recommendations.reset_index(
        drop=True
    )


# =============================================================================
# 11. TEST LEARNERS
# =============================================================================

print("\n" + "=" * 80)
print("TESTING HYBRID RECOMMENDER")
print("=" * 80)


test_profiles = [
    {
        "user_id": int(learners.iloc[0]["id_student"]),
        "profile": """
        Python Data Science Machine Learning
        Data Analysis SQL Statistics
        Data Visualization Pandas NumPy
        """
    },
    {
        "user_id": int(learners.iloc[1]["id_student"]),
        "profile": """
        Web Development HTML CSS JavaScript
        React Node.js Express MongoDB
        Frontend Backend Full Stack
        """
    },
    {
        "user_id": int(learners.iloc[2]["id_student"]),
        "profile": """
        Blockchain cryptocurrency Ethereum
        Smart Contracts Solidity Web3
        Decentralized Applications distributed systems
        """
    }
]


all_results = []


for test in test_profiles:

    user_id = test["user_id"]

    recommendations = recommend_hybrid(
        user_id=user_id,
        user_profile=test["profile"],
        top_k=10
    )

    print("\nTop 10 Recommendations:")
    print()

    display_columns = [
        "course_id",
        "course_name",
        "difficulty_level",
        "content_score",
        "collaborative_score",
        "difficulty_fit",
        "hybrid_score"
    ]

    print(
        recommendations[
            display_columns
        ].to_string(index=False)
    )

    temp = recommendations.copy()

    temp["test_user_id"] = user_id

    all_results.append(temp)


# =============================================================================
# 12. SAVE TEST RESULTS
# =============================================================================

final_results = pd.concat(
    all_results,
    ignore_index=True
)

output_file = (
    "data/processed/"
    "hybrid_recommendation_test_results.csv"
)

final_results.to_csv(
    output_file,
    index=False
)


# =============================================================================
# 13. SAVE HYBRID CONFIGURATION
# =============================================================================

hybrid_config = {
    "content_weight": 0.40,
    "collaborative_weight": 0.30,
    "performance_weight": 0.15,
    "difficulty_weight": 0.15,
    "top_k": 10
}

config_file = (
    "models/hybrid_config.joblib"
)

joblib.dump(
    hybrid_config,
    config_file
)


# =============================================================================
# 14. FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("STEP 13 COMPLETE")
print("=" * 80)

print("\nHybrid components:")
print("1. Content-Based Filtering → TF-IDF + Cosine Similarity")
print("2. Collaborative Filtering → User-Based KNN")
print("3. Learner Performance")
print("4. Difficulty Fit")

print("\nHybrid weights:")
print("Content Score       : 40%")
print("Collaborative Score : 30%")
print("Performance Score   : 15%")
print("Difficulty Fit      : 15%")

print("\nSaved:")
print(f"1. {output_file}")
print(f"2. {config_file}")

print("\nIMPORTANT:")
print("The current hybrid weights are an experimental baseline.")
print("They are not claimed to be optimal.")

print("\n" + "=" * 80)