"""
STEP 14
Rigorous Evaluation of Recommendation Models

Models:
1. Popularity Baseline
2. TF-IDF Content-Based
3. KNN Collaborative Filtering
4. Hybrid Recommendation

Metrics:
- Precision@5
- Precision@10
- Recall@5
- Recall@10
- NDCG@5
- NDCG@10

IMPORTANT:
The interaction dataset is synthetic and is used only for
controlled recommender-system experimentation.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse import load_npz

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

INTERACTIONS_FILE = (
    DATA_PROCESSED / "synthetic_interactions.csv"
)

COURSES_FILE = (
    DATA_PROCESSED / "courses_ready.csv"
)

LEARNERS_FILE = (
    DATA_PROCESSED / "learner_features_ready.csv"
)

TFIDF_MATRIX_FILE = (
    MODELS / "tfidf_matrix.npz"
)

OUTPUT_FILE = (
    DATA_PROCESSED / "hybrid_evaluation_results.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TOP_K_VALUES = [5, 10]

KNN_NEIGHBORS = 10

warnings.filterwarnings("ignore")


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 14 - RIGOROUS HYBRID EVALUATION")
print("=" * 70)

print("\nLoading datasets...")

interactions = pd.read_csv(
    INTERACTIONS_FILE
)

courses = pd.read_csv(
    COURSES_FILE
)

learners = pd.read_csv(
    LEARNERS_FILE
)

tfidf_matrix = load_npz(
    TFIDF_MATRIX_FILE
).tocsr()

print(
    f"Interactions: {len(interactions):,}"
)

print(
    f"Courses:      {len(courses):,}"
)

print(
    f"Learners:     {len(learners):,}"
)

print(
    f"TF-IDF shape: {tfidf_matrix.shape}"
)


# ============================================================
# 4. VALIDATE DATASET SCHEMA
# ============================================================

required_interaction_columns = {
    "user_id",
    "course_id",
    "interaction_type",
    "interaction_value",
    "preference_score"
}

required_course_columns = {
    "course_id",
    "course_name",
    "difficulty_level"
}

required_learner_columns = {
    "id_student",
    "average_score"
}

missing_interaction = (
    required_interaction_columns
    - set(interactions.columns)
)

missing_courses = (
    required_course_columns
    - set(courses.columns)
)

missing_learners = (
    required_learner_columns
    - set(learners.columns)
)

if missing_interaction:
    raise ValueError(
        f"Missing interaction columns: "
        f"{missing_interaction}"
    )

if missing_courses:
    raise ValueError(
        f"Missing course columns: "
        f"{missing_courses}"
    )

if missing_learners:
    raise ValueError(
        f"Missing learner columns: "
        f"{missing_learners}"
    )


# ============================================================
# 5. STANDARDIZE TYPES
# ============================================================

interactions["user_id"] = (
    interactions["user_id"]
    .astype(int)
)

interactions["course_id"] = (
    interactions["course_id"]
    .astype(str)
)

interactions["preference_score"] = (
    interactions["preference_score"]
    .astype(float)
)

interactions["interaction_value"] = (
    interactions["interaction_value"]
    .astype(float)
)


# ============================================================
# 6. CREATE INTERACTION STRENGTH
# ============================================================

print("\nCreating interaction strength...")

# interaction_value already contains:
#
# clicked    = 0.25
# viewed     = 0.50
# started    = 0.75
# completed  = 1.00

interactions["interaction_strength"] = (
    interactions["interaction_value"]
    * interactions["preference_score"]
)

print(
    "Interaction strength statistics:"
)

print(
    interactions[
        "interaction_strength"
    ].describe().to_string()
)


# ============================================================
# 7. COURSE MAPPINGS
# ============================================================

print("\nCreating course mappings...")

course_ids = (
    courses["course_id"]
    .astype(str)
    .tolist()
)

course_to_catalog_index = {
    course_id: idx
    for idx, course_id
    in enumerate(course_ids)
}

catalog_index_to_course = {
    idx: course_id
    for idx, course_id
    in enumerate(course_ids)
}

course_difficulty = (
    courses["difficulty_level"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# 8. LEARNER MAPPINGS
# ============================================================

learner_ids = (
    learners["id_student"]
    .astype(int)
    .tolist()
)

learner_to_index = {
    learner_id: idx
    for idx, learner_id
    in enumerate(learner_ids)
}

learner_index_to_id = {
    idx: learner_id
    for idx, learner_id
    in enumerate(learner_ids)
}


# ============================================================
# 9. KEEP VALID INTERACTIONS
# ============================================================

valid_mask = (
    interactions["user_id"]
    .isin(learner_to_index)
    &
    interactions["course_id"]
    .isin(course_to_catalog_index)
)

interactions = (
    interactions[valid_mask]
    .copy()
)

print(
    f"\nValid interactions after mapping: "
    f"{len(interactions):,}"
)


# ============================================================
# 10. PER-USER TRAIN / TEST HOLDOUT
# ============================================================

print(
    "\nCreating per-user train/test split..."
)

rng = np.random.default_rng(
    RANDOM_STATE
)

test_indices = []

for user_id, group in interactions.groupby(
    "user_id",
    sort=False
):

    selected_index = rng.choice(
        group.index.to_numpy()
    )

    test_indices.append(
        selected_index
    )

test_indices = set(
    test_indices
)

test_data = (
    interactions
    .loc[sorted(test_indices)]
    .copy()
)

train_data = (
    interactions
    .drop(index=test_indices)
    .copy()
)

print(
    f"Training interactions: "
    f"{len(train_data):,}"
)

print(
    f"Test interactions:     "
    f"{len(test_data):,}"
)

print(
    f"Test users:            "
    f"{test_data['user_id'].nunique():,}"
)


# ============================================================
# 11. BUILD TRAINING USER-COURSE MATRIX
# ============================================================

print(
    "\nBuilding training user-course matrix..."
)

num_users = len(learner_ids)
num_courses = len(course_ids)

train_rows = []
train_cols = []
train_values = []

for row in train_data.itertuples(
    index=False
):

    user_index = learner_to_index[
        row.user_id
    ]

    course_index = (
        course_to_catalog_index[
            row.course_id
        ]
    )

    train_rows.append(
        user_index
    )

    train_cols.append(
        course_index
    )

    train_values.append(
        row.interaction_strength
    )

train_catalog_matrix = csr_matrix(
    (
        train_values,
        (
            train_rows,
            train_cols
        )
    ),
    shape=(
        num_users,
        num_courses
    )
)

print(
    f"Training matrix shape: "
    f"{train_catalog_matrix.shape}"
)

print(
    f"Non-zero training interactions: "
    f"{train_catalog_matrix.nnz:,}"
)


# ============================================================
# 12. BUILD CONTENT-BASED LEARNER PROFILES
# ============================================================

print(
    "\nBuilding content-based learner profiles..."
)

tfidf_normalized = normalize(
    tfidf_matrix,
    norm="l2",
    axis=1
)

user_weight_sums = np.asarray(
    train_catalog_matrix.sum(
        axis=1
    )
).ravel()

user_weight_sums[
    user_weight_sums == 0
] = 1.0

normalized_train_matrix = (
    train_catalog_matrix.multiply(
        1.0 /
        user_weight_sums[:, None]
    )
)

profile_matrix = (
    normalized_train_matrix
    @ tfidf_normalized
)

profile_matrix = normalize(
    profile_matrix,
    norm="l2",
    axis=1
)

print(
    f"Learner profile matrix: "
    f"{profile_matrix.shape}"
)


# ============================================================
# 13. CONTENT SCORE MATRIX
# ============================================================

print(
    "\nCalculating content-based scores..."
)

content_score_matrix = (
    profile_matrix
    @ tfidf_normalized.T
)

content_score_matrix = (
    content_score_matrix.toarray()
)

content_score_matrix = np.clip(
    content_score_matrix,
    0,
    1
)

print(
    f"Content score matrix: "
    f"{content_score_matrix.shape}"
)


# ============================================================
# 14. POPULARITY BASELINE
# ============================================================

print(
    "\nBuilding popularity baseline..."
)

popularity_scores = np.asarray(
    train_catalog_matrix.sum(
        axis=0
    )
).ravel()

print(
    "Popularity baseline ready."
)


# ============================================================
# 15. TRAIN KNN COLLABORATIVE MODEL
# ============================================================

print(
    "\nTraining KNN collaborative model..."
)

knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=KNN_NEIGHBORS + 1,
    n_jobs=1
)

knn.fit(
    train_catalog_matrix
)

print(
    "Finding nearest learners..."
)

distances, neighbor_indices = (
    knn.kneighbors(
        train_catalog_matrix,
        n_neighbors=KNN_NEIGHBORS + 1
    )
)

similarities = (
    1.0 - distances
)

print(
    f"KNN trained with "
    f"{KNN_NEIGHBORS} neighbors."
)


# ============================================================
# 16. PERFORMANCE / DIFFICULTY PERSONALIZATION
# ============================================================

print(
    "\nPreparing performance-difficulty "
    "personalization..."
)

learner_performance = (
    learners
    .set_index("id_student")
    ["average_score"]
    .fillna(0)
    .astype(float)
    / 100.0
)

difficulty_normalization = {
    "beginner": 0.25,
    "intermediate": 0.55,
    "advanced": 0.85
}


def get_performance_difficulty_scores(
    user_id
):

    performance = float(
        learner_performance.get(
            user_id,
            0.0
        )
    )

    scores = np.zeros(
        num_courses,
        dtype=float
    )

    for course_index in range(
        num_courses
    ):

        difficulty = (
            course_difficulty.iloc[
                course_index
            ]
        )

        difficulty_value = (
            difficulty_normalization.get(
                difficulty,
                0.55
            )
        )

        fit = (
            1.0
            - abs(
                performance
                - difficulty_value
            )
        )

        scores[course_index] = (
            np.clip(
                fit,
                0.0,
                1.0
            )
        )

    return scores


# ============================================================
# 17. HELPER FUNCTIONS
# ============================================================

def get_seen_courses(
    user_index
):

    row = (
        train_catalog_matrix
        .getrow(user_index)
    )

    return set(
        row.indices.tolist()
    )


def get_knn_scores(
    user_index
):

    neighbors = (
        neighbor_indices[
            user_index
        ]
    )

    neighbor_sims = (
        similarities[
            user_index
        ]
    )

    # Remove the learner themselves.
    valid_mask = (
        neighbors
        != user_index
    )

    neighbors = (
        neighbors[
            valid_mask
        ]
    )

    neighbor_sims = (
        neighbor_sims[
            valid_mask
        ]
    )

    weighted_neighbors = (
        train_catalog_matrix[
            neighbors
        ].multiply(
            neighbor_sims[:, None]
        )
    )

    collaborative_scores = (
        np.asarray(
            weighted_neighbors.sum(
                axis=0
            )
        ).ravel()
    )

    maximum = (
        collaborative_scores.max()
    )

    if maximum > 0:

        collaborative_scores = (
            collaborative_scores
            / maximum
        )

    return collaborative_scores


def get_top_courses(
    scores,
    seen_courses,
    top_k=10
):

    scores = np.asarray(
        scores,
        dtype=float
    ).copy()

    if seen_courses:

        scores[
            list(seen_courses)
        ] = -np.inf

    available = np.isfinite(
        scores
    ).sum()

    actual_k = min(
        top_k,
        available
    )

    if actual_k <= 0:

        return []

    candidate_indices = (
        np.argpartition(
            -scores,
            actual_k - 1
        )[:actual_k]
    )

    candidate_indices = (
        candidate_indices[
            np.argsort(
                -scores[
                    candidate_indices
                ]
            )
        ]
    )

    return (
        candidate_indices.tolist()
    )


def calculate_metrics(
    recommended_courses,
    test_course,
    k
):

    recommended = (
        recommended_courses[:k]
    )

    if test_course in recommended:

        rank = (
            recommended.index(
                test_course
            )
            + 1
        )

        precision = (
            1.0 / k
        )

        recall = 1.0

        ndcg = (
            1.0
            / np.log2(
                rank + 1
            )
        )

    else:

        precision = 0.0
        recall = 0.0
        ndcg = 0.0

    return (
        precision,
        recall,
        ndcg
    )


# ============================================================
# 18. TEST LOOKUP
# ============================================================

test_lookup = {
    int(row.user_id):
        str(row.course_id)

    for row in test_data.itertuples(
        index=False
    )
}

total_users = len(
    test_lookup
)


# ============================================================
# 19. MODEL EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("STARTING MODEL EVALUATION")
print("=" * 70)

results = []

for counter, user_id in enumerate(
    test_lookup.keys(),
    start=1
):

    user_index = (
        learner_to_index[
            user_id
        ]
    )

    test_course_id = (
        test_lookup[
            user_id
        ]
    )

    test_course_index = (
        course_to_catalog_index[
            test_course_id
        ]
    )

    seen_courses = (
        get_seen_courses(
            user_index
        )
    )

    # --------------------------------------------------------
    # POPULARITY
    # --------------------------------------------------------

    popularity_recommendations = (
        get_top_courses(
            popularity_scores,
            seen_courses,
            top_k=10
        )
    )

    # --------------------------------------------------------
    # CONTENT
    # --------------------------------------------------------

    content_scores = (
        content_score_matrix[
            user_index
        ]
    )

    content_recommendations = (
        get_top_courses(
            content_scores,
            seen_courses,
            top_k=10
        )
    )

    # --------------------------------------------------------
    # COLLABORATIVE
    # --------------------------------------------------------

    collaborative_scores = (
        get_knn_scores(
            user_index
        )
    )

    collaborative_recommendations = (
        get_top_courses(
            collaborative_scores,
            seen_courses,
            top_k=10
        )
    )

    # --------------------------------------------------------
    # PERFORMANCE / DIFFICULTY
    # --------------------------------------------------------

    performance_difficulty_scores = (
        get_performance_difficulty_scores(
            user_id
        )
    )

    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    hybrid_scores = (
        0.40 * content_scores
        +
        0.30 * collaborative_scores
        +
        0.30 * performance_difficulty_scores
    )

    hybrid_recommendations = (
        get_top_courses(
            hybrid_scores,
            seen_courses,
            top_k=10
        )
    )

    model_recommendations = {

        "Popularity":
            popularity_recommendations,

        "Content-Based":
            content_recommendations,

        "KNN Collaborative":
            collaborative_recommendations,

        "Hybrid":
            hybrid_recommendations
    }

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    for model_name, recommendations in (
        model_recommendations.items()
    ):

        for k in TOP_K_VALUES:

            precision, recall, ndcg = (
                calculate_metrics(
                    recommendations,
                    test_course_index,
                    k
                )
            )

            results.append({

                "user_id":
                    user_id,

                "model":
                    model_name,

                "k":
                    k,

                "precision":
                    precision,

                "recall":
                    recall,

                "ndcg":
                    ndcg
            })

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        counter % 500 == 0
        or counter == total_users
    ):

        print(
            f"Evaluated "
            f"{counter:,}/"
            f"{total_users:,} learners..."
        )


# ============================================================
# 20. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

print(
    "\nEvaluation completed."
)


# ============================================================
# 21. AGGREGATED RESULTS
# ============================================================

summary = (
    results_df
    .groupby(
        ["model", "k"],
        as_index=False
    )
    [
        [
            "precision",
            "recall",
            "ndcg"
        ]
    ]
    .mean()
)

summary = summary.sort_values(
    ["k", "model"]
)


# ============================================================
# 22. PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL EVALUATION RESULTS")
print("=" * 70)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


# ============================================================
# 23. SAVE DETAILED RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nDetailed results saved to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 24. SAVE SUMMARY
# ============================================================

summary_file = (
    DATA_PROCESSED
    / "hybrid_evaluation_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print(
    "\nSummary results saved to:"
)

print(
    summary_file
)


# ============================================================
# 25. COMPLETION MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("STEP 14 COMPLETE")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "These results use synthetic interactions "
    "for controlled experimentation."
)

print(
    "Precision, Recall and NDCG are ranking "
    "metrics, not classification accuracy."
)

print(
    "Hybrid weights are experimental baseline "
    "weights and are not claimed to be optimal."
)