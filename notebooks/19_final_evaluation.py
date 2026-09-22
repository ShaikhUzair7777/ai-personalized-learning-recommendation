"""
STEP 19 — FINAL ML EVALUATION

AI-Powered Personalized Learning Recommendation System
Using Hybrid Machine Learning

Evaluates:
1. Popularity Baseline
2. Content-Based Filtering
3. KNN Collaborative Filtering
4. Hybrid Recommendation

Metrics:
- Precision@5
- Recall@5
- NDCG@5
- Precision@10
- Recall@10
- NDCG@10

IMPORTANT:
This is a controlled evaluation using the project's synthetic
interaction dataset. Results must not be described as real-world
validation or recommendation accuracy.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import joblib

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used"
)

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

INTERACTIONS_FILE = DATA_DIR / "synthetic_interactions.csv"
COURSES_FILE = DATA_DIR / "courses_ready.csv"
LEARNERS_FILE = DATA_DIR / "learner_features_ready.csv"

OUTPUT_FILE = DATA_DIR / "final_evaluation_results.csv"

TOP_K_VALUES = [5, 10]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_string(value):
    """Convert an ID/value safely to string."""
    if pd.isna(value):
        return ""
    return str(value)


def minmax_normalize(values):
    """
    Normalize values into [0, 1].

    Handles:
    - empty arrays
    - constant arrays
    - NaN / infinity
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return values

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    minimum = values.min()
    maximum = values.max()

    if maximum <= minimum:
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


def precision_at_k(recommended, relevant, k):
    """Calculate Precision@K."""
    recommended = recommended[:k]

    if k <= 0:
        return 0.0

    hits = sum(
        1
        for item in recommended
        if item in relevant
    )

    return hits / k


def recall_at_k(recommended, relevant, k):
    """Calculate Recall@K."""
    if not relevant:
        return 0.0

    recommended = recommended[:k]

    hits = sum(
        1
        for item in recommended
        if item in relevant
    )

    return hits / len(relevant)


def dcg(relevances):
    """Calculate discounted cumulative gain."""
    if not relevances:
        return 0.0

    total = 0.0

    for position, relevance in enumerate(
        relevances,
        start=1
    ):
        total += relevance / np.log2(position + 1)

    return total


def ndcg_at_k(recommended, relevant, k):
    """Calculate NDCG@K."""
    recommended = recommended[:k]

    relevances = [
        1 if item in relevant else 0
        for item in recommended
    ]

    actual_dcg = dcg(relevances)

    ideal_relevances = sorted(
        relevances,
        reverse=True
    )

    ideal_dcg = dcg(
        ideal_relevances
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 72)
print("STEP 19 — FINAL ML EVALUATION")
print("=" * 72)

print("\n[1/8] Loading datasets...")

interactions = pd.read_csv(
    INTERACTIONS_FILE
)

courses = pd.read_csv(
    COURSES_FILE
)

learners = pd.read_csv(
    LEARNERS_FILE
)

print(
    f"Interactions : {len(interactions):,}"
)

print(
    f"Courses      : {len(courses):,}"
)

print(
    f"Learners     : {len(learners):,}"
)

print(
    "\nCourse columns detected:"
)

print(
    list(courses.columns)
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_interaction_columns = {
    "user_id",
    "course_id",
    "interaction_type",
    "interaction_value",
    "preference_score"
}

missing_interaction_columns = (
    required_interaction_columns
    - set(interactions.columns)
)

if missing_interaction_columns:
    raise ValueError(
        "Missing interaction columns: "
        + str(sorted(missing_interaction_columns))
    )


if "course_id" not in courses.columns:
    raise ValueError(
        "courses_ready.csv must contain course_id."
    )


# ============================================================
# NORMALIZE IDS
# ============================================================

interactions["user_id"] = (
    interactions["user_id"]
    .map(safe_string)
)

interactions["course_id"] = (
    interactions["course_id"]
    .map(safe_string)
)

courses["course_id"] = (
    courses["course_id"]
    .map(safe_string)
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

print("\n[2/8] Loading model artifacts...")

tfidf_matrix = sparse.load_npz(
    MODEL_DIR / "tfidf_matrix.npz"
)

tfidf_vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.joblib"
)

knn_model = joblib.load(
    MODEL_DIR / "knn_collaborative_model.joblib"
)

user_to_index = joblib.load(
    MODEL_DIR / "user_to_index.joblib"
)

index_to_user = joblib.load(
    MODEL_DIR / "index_to_user.joblib"
)

course_to_index = joblib.load(
    MODEL_DIR / "course_to_index.joblib"
)

index_to_course = joblib.load(
    MODEL_DIR / "index_to_course.joblib"
)

user_course_matrix = sparse.load_npz(
    MODEL_DIR / "user_course_matrix.npz"
)

hybrid_config = joblib.load(
    MODEL_DIR / "hybrid_config.joblib"
)

print(
    f"TF-IDF matrix       : {tfidf_matrix.shape}"
)

print(
    f"KNN matrix          : {user_course_matrix.shape}"
)

print(
    f"KNN users           : {len(user_to_index):,}"
)

print(
    f"KNN courses         : {len(course_to_index):,}"
)

print(
    "\nHybrid configuration:"
)

print(hybrid_config)


# ============================================================
# BUILD AUTHORITATIVE COURSE MAPPINGS
# ============================================================

print("\n[3/8] Preparing course mappings...")

course_dataframe_index = {}

for row_position, course_id in enumerate(
    courses["course_id"]
):
    course_dataframe_index[course_id] = row_position


# Authoritative TF-IDF course ordering.
tfidf_course_ids = []

for i in range(tfidf_matrix.shape[0]):

    if i in index_to_course:
        course_id = safe_string(
            index_to_course[i]
        )

    elif str(i) in index_to_course:
        course_id = safe_string(
            index_to_course[str(i)]
        )

    else:
        course_id = ""

    tfidf_course_ids.append(
        course_id
    )


tfidf_course_index = {
    course_id: index
    for index, course_id
    in enumerate(tfidf_course_ids)
    if course_id
}


# ============================================================
# BUILD COURSE DIFFICULTY LOOKUP ONCE
# ============================================================

print(
    "\nPreparing course difficulty lookup..."
)

difficulty_column_candidates = [
    "difficulty_level",
    "course_difficulty",
    "difficulty",
    "level"
]

difficulty_column = None

for candidate in difficulty_column_candidates:

    if candidate in courses.columns:
        difficulty_column = candidate
        break


print(
    f"Difficulty column: {difficulty_column}"
)


course_difficulty_scores = {}


for _, row in courses.iterrows():

    course_id = safe_string(
        row["course_id"]
    )

    if difficulty_column is None:

        difficulty_score = 0.50

    else:

        value = safe_string(
            row[difficulty_column]
        ).lower()

        if (
            "beginner" in value
            or "basic" in value
            or "easy" in value
        ):
            difficulty_score = 0.25

        elif (
            "intermediate" in value
            or "medium" in value
            or "moderate" in value
        ):
            difficulty_score = 0.50

        elif (
            "advanced" in value
            or "expert" in value
            or "hard" in value
        ):
            difficulty_score = 0.75

        else:
            difficulty_score = 0.50

    course_difficulty_scores[
        course_id
    ] = difficulty_score


# ============================================================
# BUILD LEARNER PERFORMANCE LOOKUP ONCE
# ============================================================

learner_performance = {}

if "id_student" in learners.columns:

    performance_column = None

    for candidate in [
        "performance_score",
        "average_score",
        "score"
    ]:

        if candidate in learners.columns:
            performance_column = candidate
            break

    if performance_column:

        for _, row in learners.iterrows():

            learner_id = safe_string(
                row["id_student"]
            )

            try:
                score = float(
                    row[performance_column]
                )
            except (
                ValueError,
                TypeError
            ):
                continue

            if np.isfinite(score):

                # If stored as percentage, convert to [0,1].
                if score > 1:
                    score = score / 100.0

                learner_performance[
                    learner_id
                ] = float(
                    np.clip(
                        score,
                        0.0,
                        1.0
                    )
                )


print(
    f"Learner performance lookup: "
    f"{len(learner_performance):,}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print(
    "\n[4/8] Preparing train/test split..."
)

# Preserve original interaction order.
interactions["_row_order"] = np.arange(
    len(interactions)
)

user_counts = (
    interactions
    .groupby("user_id")
    .size()
)

eligible_users = set(
    user_counts[
        user_counts >= 2
    ].index
)

evaluation_data = interactions[
    interactions["user_id"].isin(
        eligible_users
    )
].copy()

evaluation_data = evaluation_data.sort_values(
    [
        "user_id",
        "_row_order"
    ]
)

test = (
    evaluation_data
    .groupby(
        "user_id",
        sort=False
    )
    .tail(1)
    .copy()
)

train = evaluation_data.drop(
    index=test.index
).copy()

print(
    f"Eligible users : {len(eligible_users):,}"
)

print(
    f"Train rows     : {len(train):,}"
)

print(
    f"Test rows      : {len(test):,}"
)


# ============================================================
# BUILD USER HISTORY ONCE
# ============================================================

print(
    "\nBuilding user training histories..."
)

user_history = (
    train
    .groupby("user_id")["course_id"]
    .agg(lambda values: set(values))
    .to_dict()
)


# ============================================================
# POPULARITY BASELINE
# ============================================================

print(
    "\n[5/8] Building popularity baseline..."
)

interaction_weights = {
    "clicked": 0.25,
    "viewed": 0.50,
    "started": 0.75,
    "completed": 1.00
}

train["interaction_weight"] = (
    train["interaction_type"]
    .map(interaction_weights)
)

train["interaction_weight"] = (
    train["interaction_weight"]
    .fillna(
        pd.to_numeric(
            train["interaction_value"],
            errors="coerce"
        ).fillna(0.0)
    )
)

train["weighted_preference"] = (
    train["interaction_weight"]
    * pd.to_numeric(
        train["preference_score"],
        errors="coerce"
    ).fillna(0.0)
)

popularity_series = (
    train
    .groupby("course_id")[
        "weighted_preference"
    ]
    .sum()
    .sort_values(
        ascending=False
    )
)

popular_courses = [
    safe_string(course_id)
    for course_id
    in popularity_series.index
]


# ============================================================
# CONTENT-BASED MODEL
# ============================================================

print(
    "\n[6/8] Preparing recommendation functions..."
)


def get_user_content_profile(user_id):
    """
    Create a user's TF-IDF profile from training history.

    Only training interactions are used.
    """

    history = user_history.get(
        user_id,
        set()
    )

    valid_indices = []

    for course_id in history:

        if course_id in tfidf_course_index:

            valid_indices.append(
                tfidf_course_index[
                    course_id
                ]
            )

    if not valid_indices:
        return None

    profile = (
        tfidf_matrix[
            valid_indices
        ]
        .mean(axis=0)
    )

    # Critical fix:
    # scipy sparse mean can return np.matrix.
    profile = np.asarray(
        profile
    )

    if profile.ndim == 1:
        profile = profile.reshape(
            1,
            -1
        )

    return profile


def content_recommendations(
    user_id,
    top_n=10
):
    """
    Content-based recommendations.
    """

    history = user_history.get(
        user_id,
        set()
    )

    profile = get_user_content_profile(
        user_id
    )

    if profile is None:
        return [
            course
            for course in popular_courses
            if course not in history
        ][:top_n]

    similarities = cosine_similarity(
        profile,
        tfidf_matrix
    ).ravel()

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    recommendations = []

    for index in ranked_indices:

        if index >= len(
            tfidf_course_ids
        ):
            continue

        course_id = (
            tfidf_course_ids[index]
        )

        if not course_id:
            continue

        if course_id in history:
            continue

        recommendations.append(
            course_id
        )

        if len(recommendations) >= top_n:
            break

    return recommendations


# ============================================================
# COLLABORATIVE RECOMMENDATIONS
# ============================================================

def collaborative_recommendations(
    user_id,
    top_n=10
):
    """
    User-based KNN collaborative recommendation.
    """

    history = user_history.get(
        user_id,
        set()
    )

    if user_id not in user_to_index:

        return [
            course
            for course in popular_courses
            if course not in history
        ][:top_n]

    user_index = user_to_index[
        user_id
    ]

    n_neighbors = min(
        knn_model.n_neighbors,
        user_course_matrix.shape[0]
    )

    distances, neighbors = (
        knn_model.kneighbors(
            user_course_matrix[
                user_index
            ],
            n_neighbors=n_neighbors
        )
    )

    scores = {}

    for neighbor_index, distance in zip(
        neighbors[0],
        distances[0]
    ):

        similarity = max(
            0.0,
            1.0 - float(distance)
        )

        if similarity <= 0:
            continue

        row = (
            user_course_matrix[
                neighbor_index
            ]
            .toarray()
            .ravel()
        )

        nonzero_indices = np.flatnonzero(
            row > 0
        )

        for course_index in (
            nonzero_indices
        ):

            if course_index not in index_to_course:
                continue

            course_id = safe_string(
                index_to_course[
                    course_index
                ]
            )

            if not course_id:
                continue

            if course_id in history:
                continue

            scores[course_id] = (
                scores.get(
                    course_id,
                    0.0
                )
                + similarity
                * float(
                    row[course_index]
                )
            )

    if not scores:

        return [
            course
            for course in popular_courses
            if course not in history
        ][:top_n]

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        course_id
        for course_id, _
        in ranked[:top_n]
    ]


# ============================================================
# COLLABORATIVE SCORE MAP
# ============================================================

def get_collaborative_score_map(
    user_id,
    candidate_courses
):
    """
    Generate normalized collaborative scores
    for a fixed candidate pool.
    """

    if user_id not in user_to_index:
        return {
            course: 0.0
            for course in candidate_courses
        }

    user_index = user_to_index[
        user_id
    ]

    n_neighbors = min(
        knn_model.n_neighbors,
        user_course_matrix.shape[0]
    )

    distances, neighbors = (
        knn_model.kneighbors(
            user_course_matrix[
                user_index
            ],
            n_neighbors=n_neighbors
        )
    )

    raw_scores = {
        course: 0.0
        for course in candidate_courses
    }

    candidate_set = set(
        candidate_courses
    )

    for neighbor_index, distance in zip(
        neighbors[0],
        distances[0]
    ):

        similarity = max(
            0.0,
            1.0 - float(distance)
        )

        if similarity <= 0:
            continue

        row = (
            user_course_matrix[
                neighbor_index
            ]
            .toarray()
            .ravel()
        )

        for course_index in np.flatnonzero(
            row > 0
        ):

            if course_index not in index_to_course:
                continue

            course_id = safe_string(
                index_to_course[
                    course_index
                ]
            )

            if course_id not in candidate_set:
                continue

            raw_scores[course_id] += (
                similarity
                * float(
                    row[course_index]
                )
            )

    values = np.array(
        [
            raw_scores[course]
            for course in candidate_courses
        ],
        dtype=float
    )

    normalized = minmax_normalize(
        values
    )

    return {
        course: float(score)
        for course, score
        in zip(
            candidate_courses,
            normalized
        )
    }


# ============================================================
# HYBRID RECOMMENDATION
# ============================================================

def hybrid_recommendations(
    user_id,
    top_n=10
):
    """
    Hybrid recommendation.

    The final model conceptually uses:
        40% skill
        30% content
        15% collaborative
        15% performance/difficulty

    During this offline synthetic-interaction evaluation,
    explicit learner skill labels are not available for the
    synthetic users.

    Therefore the skill component uses the user's learned
    content relevance as a controlled proxy.

    This must be documented as an evaluation limitation.
    """

    history = user_history.get(
        user_id,
        set()
    )

    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------

    content_pool = content_recommendations(
        user_id,
        top_n=100
    )

    collaborative_pool = collaborative_recommendations(
        user_id,
        top_n=100
    )

    popularity_pool = [
        course
        for course in popular_courses
        if course not in history
    ][:100]

    candidate_pool = list(
        dict.fromkeys(
            content_pool
            + collaborative_pool
            + popularity_pool
        )
    )

    if not candidate_pool:
        return []

    # --------------------------------------------------------
    # Content scores
    # --------------------------------------------------------

    profile = get_user_content_profile(
        user_id
    )

    content_scores = {
        course: 0.0
        for course in candidate_pool
    }

    valid_candidates = []

    valid_indices = []

    if profile is not None:

        for course_id in candidate_pool:

            if course_id in tfidf_course_index:

                valid_candidates.append(
                    course_id
                )

                valid_indices.append(
                    tfidf_course_index[
                        course_id
                    ]
                )

        if valid_indices:

            candidate_matrix = (
                tfidf_matrix[
                    valid_indices
                ]
            )

            similarity_values = (
                cosine_similarity(
                    profile,
                    candidate_matrix
                )
                .ravel()
            )

            for course_id, score in zip(
                valid_candidates,
                similarity_values
            ):

                content_scores[
                    course_id
                ] = float(
                    max(
                        0.0,
                        score
                    )
                )

    # --------------------------------------------------------
    # Skill proxy
    # --------------------------------------------------------

    skill_scores = {
        course: content_scores.get(
            course,
            0.0
        )
        for course in candidate_pool
    }

    # --------------------------------------------------------
    # Collaborative scores
    # --------------------------------------------------------

    collaborative_scores = (
        get_collaborative_score_map(
            user_id,
            candidate_pool
        )
    )

    # --------------------------------------------------------
    # Learner performance
    # --------------------------------------------------------

    learner_score = learner_performance.get(
        safe_string(user_id),
        None
    )

    # The synthetic user population is separate from
    # OULAD learner IDs. If no matching performance
    # record exists, use a neutral value.

    if learner_score is None:

        # Estimate only from TRAINING interactions.
        # This does not use the held-out test item.
        user_training = train[
            train["user_id"] == user_id
        ]

        if len(user_training) > 0:

            weighted_preference = (
                pd.to_numeric(
                    user_training[
                        "preference_score"
                    ],
                    errors="coerce"
                )
                .fillna(0.0)
                .mean()
            )

            learner_score = float(
                np.clip(
                    weighted_preference,
                    0.0,
                    1.0
                )
            )

        else:

            learner_score = 0.5

    # --------------------------------------------------------
    # Final hybrid ranking
    # --------------------------------------------------------

    hybrid_scores = {}

    for course_id in candidate_pool:

        skill_score = skill_scores.get(
            course_id,
            0.0
        )

        content_score = content_scores.get(
            course_id,
            0.0
        )

        collaborative_score = (
            collaborative_scores.get(
                course_id,
                0.0
            )
        )

        difficulty_score = (
            course_difficulty_scores.get(
                course_id,
                0.50
            )
        )

        performance_difficulty_score = (
            1.0
            - abs(
                learner_score
                - difficulty_score
            )
        )

        final_score = (
            0.40 * skill_score
            + 0.30 * content_score
            + 0.15 * collaborative_score
            + 0.15 * performance_difficulty_score
        )

        hybrid_scores[
            course_id
        ] = float(
            final_score
        )

    ranked = sorted(
        hybrid_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        course_id
        for course_id, _
        in ranked[:top_n]
    ]


# ============================================================
# EVALUATION
# ============================================================

print(
    "\n[7/8] Evaluating recommendation methods..."
)

evaluation_records = []

total_users = len(test)

for counter, (_, row) in enumerate(
    test.iterrows(),
    start=1
):

    user_id = safe_string(
        row["user_id"]
    )

    true_course = safe_string(
        row["course_id"]
    )

    relevant = {
        true_course
    }

    # --------------------------------------------------------
    # Generate recommendations.
    # Each function only uses TRAINING history.
    # --------------------------------------------------------

    recommendations = {
        "Popularity": [
            course
            for course in popular_courses
            if course not in user_history.get(
                user_id,
                set()
            )
        ][:10],

        "Content-Based": content_recommendations(
            user_id,
            top_n=10
        ),

        "KNN Collaborative": collaborative_recommendations(
            user_id,
            top_n=10
        ),

        "Hybrid": hybrid_recommendations(
            user_id,
            top_n=10
        )
    }

    # --------------------------------------------------------
    # Calculate metrics.
    # --------------------------------------------------------

    for method, recs in recommendations.items():

        for k in TOP_K_VALUES:

            evaluation_records.append(
                {
                    "user_id": user_id,
                    "method": method,
                    "k": k,
                    "precision": precision_at_k(
                        recs,
                        relevant,
                        k
                    ),
                    "recall": recall_at_k(
                        recs,
                        relevant,
                        k
                    ),
                    "ndcg": ndcg_at_k(
                        recs,
                        relevant,
                        k
                    )
                }
            )

    if (
        counter % 250 == 0
        or counter == total_users
    ):

        print(
            f"  Evaluated "
            f"{counter:,} / "
            f"{total_users:,} users"
        )


# ============================================================
# AGGREGATE RESULTS
# ============================================================

print(
    "\n[8/8] Aggregating final metrics..."
)

results = pd.DataFrame(
    evaluation_records
)

summary = (
    results
    .groupby(
        ["method", "k"],
        as_index=False
    )[
        [
            "precision",
            "recall",
            "ndcg"
        ]
    ]
    .mean()
)

# Stable presentation order.
method_order = [
    "Popularity",
    "Content-Based",
    "KNN Collaborative",
    "Hybrid"
]

summary["method"] = pd.Categorical(
    summary["method"],
    categories=method_order,
    ordered=True
)

summary = summary.sort_values(
    [
        "k",
        "method"
    ]
)

summary["method"] = (
    summary["method"]
    .astype(str)
)


# ============================================================
# SAVE RESULTS
# ============================================================

summary.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 72)
print("FINAL EVALUATION RESULTS")
print("=" * 72)

for k in TOP_K_VALUES:

    print(
        f"\n@{k}"
    )

    subset = summary[
        summary["k"] == k
    ]

    print(
        f"{'Method':<24}"
        f"{'Precision':>14}"
        f"{'Recall':>14}"
        f"{'NDCG':>14}"
    )

    print(
        "-" * 66
    )

    for _, result in subset.iterrows():

        print(
            f"{result['method']:<24}"
            f"{result['precision']:>14.6f}"
            f"{result['recall']:>14.6f}"
            f"{result['ndcg']:>14.6f}"
        )


print("\n")
print("=" * 72)
print("Evaluation completed successfully.")
print("=" * 72)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nNOTE:"
)

print(
    "This evaluation uses synthetic interaction data "
    "as a controlled experimental dataset. "
    "The results should not be interpreted as "
    "real-world validation or recommendation accuracy."
)