"""
STEP 16
Final Recommendation Engine Validation

Purpose:
- Validate the practical recommendation behavior.
- Generate personalized recommendations for sample learners.
- Show individual scoring components.

Models:
1. TF-IDF Content-Based
2. KNN Collaborative
3. Performance-Difficulty Personalization
4. Hybrid Recommendation

IMPORTANT:
The collaborative interaction dataset is synthetic.
Therefore these recommendations are a controlled
prototype demonstration and not evidence of real-world
student behavior.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from joblib import load


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

COURSES_FILE = (
    DATA_PROCESSED / "courses_ready.csv"
)

LEARNERS_FILE = (
    DATA_PROCESSED / "learner_features_ready.csv"
)

INTERACTIONS_FILE = (
    DATA_PROCESSED / "synthetic_interactions.csv"
)

TFIDF_MATRIX_FILE = (
    MODELS / "tfidf_matrix.npz"
)

USER_COURSE_MATRIX_FILE = (
    MODELS / "user_course_matrix.npz"
)

USER_TO_INDEX_FILE = (
    MODELS / "user_to_index.joblib"
)

COURSE_TO_INDEX_FILE = (
    MODELS / "course_to_index.joblib"
)


OUTPUT_FILE = (
    DATA_PROCESSED
    / "final_demo_recommendations.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TOP_K = 10

KNN_NEIGHBORS = 10

INTERACTION_WEIGHTS = {
    "clicked": 0.25,
    "viewed": 0.50,
    "started": 0.75,
    "completed": 1.00
}


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 16 - FINAL RECOMMENDATION ENGINE VALIDATION")
print("=" * 70)

print("\nLoading datasets and models...")

courses = pd.read_csv(
    COURSES_FILE
)

learners = pd.read_csv(
    LEARNERS_FILE
)

interactions = pd.read_csv(
    INTERACTIONS_FILE
)

tfidf_matrix = load_npz(
    TFIDF_MATRIX_FILE
).tocsr()

user_course_matrix = load_npz(
    USER_COURSE_MATRIX_FILE
).tocsr()

user_to_index = load(
    USER_TO_INDEX_FILE
)

course_to_index = load(
    COURSE_TO_INDEX_FILE
)

print(
    f"Courses: {len(courses):,}"
)

print(
    f"Learners: {len(learners):,}"
)

print(
    f"Interactions: {len(interactions):,}"
)

print(
    f"TF-IDF matrix: {tfidf_matrix.shape}"
)

print(
    f"KNN matrix: {user_course_matrix.shape}"
)


# ============================================================
# 4. COURSE MAPPINGS
# ============================================================

course_ids = (
    courses["course_id"]
    .astype(str)
    .tolist()
)

course_id_to_catalog_index = {
    course_id: index
    for index, course_id
    in enumerate(course_ids)
}

catalog_index_to_course_id = {
    index: course_id
    for index, course_id
    in enumerate(course_ids)
}


# ============================================================
# 5. LEARNER MAPPINGS
# ============================================================

learners["id_student"] = (
    learners["id_student"]
    .astype(int)
)

learner_lookup = (
    learners
    .set_index("id_student")
)


# ============================================================
# 6. INTERACTION STRENGTH
# ============================================================

interactions["user_id"] = (
    interactions["user_id"]
    .astype(int)
)

interactions["course_id"] = (
    interactions["course_id"]
    .astype(str)
)

interactions["interaction_value"] = (
    interactions["interaction_value"]
    .astype(float)
)

interactions["preference_score"] = (
    interactions["preference_score"]
    .astype(float)
)

interactions["interaction_strength"] = (
    interactions["interaction_value"]
    *
    interactions["preference_score"]
)


# ============================================================
# 7. COURSE DIFFICULTY
# ============================================================

courses["difficulty_clean"] = (
    courses["difficulty_level"]
    .astype(str)
    .str.strip()
    .str.lower()
)

difficulty_values = {
    "beginner": 0.25,
    "intermediate": 0.55,
    "advanced": 0.85
}


# ============================================================
# 8. TRAIN KNN
# ============================================================

print("\nTraining KNN model for demo...")

knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=KNN_NEIGHBORS + 1,
    n_jobs=1
)

knn.fit(
    user_course_matrix
)

print("KNN ready.")


# ============================================================
# 9. CONTENT PROFILE FUNCTION
# ============================================================

def get_content_profile(
    user_id
):
    """
    Creates a learner's content profile from
    their interacted courses.

    The profile is a weighted average of course
    TF-IDF vectors.
    """

    user_interactions = (
        interactions[
            interactions["user_id"]
            == user_id
        ]
    )

    if user_interactions.empty:

        return None

    vectors = []
    weights = []

    for row in user_interactions.itertuples(
        index=False
    ):

        course_id = str(
            row.course_id
        )

        if course_id not in (
            course_id_to_catalog_index
        ):
            continue

        course_index = (
            course_id_to_catalog_index[
                course_id
            ]
        )

        vectors.append(
            tfidf_matrix[
                course_index
            ]
        )

        weights.append(
            row.interaction_strength
        )

    if not vectors:

        return None

    profile = vectors[0] * weights[0]

    for vector, weight in zip(
        vectors[1:],
        weights[1:]
    ):

        profile = (
            profile
            +
            vector * weight
        )

    total_weight = sum(
        weights
    )

    if total_weight > 0:

        profile = (
            profile
            / total_weight
        )

    return profile


# ============================================================
# 10. COLLABORATIVE SCORE FUNCTION
# ============================================================

def get_collaborative_scores(
    user_id
):
    """
    Generates collaborative scores for
    the complete 3,424-course catalog.
    """

    scores = np.zeros(
        len(courses),
        dtype=float
    )

    if user_id not in user_to_index:

        return scores

    user_index = (
        user_to_index[
            user_id
        ]
    )

    distances, neighbors = (
        knn.kneighbors(
            user_course_matrix[
                user_index
            ],
            n_neighbors=KNN_NEIGHBORS + 1
        )
    )

    similarities = (
        1.0 - distances[0]
    )

    neighbor_indices = (
        neighbors[0]
    )

    for neighbor_index, similarity in zip(
        neighbor_indices,
        similarities
    ):

        if neighbor_index == user_index:
            continue

        neighbor_row = (
            user_course_matrix[
                neighbor_index
            ]
        )

        course_indices = (
            neighbor_row.indices
        )

        course_values = (
            neighbor_row.data
        )

        for course_index, value in zip(
            course_indices,
            course_values
        ):

            course_id = (
                catalog_index_to_course_id.get(
                    course_index
                )
            )

            if course_id is None:
                continue

            catalog_index = (
                course_id_to_catalog_index[
                    course_id
                ]
            )

            scores[
                catalog_index
            ] += (
                similarity
                * value
            )

    maximum = scores.max()

    if maximum > 0:

        scores = (
            scores
            / maximum
        )

    return scores


# ============================================================
# 11. PERFORMANCE / DIFFICULTY SCORE
# ============================================================

def get_performance_difficulty_scores(
    user_id
):

    scores = np.zeros(
        len(courses),
        dtype=float
    )

    if user_id not in learner_lookup.index:

        return scores

    average_score = learner_lookup.loc[
        user_id,
        "average_score"
    ]

    if pd.isna(average_score):

        average_score = 0.0

    performance = (
        float(average_score)
        / 100.0
    )

    for index, difficulty in enumerate(
        courses["difficulty_clean"]
    ):

        difficulty_value = (
            difficulty_values.get(
                difficulty,
                0.55
            )
        )

        fit = (
            1.0
            -
            abs(
                performance
                -
                difficulty_value
            )
        )

        scores[index] = np.clip(
            fit,
            0,
            1
        )

    return scores


# ============================================================
# 12. HYBRID RECOMMENDER
# ============================================================

def recommend_for_user(
    user_id,
    top_k=10
):

    print("\n" + "-" * 70)
    print(
        f"Generating recommendations "
        f"for learner {user_id}"
    )
    print("-" * 70)

    if user_id not in learner_lookup.index:

        print(
            "Learner profile not found."
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # LEARNER INFORMATION
    # --------------------------------------------------------

    learner = learner_lookup.loc[
        user_id
    ]

    average_score = learner[
        "average_score"
    ]

    engagement_score = learner[
        "engagement_score"
    ]

    performance_level = learner[
        "performance_level"
    ]

    engagement_level = learner[
        "engagement_level"
    ]

    print(
        f"Average performance: "
        f"{average_score:.2f}"
    )

    print(
        f"Performance level: "
        f"{performance_level}"
    )

    print(
        f"Engagement score: "
        f"{engagement_score:.2f}"
    )

    print(
        f"Engagement level: "
        f"{engagement_level}"
    )

    # --------------------------------------------------------
    # CONTENT SCORE
    # --------------------------------------------------------

    profile = (
        get_content_profile(
            user_id
        )
    )

    if profile is None:

        content_scores = np.zeros(
            len(courses)
        )

    else:

        content_scores = (
            cosine_similarity(
                profile,
                tfidf_matrix
            )[0]
        )

        content_scores = np.clip(
            content_scores,
            0,
            1
        )

    # --------------------------------------------------------
    # COLLABORATIVE SCORE
    # --------------------------------------------------------

    collaborative_scores = (
        get_collaborative_scores(
            user_id
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
        0.40
        * content_scores
        +
        0.30
        * collaborative_scores
        +
        0.30
        * performance_difficulty_scores
    )

    # --------------------------------------------------------
    # REMOVE PREVIOUSLY INTERACTED COURSES
    # --------------------------------------------------------

    previous_courses = set(
        interactions.loc[
            interactions["user_id"]
            == user_id,
            "course_id"
        ]
        .astype(str)
        .tolist()
    )

    for course_id in previous_courses:

        if course_id in (
            course_id_to_catalog_index
        ):

            index = (
                course_id_to_catalog_index[
                    course_id
                ]
            )

            hybrid_scores[index] = -np.inf

    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    valid_count = np.isfinite(
        hybrid_scores
    ).sum()

    actual_k = min(
        top_k,
        valid_count
    )

    top_indices = np.argpartition(
        -hybrid_scores,
        actual_k - 1
    )[:actual_k]

    top_indices = top_indices[
        np.argsort(
            -hybrid_scores[
                top_indices
            ]
        )
    ]

    recommendations = []

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        course = courses.iloc[
            index
        ]

        recommendations.append({

            "user_id":
                user_id,

            "rank":
                rank,

            "course_id":
                course["course_id"],

            "course_name":
                course["course_name"],

            "university":
                course["university"],

            "difficulty":
                course["difficulty_level"],

            "rating":
                course["course_rating"],

            "content_score":
                content_scores[index],

            "collaborative_score":
                collaborative_scores[index],

            "performance_difficulty_score":
                performance_difficulty_scores[
                    index
                ],

            "hybrid_score":
                hybrid_scores[index]
        })

    result = pd.DataFrame(
        recommendations
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nTop recommendations:")

    display_columns = [
        "rank",
        "course_id",
        "course_name",
        "difficulty",
        "content_score",
        "collaborative_score",
        "performance_difficulty_score",
        "hybrid_score"
    ]

    print(
        result[
            display_columns
        ].to_string(
            index=False
        )
    )

    return result


# ============================================================
# 13. SELECT TEST LEARNERS
# ============================================================

print("\n")
print("=" * 70)
print("SELECTING DEMONSTRATION LEARNERS")
print("=" * 70)

# Use learners that actually exist in
# the collaborative dataset.

available_demo_users = [
    int(user_id)
    for user_id in user_to_index.keys()
    if int(user_id)
    in learner_lookup.index
]

available_demo_users = sorted(
    available_demo_users
)

if len(available_demo_users) < 3:

    raise ValueError(
        "Fewer than 3 valid demonstration "
        "learners are available."
    )


# Choose deterministic users so results
# remain reproducible.

demo_users = [
    available_demo_users[0],
    available_demo_users[
        len(available_demo_users) // 2
    ],
    available_demo_users[-1]
]

print(
    "Selected demonstration learners:"
)

for user_id in demo_users:

    learner = learner_lookup.loc[
        user_id
    ]

    print(
        f"  {user_id} | "
        f"Performance: "
        f"{learner['average_score']:.2f} | "
        f"Level: "
        f"{learner['performance_level']} | "
        f"Engagement: "
        f"{learner['engagement_level']}"
    )


# ============================================================
# 14. GENERATE RECOMMENDATIONS
# ============================================================

all_results = []

for user_id in demo_users:

    result = recommend_for_user(
        user_id,
        top_k=TOP_K
    )

    if not result.empty:

        all_results.append(
            result
        )


# ============================================================
# 15. SAVE RESULTS
# ============================================================

if all_results:

    final_results = pd.concat(
        all_results,
        ignore_index=True
    )

    final_results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("STEP 16 COMPLETE")
    print("=" * 70)

    print(
        "\nFinal recommendation results saved to:"
    )

    print(
        OUTPUT_FILE
    )

else:

    print(
        "\nNo recommendation results generated."
    )