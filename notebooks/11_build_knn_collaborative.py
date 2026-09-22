"""
STEP 11 - KNN COLLABORATIVE FILTERING

Builds a user-based KNN collaborative filtering model
using the synthetic learner-course interaction dataset.

IMPORTANT:
The interaction data is synthetic and is used only for
controlled recommender-system experimentation.
"""

import os
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/synthetic_interactions.csv"
OUTPUT_DIR = "models"

K_NEIGHBORS = 10


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("KNN COLLABORATIVE FILTERING")
print("=" * 80)


# ============================================================
# 1. LOAD INTERACTION DATA
# ============================================================

print("\nLoading synthetic interaction data...")

interactions = pd.read_csv(INPUT_FILE)

print(f"Total interactions: {len(interactions):,}")
print(f"Unique learners: {interactions['user_id'].nunique():,}")
print(f"Unique courses: {interactions['course_id'].nunique():,}")


# ============================================================
# 2. CONVERT INTERACTIONS TO NUMERIC VALUES
# ============================================================

print("\nConverting interaction types to preference values...")

interaction_weights = {
    "clicked": 0.25,
    "viewed": 0.50,
    "started": 0.75,
    "completed": 1.00
}

interactions["interaction_weight"] = (
    interactions["interaction_type"]
    .map(interaction_weights)
)

# Combine the interaction weight with the generated
# preference score.
interactions["interaction_strength"] = (
    interactions["interaction_weight"]
    * interactions["preference_score"]
)

print("\nInteraction strength statistics:")
print(
    interactions["interaction_strength"]
    .describe()
)


# ============================================================
# 3. CREATE USER AND COURSE INDEXES
# ============================================================

print("\nCreating learner and course indexes...")

users = sorted(interactions["user_id"].unique())
courses = sorted(interactions["course_id"].unique())

user_to_index = {
    user_id: index
    for index, user_id in enumerate(users)
}

course_to_index = {
    course_id: index
    for index, course_id in enumerate(courses)
}

index_to_user = {
    index: user_id
    for user_id, index in user_to_index.items()
}

index_to_course = {
    index: course_id
    for course_id, index in course_to_index.items()
}


# ============================================================
# 4. CREATE USER-COURSE MATRIX
# ============================================================

print("\nBuilding user-course interaction matrix...")

rows = interactions["user_id"].map(user_to_index)
cols = interactions["course_id"].map(course_to_index)
values = interactions["interaction_strength"]

user_course_matrix = csr_matrix(
    (
        values,
        (rows, cols)
    ),
    shape=(len(users), len(courses))
)

print(f"Matrix shape: {user_course_matrix.shape}")
print(f"Non-zero interactions: {user_course_matrix.nnz:,}")

total_cells = (
    user_course_matrix.shape[0]
    * user_course_matrix.shape[1]
)

density = (
    user_course_matrix.nnz / total_cells
) * 100

print(f"Matrix density: {density:.4f}%")


# ============================================================
# 5. TRAIN KNN MODEL
# ============================================================

print("\nTraining KNN collaborative filtering model...")

knn_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=K_NEIGHBORS + 1,
    n_jobs=-1
)

knn_model.fit(user_course_matrix)

print("KNN model trained successfully.")

print(f"Number of neighbors: {K_NEIGHBORS}")
print("Distance metric: cosine")


# ============================================================
# 6. TEST KNN WITH SAMPLE LEARNER
# ============================================================

sample_user = users[0]

print("\n" + "=" * 80)
print("SAMPLE LEARNER TEST")
print("=" * 80)

print(f"\nSample learner: {sample_user}")

sample_index = user_to_index[sample_user]

sample_vector = user_course_matrix[
    sample_index
]

distances, neighbor_indices = knn_model.kneighbors(
    sample_vector,
    n_neighbors=K_NEIGHBORS + 1
)

print("\nNearest learners:")

for rank, (distance, neighbor_index) in enumerate(
    zip(distances[0], neighbor_indices[0]),
    start=1
):

    neighbor_user = index_to_user[neighbor_index]

    # Skip the learner themselves.
    if neighbor_user == sample_user:
        continue

    similarity = 1 - distance

    print(
        f"{rank}. "
        f"Learner: {neighbor_user} | "
        f"Similarity: {similarity:.4f}"
    )


# ============================================================
# 7. RECOMMEND COURSES USING NEIGHBORS
# ============================================================

def recommend_courses_knn(
    user_id,
    top_k=10,
    n_neighbors=10
):
    """
    Generate course recommendations for a learner
    using similar learners.
    """

    if user_id not in user_to_index:
        raise ValueError(
            f"Unknown user_id: {user_id}"
        )

    user_index = user_to_index[user_id]

    user_vector = user_course_matrix[
        user_index
    ]

    distances, neighbor_indices = knn_model.kneighbors(
        user_vector,
        n_neighbors=n_neighbors + 1
    )

    # Dictionary for accumulating course scores.
    course_scores = {}

    # Courses already interacted with by this learner.
    interacted_courses = set(
        interactions.loc[
            interactions["user_id"] == user_id,
            "course_id"
        ]
    )

    for distance, neighbor_index in zip(
        distances[0],
        neighbor_indices[0]
    ):

        neighbor_user = index_to_user[
            neighbor_index
        ]

        # Skip the target learner.
        if neighbor_user == user_id:
            continue

        similarity = 1 - distance

        neighbor_row = user_course_matrix[
            neighbor_index
        ]

        course_indices = neighbor_row.indices
        course_values = neighbor_row.data

        for course_index, value in zip(
            course_indices,
            course_values
        ):

            course_id = index_to_course[
                course_index
            ]

            # Don't recommend courses the learner
            # has already interacted with.
            if course_id in interacted_courses:
                continue

            score = similarity * value

            course_scores[course_id] = (
                course_scores.get(course_id, 0)
                + score
            )

    if not course_scores:
        return pd.DataFrame(
            columns=[
                "course_id",
                "knn_score"
            ]
        )

    recommendations = pd.DataFrame(
        list(course_scores.items()),
        columns=[
            "course_id",
            "knn_score"
        ]
    )

    recommendations = (
        recommendations
        .sort_values(
            "knn_score",
            ascending=False
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    return recommendations


# ============================================================
# 8. GENERATE SAMPLE RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 80)
print("SAMPLE COURSE RECOMMENDATIONS")
print("=" * 80)

recommendations = recommend_courses_knn(
    sample_user,
    top_k=10,
    n_neighbors=K_NEIGHBORS
)

print(
    f"\nRecommendations for learner "
    f"{sample_user}:"
)

print(
    recommendations.to_string(
        index=False
    )
)


# ============================================================
# 9. SAVE MODEL INFORMATION
# ============================================================

print("\nSaving KNN model artifacts...")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# Save matrix.
from scipy.sparse import save_npz

save_npz(
    os.path.join(
        OUTPUT_DIR,
        "user_course_matrix.npz"
    ),
    user_course_matrix
)

# Save user/course mappings.
import joblib

joblib.dump(
    user_to_index,
    os.path.join(
        OUTPUT_DIR,
        "user_to_index.joblib"
    )
)

joblib.dump(
    course_to_index,
    os.path.join(
        OUTPUT_DIR,
        "course_to_index.joblib"
    )
)

joblib.dump(
    index_to_user,
    os.path.join(
        OUTPUT_DIR,
        "index_to_user.joblib"
    )
)

joblib.dump(
    index_to_course,
    os.path.join(
        OUTPUT_DIR,
        "index_to_course.joblib"
    )
)

joblib.dump(
    knn_model,
    os.path.join(
        OUTPUT_DIR,
        "knn_collaborative_model.joblib"
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("STEP 11 KNN MODEL COMPLETE")
print("=" * 80)

print("\nSaved model files:")

print(
    "models/user_course_matrix.npz"
)

print(
    "models/user_to_index.joblib"
)

print(
    "models/course_to_index.joblib"
)

print(
    "models/index_to_user.joblib"
)

print(
    "models/index_to_course.joblib"
)

print(
    "models/knn_collaborative_model.joblib"
)

print("\nIMPORTANT:")
print(
    "The interaction dataset used here is SYNTHETIC."
)

print(
    "KNN similarity is NOT the same as recommendation accuracy."
)

print(
    "Formal evaluation will be performed in the next step "
    "using held-out interactions."
)

print("\n" + "=" * 80)