"""
STEP 12 - EFFICIENT KNN EVALUATION

Evaluates KNN Collaborative Filtering against
a Popularity Baseline.

Metrics:
    Precision@K
    Recall@K
    NDCG@K

IMPORTANT:
The interaction data is SYNTHETIC and is used only
for controlled recommender-system experimentation.
"""

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/synthetic_interactions.csv"

RANDOM_STATE = 42

K_NEIGHBORS = 10

EVALUATION_K = [5, 10]


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("EFFICIENT KNN COLLABORATIVE FILTERING EVALUATION")
print("=" * 80)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading interaction data...")

df = pd.read_csv(INPUT_FILE)

print(f"Total interactions: {len(df):,}")
print(f"Unique learners: {df['user_id'].nunique():,}")
print(f"Unique courses: {df['course_id'].nunique():,}")


# ============================================================
# 2. CONVERT INTERACTIONS TO STRENGTH
# ============================================================

interaction_weights = {
    "clicked": 0.25,
    "viewed": 0.50,
    "started": 0.75,
    "completed": 1.00
}

df["interaction_weight"] = (
    df["interaction_type"].map(
        interaction_weights
    )
)

df["interaction_strength"] = (
    df["interaction_weight"]
    * df["preference_score"]
)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

print("\nCreating per-user train/test split...")

rng = np.random.default_rng(
    RANDOM_STATE
)

train_parts = []
test_parts = []

for user_id, user_data in df.groupby(
    "user_id"
):

    user_data = user_data.copy()

    # Hold out exactly one interaction
    # from every learner.
    test_index = rng.choice(
        user_data.index,
        size=1,
        replace=False
    )

    test_parts.append(
        user_data.loc[test_index]
    )

    train_parts.append(
        user_data.drop(test_index)
    )


train_df = pd.concat(
    train_parts,
    ignore_index=True
)

test_df = pd.concat(
    test_parts,
    ignore_index=True
)

print(
    f"Training interactions: "
    f"{len(train_df):,}"
)

print(
    f"Testing interactions: "
    f"{len(test_df):,}"
)

print(
    f"Test learners: "
    f"{test_df['user_id'].nunique():,}"
)


# ============================================================
# 4. CREATE USER / COURSE INDEXES
# ============================================================

print("\nCreating indexes...")

users = sorted(
    df["user_id"].unique()
)

courses = sorted(
    df["course_id"].unique()
)

user_to_index = {
    user_id: i
    for i, user_id in enumerate(users)
}

course_to_index = {
    course_id: i
    for i, course_id in enumerate(courses)
}

index_to_course = {
    i: course_id
    for course_id, i in course_to_index.items()
}


# ============================================================
# 5. BUILD TRAINING MATRIX
# ============================================================

print("\nBuilding training user-course matrix...")

rows = train_df["user_id"].map(
    user_to_index
)

cols = train_df["course_id"].map(
    course_to_index
)

values = train_df[
    "interaction_strength"
]

train_matrix = csr_matrix(
    (
        values,
        (rows, cols)
    ),
    shape=(
        len(users),
        len(courses)
    )
)

print(
    f"Training matrix shape: "
    f"{train_matrix.shape}"
)

print(
    f"Non-zero interactions: "
    f"{train_matrix.nnz:,}"
)


# ============================================================
# 6. TRAIN KNN
# ============================================================

print("\nTraining KNN model...")

knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=K_NEIGHBORS + 1,
    n_jobs=1
)

knn.fit(
    train_matrix
)

print("KNN model trained successfully.")


# ============================================================
# 7. FIND ALL USER NEIGHBORS ONCE
# ============================================================

print("\nFinding nearest learners for all users...")

distances, neighbor_indices = (
    knn.kneighbors(
        train_matrix,
        n_neighbors=K_NEIGHBORS + 1
    )
)

print("Nearest-neighbor search complete.")


# ============================================================
# 8. CREATE LOOKUP TABLES
# ============================================================

print("\nPreparing evaluation lookups...")

# Training courses for each user
user_seen_courses = {}

for user_id, group in train_df.groupby(
    "user_id"
):

    user_seen_courses[user_id] = set(
        group["course_id"]
    )


# Test course for each user
user_test_courses = {}

for user_id, group in test_df.groupby(
    "user_id"
):

    user_test_courses[user_id] = set(
        group["course_id"]
    )


# ============================================================
# 9. GENERATE KNN RECOMMENDATIONS
# ============================================================

def generate_knn_recommendations(
    user_index,
    top_k
):

    user_id = users[user_index]

    seen_courses = user_seen_courses[
        user_id
    ]

    course_scores = {}

    # Neighbor list includes the user themselves
    # as the first result, so skip it.
    for distance, neighbor_index in zip(
        distances[user_index],
        neighbor_indices[user_index]
    ):

        if neighbor_index == user_index:
            continue

        similarity = 1 - distance

        neighbor_row = train_matrix[
            neighbor_index
        ]

        for course_index, value in zip(
            neighbor_row.indices,
            neighbor_row.data
        ):

            course_id = index_to_course[
                course_index
            ]

            if course_id in seen_courses:
                continue

            score = (
                similarity * value
            )

            course_scores[course_id] = (
                course_scores.get(
                    course_id,
                    0.0
                )
                + score
            )

    ranked = sorted(
        course_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        course_id
        for course_id, score
        in ranked[:top_k]
    ]


# ============================================================
# 10. POPULARITY BASELINE
# ============================================================

print("\nPreparing popularity baseline...")

popularity = (
    train_df
    .groupby("course_id")
    ["interaction_strength"]
    .sum()
    .sort_values(
        ascending=False
    )
)

popular_courses = (
    popularity.index.tolist()
)


def generate_popularity_recommendations(
    user_id,
    top_k
):

    seen_courses = user_seen_courses[
        user_id
    ]

    recommendations = []

    for course_id in popular_courses:

        if course_id not in seen_courses:

            recommendations.append(
                course_id
            )

        if len(recommendations) >= top_k:
            break

    return recommendations


# ============================================================
# 11. METRIC FUNCTIONS
# ============================================================

def precision_at_k(
    recommendations,
    relevant,
    k
):

    recommendations = (
        recommendations[:k]
    )

    if not recommendations:
        return 0.0

    hits = sum(
        item in relevant
        for item in recommendations
    )

    return hits / len(recommendations)


def recall_at_k(
    recommendations,
    relevant,
    k
):

    if not relevant:
        return 0.0

    recommendations = (
        recommendations[:k]
    )

    hits = sum(
        item in relevant
        for item in recommendations
    )

    return hits / len(relevant)


def ndcg_at_k(
    recommendations,
    relevant,
    k
):

    recommendations = (
        recommendations[:k]
    )

    dcg = 0.0

    for rank, item in enumerate(
        recommendations,
        start=1
    ):

        if item in relevant:

            dcg += (
                1.0
                / np.log2(rank + 1)
            )

    ideal_hits = min(
        len(relevant),
        k
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0
        / np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    return dcg / idcg


# ============================================================
# 12. EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("RUNNING EVALUATION")
print("=" * 80)

results = []

test_users = list(
    user_test_courses.keys()
)

total_users = len(test_users)

for model_name in [
    "Popularity",
    "KNN"
]:

    print(
        f"\nEvaluating {model_name}..."
    )

    for k in EVALUATION_K:

        precision_scores = []
        recall_scores = []
        ndcg_scores = []

        for counter, user_id in enumerate(
            test_users,
            start=1
        ):

            user_index = user_to_index[
                user_id
            ]

            relevant = user_test_courses[
                user_id
            ]

            if model_name == "KNN":

                recommendations = (
                    generate_knn_recommendations(
                        user_index,
                        k
                    )
                )

            else:

                recommendations = (
                    generate_popularity_recommendations(
                        user_id,
                        k
                    )
                )

            precision_scores.append(
                precision_at_k(
                    recommendations,
                    relevant,
                    k
                )
            )

            recall_scores.append(
                recall_at_k(
                    recommendations,
                    relevant,
                    k
                )
            )

            ndcg_scores.append(
                ndcg_at_k(
                    recommendations,
                    relevant,
                    k
                )
            )

            # Progress every 1,000 users.
            if counter % 1000 == 0:

                print(
                    f"  Processed "
                    f"{counter:,}/"
                    f"{total_users:,}"
                    f" users"
                )

        results.append(
            {
                "model": model_name,
                "k": k,
                "precision": np.mean(
                    precision_scores
                ),
                "recall": np.mean(
                    recall_scores
                ),
                "ndcg": np.mean(
                    ndcg_scores
                )
            }
        )


# ============================================================
# 13. RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 80)
print("EVALUATION RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

OUTPUT_FILE = (
    "data/processed/"
    "knn_evaluation_results.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nResults saved to:"
    f"\n{OUTPUT_FILE}"
)


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 80)
print("STEP 12 COMPLETE")
print("=" * 80)

print(
    "\nModels evaluated:"
)

print(
    "1. Popularity Baseline"
)

print(
    "2. KNN Collaborative Filtering"
)

print(
    "\nMetrics:"
)

print(
    "Precision@5 / Precision@10"
)

print(
    "Recall@5 / Recall@10"
)

print(
    "NDCG@5 / NDCG@10"
)

print(
    "\nIMPORTANT:"
)

print(
    "These are controlled experiments "
    "using SYNTHETIC interactions."
)

print(
    "They should not be presented as "
    "real-world student behavior."
)

print("\n" + "=" * 80)