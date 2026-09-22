import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from joblib import load
from scipy.sparse import load_npz


# ============================================================
# AI PERSONALIZED LEARNING RECOMMENDATION SYSTEM
# Content-Based Recommendation Engine
# ============================================================

DATA_PATH = Path("data/processed/coursera_clean.csv")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.joblib")
MATRIX_PATH = Path("models/tfidf_matrix.npz")


# ------------------------------------------------------------
# Load model and data
# ------------------------------------------------------------

print("=" * 70)
print("AI PERSONALIZED LEARNING RECOMMENDATION SYSTEM")
print("COURSE RECOMMENDATION ENGINE")
print("=" * 70)

print("\nLoading course dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Courses loaded: {len(df):,}")

print("\nLoading TF-IDF vectorizer...")
vectorizer = load(VECTORIZER_PATH)

print("Loading TF-IDF matrix...")
tfidf_matrix = load_npz(MATRIX_PATH)

print("Model loaded successfully!")


# ------------------------------------------------------------
# Recommendation function
# ------------------------------------------------------------

def recommend_courses(user_profile, top_k=10):
    """
    Recommend courses based on a learner's interests,
    skills and learning goals.
    """

    print("\n" + "=" * 70)
    print("GENERATING RECOMMENDATIONS")
    print("=" * 70)

    print(f"\nLearner profile:")
    print(user_profile)

    # Convert learner profile into TF-IDF vector
    user_vector = vectorizer.transform([user_profile])

    # Calculate similarity against all courses
    similarity_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()

    # Get top course indices
    top_indices = similarity_scores.argsort()[-top_k:][::-1]

    recommendations = df.iloc[top_indices].copy()

    recommendations["similarity_score"] = (
        similarity_scores[top_indices]
    )

    # Select useful columns
    recommendations = recommendations[
        [
            "course_name",
            "university",
            "difficulty_level",
            "course_rating",
            "course_url",
            "similarity_score"
        ]
    ]

    return recommendations.reset_index(drop=True)


# ============================================================
# TEST 1 — Data Science learner
# ============================================================

profile_1 = """
Python Data Science Machine Learning
Data Analysis SQL Statistics
Data Visualization Pandas NumPy
"""

results_1 = recommend_courses(
    profile_1,
    top_k=10
)

print("\n" + "=" * 70)
print("RECOMMENDATIONS FOR DATA SCIENCE LEARNER")
print("=" * 70)

for i, row in results_1.iterrows():

    print(
        f"\n{i + 1}. {row['course_name']}"
        f"\n   University : {row['university']}"
        f"\n   Difficulty : {row['difficulty_level']}"
        f"\n   Rating     : {row['course_rating']}"
        f"\n   Similarity : {row['similarity_score']:.4f}"
        f"\n   URL        : {row['course_url']}"
    )


# ============================================================
# TEST 2 — Web Development learner
# ============================================================

profile_2 = """
Web Development HTML CSS JavaScript
React Node.js Express MongoDB
Frontend Backend Full Stack
"""

results_2 = recommend_courses(
    profile_2,
    top_k=10
)

print("\n" + "=" * 70)
print("RECOMMENDATIONS FOR WEB DEVELOPMENT LEARNER")
print("=" * 70)

for i, row in results_2.iterrows():

    print(
        f"\n{i + 1}. {row['course_name']}"
        f"\n   University : {row['university']}"
        f"\n   Difficulty : {row['difficulty_level']}"
        f"\n   Rating     : {row['course_rating']}"
        f"\n   Similarity : {row['similarity_score']:.4f}"
    )


# ============================================================
# TEST 3 — Blockchain learner
# ============================================================

profile_3 = """
Blockchain cryptocurrency Ethereum
Smart Contracts Solidity Web3
Decentralized Applications distributed systems
"""

results_3 = recommend_courses(
    profile_3,
    top_k=10
)

print("\n" + "=" * 70)
print("RECOMMENDATIONS FOR BLOCKCHAIN LEARNER")
print("=" * 70)

for i, row in results_3.iterrows():

    print(
        f"\n{i + 1}. {row['course_name']}"
        f"\n   University : {row['university']}"
        f"\n   Difficulty : {row['difficulty_level']}"
        f"\n   Rating     : {row['course_rating']}"
        f"\n   Similarity : {row['similarity_score']:.4f}"
    )


print("\n" + "=" * 70)
print("RECOMMENDATION ENGINE TEST COMPLETE")
print("=" * 70)