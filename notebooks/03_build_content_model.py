import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ============================================================
# AI PERSONALIZED LEARNING RECOMMENDATION SYSTEM
# Content-Based Recommendation Model
# TF-IDF + Cosine Similarity
# ============================================================

DATA_PATH = Path("data/processed/coursera_clean.csv")
MODEL_DIR = Path("models")

TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = MODEL_DIR / "tfidf_matrix.npz"

print("=" * 70)
print("CONTENT-BASED RECOMMENDATION MODEL")
print("TF-IDF + COSINE SIMILARITY")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load cleaned dataset
# ------------------------------------------------------------

print("\nLoading cleaned dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Courses loaded: {len(df):,}")

# ------------------------------------------------------------
# 2. Validate combined text
# ------------------------------------------------------------

if "combined_text" not in df.columns:
    raise ValueError("combined_text column not found!")

df["combined_text"] = df["combined_text"].fillna("")

print("Combined text validated.")

# ------------------------------------------------------------
# 3. TF-IDF Vectorization
# ------------------------------------------------------------

print("\nCreating TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

print("TF-IDF completed!")

print(f"Matrix shape: {tfidf_matrix.shape}")
print(f"Number of courses: {tfidf_matrix.shape[0]:,}")
print(f"Number of features: {tfidf_matrix.shape[1]:,}")

# ------------------------------------------------------------
# 4. Save vectorizer
# ------------------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    vectorizer,
    TFIDF_PATH
)

print(f"\nTF-IDF vectorizer saved to:")
print(TFIDF_PATH)

# ------------------------------------------------------------
# 5. Save sparse matrix
# ------------------------------------------------------------

from scipy.sparse import save_npz

save_npz(
    MATRIX_PATH,
    tfidf_matrix
)

print("TF-IDF matrix saved to:")
print(MATRIX_PATH)

# ------------------------------------------------------------
# 6. Test similarity
# ------------------------------------------------------------

print("\nTesting similarity model...")

test_course_index = 0

similarities = cosine_similarity(
    tfidf_matrix[test_course_index],
    tfidf_matrix
).flatten()

# Don't recommend the course itself
similarities[test_course_index] = -1

top_indices = similarities.argsort()[-10:][::-1]

print("\n" + "=" * 70)
print("TOP 10 SIMILAR COURSES")
print("=" * 70)

original_course = df.iloc[test_course_index]

print(
    f"\nReference Course:\n"
    f"{original_course['course_name']}"
)

print("\nRecommended similar courses:\n")

for rank, index in enumerate(top_indices, start=1):

    course = df.iloc[index]

    print(
        f"{rank:2}. "
        f"{course['course_name']} | "
        f"Similarity: {similarities[index]:.4f} | "
        f"Difficulty: {course['difficulty_level']} | "
        f"Rating: {course['course_rating']}"
    )

# ------------------------------------------------------------
# 7. Model statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

print(f"Courses               : {len(df):,}")
print(f"TF-IDF features       : {tfidf_matrix.shape[1]:,}")
print(f"TF-IDF matrix rows    : {tfidf_matrix.shape[0]:,}")
print(f"TF-IDF matrix columns : {tfidf_matrix.shape[1]:,}")
print(f"Matrix non-zero values: {tfidf_matrix.nnz:,}")

print("\n" + "=" * 70)
print("CONTENT MODEL COMPLETE")
print("=" * 70)