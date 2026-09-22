from pathlib import Path
import joblib
import numpy as np

ROOT = Path(r"D:\AI-Personalized-Learning-Recommendation")
MODEL_DIR = ROOT / "models"

EXPECTED_FILES = [
    "course_to_index.joblib",
    "hybrid_config.joblib",
    "index_to_course.joblib",
    "index_to_user.joblib",
    "knn_collaborative_model.joblib",
    "tfidf_matrix.npz",
    "tfidf_vectorizer.joblib",
    "user_course_matrix.npz",
    "user_to_index.joblib",
]


def check_file(filename):
    path = MODEL_DIR / filename

    if not path.exists():
        return False, "MISSING"

    size_bytes = path.stat().st_size

    if size_bytes <= 0:
        return False, "EMPTY"

    return True, f"{size_bytes / 1024:.2f} KB"

print("=" * 70)
print("STEP 21 — ML MODEL ARTIFACT VERIFICATION")
print("=" * 70)

print(f"\nProject root:")
print(ROOT)

print(f"\nModel directory:")
print(MODEL_DIR)

print("\nChecking required files...\n")

all_ok = True

for filename in EXPECTED_FILES:
    ok, info = check_file(filename)

    status = "OK" if ok else "FAILED"

    print(f"[{status:<6}] {filename:<40} {info}")

    if not ok:
        all_ok = False


print("\n" + "=" * 70)

if not all_ok:
    print("RESULT: Some required artifacts are missing.")
    print("Do NOT continue to FastAPI until the missing artifacts are rebuilt.")
else:
    print("RESULT: All required ML artifacts are present.")

    print("\nTesting artifact loading...")

    try:
        course_to_index = joblib.load(
            MODEL_DIR / "course_to_index.joblib"
        )

        hybrid_config = joblib.load(
            MODEL_DIR / "hybrid_config.joblib"
        )

        index_to_course = joblib.load(
            MODEL_DIR / "index_to_course.joblib"
        )

        index_to_user = joblib.load(
            MODEL_DIR / "index_to_user.joblib"
        )

        knn_model = joblib.load(
            MODEL_DIR / "knn_collaborative_model.joblib"
        )

        tfidf_vectorizer = joblib.load(
            MODEL_DIR / "tfidf_vectorizer.joblib"
        )

        user_to_index = joblib.load(
            MODEL_DIR / "user_to_index.joblib"
        )

        tfidf_matrix = np.load(
            MODEL_DIR / "tfidf_matrix.npz"
        )

        user_course_matrix = np.load(
            MODEL_DIR / "user_course_matrix.npz"
        )

        print("\nAll artifacts loaded successfully.")

        print("\nArtifact summary:")
        print(f"Courses mapped       : {len(course_to_index):,}")
        print(f"Users mapped         : {len(user_to_index):,}")
        print(f"Index courses        : {len(index_to_course):,}")
        print(f"Index users          : {len(index_to_user):,}")

        print(
            f"TF-IDF matrix files  : "
            f"{tfidf_matrix.files}"
        )

        print(
            f"User-course matrix   : "
            f"{user_course_matrix.files}"
        )

        print(f"Hybrid configuration : {hybrid_config}")

        print("\nKNN model:")
        print(knn_model)

        print("\nTF-IDF vectorizer:")
        print(tfidf_vectorizer)

        print("\n" + "=" * 70)
        print("STEP 21 COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        print("\nArtifact loading FAILED.")
        print(f"Error: {e}")
        raise