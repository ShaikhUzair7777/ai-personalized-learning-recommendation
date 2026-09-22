"""
STEP 17
Skill-Gap + Learning-Goal Personalization

Purpose:
- Add learning goals and target skills.
- Identify skills represented in a learner's goal.
- Compare target skills against course skills/content.
- Calculate a Skill-Gap Score.
- Produce a more explainable personalized ranking.

This is a prototype skill-gap layer.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED = ROOT / "data" / "processed"

COURSES_FILE = (
    DATA_PROCESSED / "courses_ready.csv"
)

LEARNERS_FILE = (
    DATA_PROCESSED / "learner_features_ready.csv"
)

OUTPUT_FILE = (
    DATA_PROCESSED
    / "skill_gap_demo_results.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 17 - SKILL-GAP PERSONALIZATION")
print("=" * 70)

print("\nLoading course and learner data...")

courses = pd.read_csv(
    COURSES_FILE
)

learners = pd.read_csv(
    LEARNERS_FILE
)

print(
    f"Courses loaded: {len(courses):,}"
)

print(
    f"Learners loaded: {len(learners):,}"
)


# ============================================================
# 3. CLEAN COURSE TEXT
# ============================================================

text_columns = [
    "course_name",
    "course_description",
    "skills"
]

for column in text_columns:

    courses[column] = (
        courses[column]
        .fillna("")
        .astype(str)
    )


courses["course_text"] = (
    courses["course_name"]
    + " "
    + courses["course_description"]
    + " "
    + courses["skills"]
)


# ============================================================
# 4. DEMONSTRATION LEARNER GOALS
# ============================================================

demo_profiles = [

    {
        "learner_id": 23698,
        "goal":
            "Become a Data Scientist and improve "
            "Machine Learning and Data Analysis skills",

        "known_skills":
            "Python SQL Pandas NumPy Data Analysis",

        "target_skills":
            "Machine Learning Statistics "
            "Data Visualization Deep Learning"
    },

    {
        "learner_id": 590762,
        "goal":
            "Become a Full Stack Web Developer",

        "known_skills":
            "HTML CSS JavaScript",

        "target_skills":
            "React Node.js Express MongoDB "
            "Backend Development APIs"
    },

    {
        "learner_id": 2716795,
        "goal":
            "Learn Blockchain Development and "
            "build decentralized applications",

        "known_skills":
            "Programming Distributed Systems",

        "target_skills":
            "Blockchain Ethereum Solidity "
            "Smart Contracts Web3"
    }
]


# ============================================================
# 5. TF-IDF FOR COURSE SKILLS
# ============================================================

print("\nBuilding skill representation...")

skill_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1
)

course_matrix = (
    skill_vectorizer.fit_transform(
        courses["course_text"]
    )
)

print(
    f"Course feature matrix: "
    f"{course_matrix.shape}"
)


# ============================================================
# 6. NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9+#.\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 7. SKILL-GAP SCORE
# ============================================================

def calculate_skill_gap_score(
    target_skills
):

    target_text = normalize_text(
        target_skills
    )

    target_vector = (
        skill_vectorizer.transform(
            [target_text]
        )
    )

    scores = cosine_similarity(
        target_vector,
        course_matrix
    )[0]

    return np.clip(
        scores,
        0,
        1
    )


# ============================================================
# 8. KNOWN-SKILL MATCH
# ============================================================

def calculate_known_skill_match(
    known_skills
):

    known_text = normalize_text(
        known_skills
    )

    known_vector = (
        skill_vectorizer.transform(
            [known_text]
        )
    )

    scores = cosine_similarity(
        known_vector,
        course_matrix
    )[0]

    return np.clip(
        scores,
        0,
        1
    )


# ============================================================
# 9. GENERATE DEMO RESULTS
# ============================================================

all_results = []

for profile in demo_profiles:

    learner_id = profile[
        "learner_id"
    ]

    print("\n")
    print("-" * 70)

    print(
        f"Learner ID: {learner_id}"
    )

    print(
        f"Goal: {profile['goal']}"
    )

    print(
        f"Known skills: "
        f"{profile['known_skills']}"
    )

    print(
        f"Target skills: "
        f"{profile['target_skills']}"
    )

    # --------------------------------------------------------
    # SKILL GAP
    # --------------------------------------------------------

    skill_gap_scores = (
        calculate_skill_gap_score(
            profile["target_skills"]
        )
    )

    # --------------------------------------------------------
    # KNOWN SKILLS
    # --------------------------------------------------------

    known_skill_scores = (
        calculate_known_skill_match(
            profile["known_skills"]
        )
    )

    # --------------------------------------------------------
    # COMBINED PERSONALIZATION
    # --------------------------------------------------------

    # Higher target-skill match gets greater importance.
    personalized_score = (
        0.70
        * skill_gap_scores
        +
        0.30
        * known_skill_scores
    )

    # --------------------------------------------------------
    # TOP COURSES
    # --------------------------------------------------------

    top_indices = np.argsort(
        -personalized_score
    )[:10]

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        course = courses.iloc[
            index
        ]

        all_results.append({

            "learner_id":
                learner_id,

            "goal":
                profile["goal"],

            "known_skills":
                profile["known_skills"],

            "target_skills":
                profile["target_skills"],

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

            "course_rating":
                course["course_rating"],

            "skill_gap_score":
                skill_gap_scores[index],

            "known_skill_match":
                known_skill_scores[index],

            "personalized_skill_score":
                personalized_score[index]
        })

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    display = pd.DataFrame(
        all_results
    )

    display = display[
        display["learner_id"]
        == learner_id
    ]

    print("\nTop skill-gap recommendations:")

    print(
        display[
            [
                "rank",
                "course_id",
                "course_name",
                "difficulty",
                "skill_gap_score",
                "known_skill_match",
                "personalized_skill_score"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 10. SAVE
# ============================================================

results = pd.DataFrame(
    all_results
)

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 11. SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("STEP 17 COMPLETE")
print("=" * 70)

print(
    f"\nGenerated recommendations: "
    f"{len(results):,}"
)

print(
    f"Output saved to:\n{OUTPUT_FILE}"
)

print(
    "\nThe Skill-Gap personalization layer "
    "is now working."
)