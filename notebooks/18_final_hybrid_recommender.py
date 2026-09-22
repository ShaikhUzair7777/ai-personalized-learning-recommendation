# ============================================================
# STEP 18D
# TWO-STAGE HYBRID PERSONALIZED LEARNING RECOMMENDER
# WITH SKILL-RELEVANCE GATE
# ============================================================
#
# Stage 1:
#   Candidate Generation
#   - Content-Based TF-IDF
#   - Collaborative Filtering KNN
#   - Skill-Gap TF-IDF
#
# Stage 2:
#   Personalized Ranking
#   - Skill Gap
#   - Content Similarity
#   - Collaborative Signal
#   - Performance/Difficulty
#
# Additional Step:
#   Skill-Relevance Gate
#
# Purpose:
#   Prevent extremely weak skill matches from dominating
#   recommendations because of high content/collaborative
#   scores.
#
# IMPORTANT:
#   The demonstration learner goals and target skills are
#   scenario-based inputs. They are NOT claimed to be
#   fields directly available in OULAD.
#
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd

from joblib import load
from scipy.sparse import load_npz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(
    r"D:\AI-Personalized-Learning-Recommendation"
)

DATA_PROCESSED = (
    BASE_DIR
    / "data"
    / "processed"
)

MODELS = (
    BASE_DIR
    / "models"
)


# ============================================================
# 2. DATA FILES
# ============================================================

COURSES_FILE = (
    DATA_PROCESSED
    / "courses_ready.csv"
)

LEARNER_FILE = (
    DATA_PROCESSED
    / "learner_features_ready.csv"
)

INTERACTIONS_FILE = (
    DATA_PROCESSED
    / "synthetic_interactions.csv"
)


# ============================================================
# 3. MODEL FILES
# ============================================================

TFIDF_VECTOR_FILE = (
    MODELS
    / "tfidf_vectorizer.joblib"
)

TFIDF_MATRIX_FILE = (
    MODELS
    / "tfidf_matrix.npz"
)

KNN_MODEL_FILE = (
    MODELS
    / "knn_collaborative_model.joblib"
)

USER_COURSE_MATRIX_FILE = (
    MODELS
    / "user_course_matrix.npz"
)

USER_TO_INDEX_FILE = (
    MODELS
    / "user_to_index.joblib"
)

INDEX_TO_COURSE_FILE = (
    MODELS
    / "index_to_course.joblib"
)

COURSE_TO_INDEX_FILE = (
    MODELS
    / "course_to_index.joblib"
)


# ============================================================
# 4. OUTPUT FILE
# ============================================================

OUTPUT_FILE = (
    DATA_PROCESSED
    / "final_hybrid_recommendations.csv"
)


# ============================================================
# 5. GENERAL CONFIGURATION
# ============================================================

TOP_N = 10

CANDIDATES_PER_MODEL = 75

KNN_NEIGHBORS = 10


# ============================================================
# 6. SKILL RELEVANCE GATE CONFIGURATION
# ============================================================
#
# A course with a skill-gap score below this value is treated
# as an extremely weak match for the learner's target skills.
#
# IMPORTANT:
# This is NOT a hard-coded list of blockchain courses,
# web-development courses, or data-science courses.
#
# It is calculated from the learner's target-skill text and
# the course catalog.
#
# ============================================================

SKILL_GATE_THRESHOLD = 0.05

MIN_RELEVANT_COURSES = 10


# ============================================================
# 7. FINAL RANKING WEIGHTS
# ============================================================
#
# Skill Gap:
#   40%
#
# Content:
#   30%
#
# Collaborative:
#   15%
#
# Performance/Difficulty:
#   15%
#
# Total:
#   100%
#
# ============================================================

SKILL_WEIGHT = 0.40

CONTENT_WEIGHT = 0.30

COLLAB_WEIGHT = 0.15

PERFORMANCE_WEIGHT = 0.15


# ============================================================
# 8. HEADER
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "STEP 18D - TWO-STAGE HYBRID RECOMMENDER"
)

print(
    "WITH SKILL-RELEVANCE GATE"
)

print(
    "=" * 70
)


# ============================================================
# 9. CHECK REQUIRED FILES
# ============================================================

required_files = [

    COURSES_FILE,

    LEARNER_FILE,

    INTERACTIONS_FILE,

    TFIDF_VECTOR_FILE,

    TFIDF_MATRIX_FILE,

    KNN_MODEL_FILE,

    USER_COURSE_MATRIX_FILE,

    USER_TO_INDEX_FILE,

    INDEX_TO_COURSE_FILE,

    COURSE_TO_INDEX_FILE

]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            "\nRequired file not found:\n"
            f"{file_path}"
        )


print(
    "\nAll required project files found."
)


# ============================================================
# 10. LOAD COURSE CATALOG
# ============================================================

print(
    "\n[1/10] Loading course catalog..."
)


courses = pd.read_csv(
    COURSES_FILE
)


courses["course_id"] = (
    courses["course_id"]
    .astype(str)
)


print(
    f"Courses loaded: "
    f"{len(courses):,}"
)


# ============================================================
# 11. COURSE ID → CATALOG INDEX
# ============================================================

course_id_to_index = {

    str(course_id): index

    for index, course_id
    in enumerate(
        courses["course_id"]
    )

}


print(
    f"Catalog course IDs: "
    f"{len(course_id_to_index):,}"
)


# ============================================================
# 12. LOAD EXISTING CONTENT TF-IDF MODEL
# ============================================================

print(
    "\n[2/10] Loading content TF-IDF model..."
)


tfidf_vectorizer = load(
    TFIDF_VECTOR_FILE
)


course_tfidf = load_npz(
    TFIDF_MATRIX_FILE
)


print(
    f"TF-IDF matrix shape: "
    f"{course_tfidf.shape}"
)


# ============================================================
# 13. LOAD LEARNER PROFILES
# ============================================================

print(
    "\n[3/10] Loading learner profiles..."
)


learners = pd.read_csv(
    LEARNER_FILE
)


learners["id_student"] = (
    learners["id_student"]
    .astype(int)
)


learner_lookup = (
    learners
    .set_index("id_student")
    .to_dict("index")
)


print(
    f"Learner profiles loaded: "
    f"{len(learners):,}"
)


# ============================================================
# 14. LOAD INTERACTION DATA
# ============================================================

print(
    "\n[4/10] Loading interaction data..."
)


interactions = pd.read_csv(
    INTERACTIONS_FILE
)


interactions["user_id"] = (
    interactions["user_id"]
    .astype(int)
)


interactions["course_id"] = (
    interactions["course_id"]
    .astype(str)
)


print(
    f"Interactions loaded: "
    f"{len(interactions):,}"
)


# ============================================================
# 15. LOAD KNN MODEL + AUTHORITATIVE MAPPINGS
# ============================================================

print(
    "\n[5/10] Loading collaborative filtering model..."
)


knn = load(
    KNN_MODEL_FILE
)


user_course_matrix = load_npz(
    USER_COURSE_MATRIX_FILE
)


user_to_index = load(
    USER_TO_INDEX_FILE
)


index_to_course = load(
    INDEX_TO_COURSE_FILE
)


course_to_knn_index = load(
    COURSE_TO_INDEX_FILE
)


print(
    f"KNN user-course matrix: "
    f"{user_course_matrix.shape}"
)


print(
    f"KNN neighbors configured: "
    f"{KNN_NEIGHBORS}"
)


print(
    f"KNN course mapping entries: "
    f"{len(index_to_course):,}"
)


# ============================================================
# 16. PREPARE COURSE TEXT
# ============================================================

print(
    "\n[6/10] Building skill-gap representation..."
)


# ------------------------------------------------------------
# Make sure expected columns exist.
# ------------------------------------------------------------

for column in [

    "course_name",

    "course_description",

    "skills"

]:

    if column not in courses.columns:

        courses[column] = ""


# ------------------------------------------------------------
# Clean text fields
# ------------------------------------------------------------

course_name_text = (
    courses["course_name"]
    .fillna("")
    .astype(str)
)


course_description_text = (
    courses["course_description"]
    .fillna("")
    .astype(str)
)


course_skills_text = (
    courses["skills"]
    .fillna("")
    .astype(str)
)


# ------------------------------------------------------------
# Combined skill representation
# ------------------------------------------------------------

skill_text = (

    course_name_text

    + " "

    + course_skills_text

    + " "

    + course_description_text

)


# ============================================================
# 17. BUILD SKILL TF-IDF
# ============================================================

skill_vectorizer = TfidfVectorizer(

    stop_words="english",

    max_features=50000,

    ngram_range=(1, 2)

)


course_skill_matrix = (
    skill_vectorizer.fit_transform(
        skill_text
    )
)


print(
    f"Skill TF-IDF matrix: "
    f"{course_skill_matrix.shape}"
)


# ============================================================
# 18. NORMALIZATION FUNCTION
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype=float
    )


    if len(scores) == 0:

        return scores


    minimum = scores.min()

    maximum = scores.max()


    if maximum == minimum:

        return np.zeros_like(
            scores
        )


    return (
        (scores - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# 19. CONTENT-BASED SCORE
# ============================================================

def get_content_scores(
    interacted_course_ids
):

    scores = np.zeros(
        len(courses),
        dtype=float
    )


    valid_indices = []

    for course_id in interacted_course_ids:

        course_id = str(
            course_id
        )


        if (
            course_id
            in course_id_to_index
        ):

            valid_indices.append(

                course_id_to_index[
                    course_id
                ]

            )


    if not valid_indices:

        return scores


    # --------------------------------------------------------
    # Build learner content profile from courses previously
    # interacted with.
    # --------------------------------------------------------

    user_profile = (
        course_tfidf[
            valid_indices
        ].mean(
            axis=0
        )
    )


    user_profile = np.asarray(
        user_profile
    )


    # --------------------------------------------------------
    # Calculate similarity between learner profile and every
    # course.
    # --------------------------------------------------------

    similarities = (
        cosine_similarity(
            user_profile,
            course_tfidf
        )
        .flatten()
    )


    return normalize_scores(
        similarities
    )


# ============================================================
# 20. COLLABORATIVE FILTERING SCORE
# ============================================================

def get_collaborative_scores(
    user_id
):

    scores = np.zeros(
        len(courses),
        dtype=float
    )


    if (
        user_id
        not in user_to_index
    ):

        return scores


    user_index = (
        user_to_index[
            user_id
        ]
    )


    # --------------------------------------------------------
    # Find nearest users.
    # +1 because the learner itself can appear in the result.
    # --------------------------------------------------------

    distances, neighbors = (
        knn.kneighbors(

            user_course_matrix[
                user_index
            ],

            n_neighbors=(
                KNN_NEIGHBORS + 1
            )

        )
    )


    similarities = (
        1.0
        -
        distances[0]
    )


    neighbor_indices = (
        neighbors[0]
    )


    # --------------------------------------------------------
    # Aggregate neighboring learner preferences.
    # --------------------------------------------------------

    for (

        neighbor_index,

        similarity

    ) in zip(

        neighbor_indices,

        similarities

    ):


        if (
            neighbor_index
            == user_index
        ):

            continue


        neighbor_row = (
            user_course_matrix[
                neighbor_index
            ]
        )


        for (

            matrix_course_index,

            value

        ) in zip(

            neighbor_row.indices,

            neighbor_row.data

        ):


            # ------------------------------------------------
            # IMPORTANT:
            # Always use the authoritative mapping generated
            # during Step 11.
            # ------------------------------------------------

            course_id = (
                index_to_course.get(
                    int(
                        matrix_course_index
                    )
                )
            )


            if course_id is None:

                continue


            course_id = str(
                course_id
            )


            if (
                course_id
                not in course_id_to_index
            ):

                continue


            catalog_index = (
                course_id_to_index[
                    course_id
                ]
            )


            scores[
                catalog_index
            ] += (

                similarity
                *
                value

            )


    maximum = scores.max()


    if maximum > 0:

        scores = (
            scores
            /
            maximum
        )


    return scores


# ============================================================
# 21. SKILL-GAP SCORE
# ============================================================

def get_skill_gap_scores(

    goal,

    known_skills,

    target_skills

):

    # --------------------------------------------------------
    # Transform learner text into the same TF-IDF space as
    # the course catalog.
    # --------------------------------------------------------

    goal_vector = (
        skill_vectorizer.transform(
            [
                str(goal)
            ]
        )
    )


    known_vector = (
        skill_vectorizer.transform(
            [
                str(known_skills)
            ]
        )
    )


    target_vector = (
        skill_vectorizer.transform(
            [
                str(target_skills)
            ]
        )
    )


    # --------------------------------------------------------
    # Target skill similarity
    # --------------------------------------------------------

    target_similarity = (
        cosine_similarity(
            target_vector,
            course_skill_matrix
        )
        .flatten()
    )


    # --------------------------------------------------------
    # Known skill similarity
    # --------------------------------------------------------

    known_similarity = (
        cosine_similarity(
            known_vector,
            course_skill_matrix
        )
        .flatten()
    )


    # --------------------------------------------------------
    # Goal similarity
    # --------------------------------------------------------

    goal_similarity = (
        cosine_similarity(
            goal_vector,
            course_skill_matrix
        )
        .flatten()
    )


    # --------------------------------------------------------
    # Combined skill-gap relevance
    #
    # Target skills have the highest contribution because
    # the recommender's purpose is to help close skill gaps.
    # --------------------------------------------------------

    scores = (

        0.70
        *
        target_similarity

        +

        0.20
        *
        known_similarity

        +

        0.10
        *
        goal_similarity

    )


    return normalize_scores(
        scores
    )


# ============================================================
# 22. PERFORMANCE / DIFFICULTY SCORE
# ============================================================

def get_performance_difficulty_scores(
    user_id
):

    scores = np.zeros(
        len(courses),
        dtype=float
    )


    if (
        user_id
        not in learner_lookup
    ):

        return scores


    learner = (
        learner_lookup[
            user_id
        ]
    )


    performance = float(

        learner.get(
            "performance_score",
            0
        )

    )


    # --------------------------------------------------------
    # Convert percentage to 0-1.
    # --------------------------------------------------------

    performance = (
        performance
        /
        100.0
    )


    performance = np.clip(

        performance,

        0.0,

        1.0

    )


    # --------------------------------------------------------
    # Difficulty representation.
    # --------------------------------------------------------

    difficulty_mapping = {

        "Beginner": 0.25,

        "Conversant": 0.40,

        "Intermediate": 0.55,

        "Advanced": 0.85

    }


    for (

        index,

        difficulty

    ) in enumerate(

        courses[
            "difficulty_level"
        ]
        .fillna("")
        .astype(str)

    ):


        difficulty_value = (
            difficulty_mapping.get(

                difficulty,

                0.55

            )
        )


        suitability = (

            1.0

            -

            abs(

                performance
                -
                difficulty_value

            )

        )


        scores[index] = suitability


    return normalize_scores(
        scores
    )


# ============================================================
# 23. STAGE 1 - CANDIDATE GENERATION
# ============================================================

def generate_candidates(

    content_scores,

    collaborative_scores,

    skill_scores,

    interacted_course_ids

):

    # --------------------------------------------------------
    # Previously interacted courses
    # --------------------------------------------------------

    interacted_indices = {

        course_id_to_index[
            str(course_id)
        ]

        for course_id
        in interacted_course_ids

        if (
            str(course_id)
            in course_id_to_index
        )

    }


    # --------------------------------------------------------
    # Candidate set
    # --------------------------------------------------------

    candidate_indices = set()


    # --------------------------------------------------------
    # Candidate sources
    # --------------------------------------------------------

    score_sources = [

        (
            "content",

            content_scores

        ),

        (
            "collaborative",

            collaborative_scores

        ),

        (
            "skill",

            skill_scores

        )

    ]


    # --------------------------------------------------------
    # Take top candidates from every signal.
    # --------------------------------------------------------

    for (

        source_name,

        scores

    ) in score_sources:


        ranked_indices = (
            np.argsort(
                scores
            )[::-1]
        )


        added = 0


        for index in ranked_indices:

            index = int(
                index
            )


            if (
                index
                in interacted_indices
            ):

                continue


            candidate_indices.add(
                index
            )


            added += 1


            if (
                added
                >= CANDIDATES_PER_MODEL
            ):

                break


    # ========================================================
    # STAGE 1.5 - SKILL RELEVANCE GATE
    # ========================================================
    #
    # If there are enough relevant resources in the candidate
    # pool, remove extremely weak skill matches.
    #
    # This prevents an unrelated course from reaching the
    # final ranking solely because another signal is large.
    #
    # ========================================================

    relevant_indices = [

        index

        for index in candidate_indices

        if (

            skill_scores[index]

            >=

            SKILL_GATE_THRESHOLD

        )

    ]


    # --------------------------------------------------------
    # Apply gate only when enough relevant alternatives exist.
    # --------------------------------------------------------

    if (

        len(relevant_indices)

        >=

        MIN_RELEVANT_COURSES

    ):

        candidate_indices = set(
            relevant_indices
        )


    return sorted(
        candidate_indices
    )


# ============================================================
# 24. STAGE 2 - PERSONALIZED RANKING
# ============================================================

def rank_candidates(

    candidate_indices,

    content_scores,

    collaborative_scores,

    skill_scores,

    performance_scores

):

    # --------------------------------------------------------
    # Initial result table
    # --------------------------------------------------------

    results = courses.iloc[
        candidate_indices
    ].copy()


    # --------------------------------------------------------
    # Add component scores
    # --------------------------------------------------------

    results[
        "content_score"
    ] = (
        content_scores[
            candidate_indices
        ]
    )


    results[
        "collaborative_score"
    ] = (
        collaborative_scores[
            candidate_indices
        ]
    )


    results[
        "skill_gap_score"
    ] = (
        skill_scores[
            candidate_indices
        ]
    )


    results[
        "performance_difficulty_score"
    ] = (
        performance_scores[
            candidate_indices
        ]
    )


    # ========================================================
    # FINAL HYBRID SCORE
    # ========================================================

    results[
        "final_score"
    ] = (

        SKILL_WEIGHT
        *
        results[
            "skill_gap_score"
        ]

        +

        CONTENT_WEIGHT
        *
        results[
            "content_score"
        ]

        +

        COLLAB_WEIGHT
        *
        results[
            "collaborative_score"
        ]

        +

        PERFORMANCE_WEIGHT
        *
        results[
            "performance_difficulty_score"
        ]

    )


    # --------------------------------------------------------
    # Sort by final personalized score.
    # --------------------------------------------------------

    results = (
        results
        .sort_values(

            "final_score",

            ascending=False

        )
        .head(
            TOP_N
        )
        .copy()
    )


    return results


# ============================================================
# 25. EXPLANATION GENERATOR
# ============================================================

def create_explanation(row):

    explanations = []


    # --------------------------------------------------------
    # Skill relevance
    # --------------------------------------------------------

    if (
        row[
            "skill_gap_score"
        ]
        >=
        0.50
    ):

        explanations.append(
            "strongly supports the learner's target skills"
        )


    elif (
        row[
            "skill_gap_score"
        ]
        >=
        0.20
    ):

        explanations.append(
            "supports some of the learner's target skills"
        )


    # --------------------------------------------------------
    # Content relevance
    # --------------------------------------------------------

    if (
        row[
            "content_score"
        ]
        >=
        0.50
    ):

        explanations.append(
            "closely matches the learner's existing learning interests"
        )


    elif (
        row[
            "content_score"
        ]
        >=
        0.20
    ):

        explanations.append(
            "matches the learner's existing learning interests"
        )


    # --------------------------------------------------------
    # Collaborative relevance
    # --------------------------------------------------------

    if (
        row[
            "collaborative_score"
        ]
        >=
        0.50
    ):

        explanations.append(
            "was strongly supported by similar learner behavior"
        )


    elif (
        row[
            "collaborative_score"
        ]
        >=
        0.20
    ):

        explanations.append(
            "was supported by similar learner behavior"
        )


    # --------------------------------------------------------
    # Performance / difficulty
    # --------------------------------------------------------

    if (
        row[
            "performance_difficulty_score"
        ]
        >=
        0.70
    ):

        explanations.append(
            "has a difficulty level compatible with current performance"
        )


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not explanations:

        explanations.append(
            "selected by the combined personalization model"
        )


    return "; ".join(
        explanations
    )


# ============================================================
# 26. DEMONSTRATION LEARNER PROFILES
# ============================================================

demo_profiles = [

    {

        "user_id": 23698,

        "profile_name":
            "Data Science Learner",

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

        "user_id": 590762,

        "profile_name":
            "Full Stack Web Development Learner",

        "goal":
            "Become a Full Stack Web Developer",

        "known_skills":
            "HTML CSS JavaScript",

        "target_skills":
            "React Node.js Express MongoDB "
            "Backend Development APIs"

    },


    {

        "user_id": 2716795,

        "profile_name":
            "Blockchain Development Learner",

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
# 27. GENERATE RECOMMENDATIONS
# ============================================================

print(
    "\n[7/10] Generating two-stage recommendations..."
)


all_results = []


for profile in demo_profiles:


    user_id = (
        profile[
            "user_id"
        ]
    )


    print(
        f"\nGenerating recommendations "
        f"for learner {user_id}..."
    )


    # --------------------------------------------------------
    # Get learner's previous interactions.
    # --------------------------------------------------------

    user_interactions = (
        interactions[
            interactions[
                "user_id"
            ]
            ==
            user_id
        ]
    )


    interacted_course_ids = set(

        user_interactions[
            "course_id"
        ]
        .astype(str)
        .tolist()

    )


    # --------------------------------------------------------
    # Stage 1 component scores
    # --------------------------------------------------------

    content_scores = (
        get_content_scores(

            interacted_course_ids

        )
    )


    collaborative_scores = (
        get_collaborative_scores(

            user_id

        )
    )


    skill_scores = (
        get_skill_gap_scores(

            profile[
                "goal"
            ],

            profile[
                "known_skills"
            ],

            profile[
                "target_skills"
            ]

        )
    )


    performance_scores = (
        get_performance_difficulty_scores(

            user_id

        )
    )


    # --------------------------------------------------------
    # Candidate generation + skill gate
    # --------------------------------------------------------

    candidate_indices = (
        generate_candidates(

            content_scores,

            collaborative_scores,

            skill_scores,

            interacted_course_ids

        )
    )


    print(
        f"Candidate pool after "
        f"skill gate: "
        f"{len(candidate_indices)}"
    )


    # --------------------------------------------------------
    # Stage 2 personalized ranking
    # --------------------------------------------------------

    recommendations = (
        rank_candidates(

            candidate_indices,

            content_scores,

            collaborative_scores,

            skill_scores,

            performance_scores

        )
    )


    # --------------------------------------------------------
    # Add learner information
    # --------------------------------------------------------

    recommendations.insert(

        0,

        "user_id",

        user_id

    )


    recommendations.insert(

        1,

        "profile_name",

        profile[
            "profile_name"
        ]

    )


    recommendations[
        "goal"
    ] = profile[
        "goal"
    ]


    recommendations[
        "known_skills"
    ] = profile[
        "known_skills"
    ]


    recommendations[
        "target_skills"
    ] = profile[
        "target_skills"
    ]


    # --------------------------------------------------------
    # Add explanation
    # --------------------------------------------------------

    recommendations[
        "explanation"
    ] = recommendations.apply(

        create_explanation,

        axis=1

    )


    all_results.append(
        recommendations
    )


# ============================================================
# 28. COMBINE ALL RESULTS
# ============================================================

print(
    "\n[8/10] Combining recommendation results..."
)


final_results = pd.concat(

    all_results,

    ignore_index=True

)


# ============================================================
# 29. SAVE RESULTS
# ============================================================

print(
    "\n[9/10] Saving final recommendations..."
)


final_results.to_csv(

    OUTPUT_FILE,

    index=False

)


print(
    f"\nSaved to:\n"
    f"{OUTPUT_FILE}"
)


# ============================================================
# 30. DISPLAY FINAL RECOMMENDATIONS
# ============================================================

print(
    "\n[10/10] FINAL TWO-STAGE RECOMMENDATIONS"
)


for profile in demo_profiles:


    user_id = (
        profile[
            "user_id"
        ]
    )


    profile_results = (
        final_results[
            final_results[
                "user_id"
            ]
            ==
            user_id
        ]
    )


    print(
        "\n"
        + "=" * 70
    )


    print(
        f"LEARNER: "
        f"{profile['profile_name']}"
    )


    print(
        f"User ID: "
        f"{user_id}"
    )


    print(
        f"Goal: "
        f"{profile['goal']}"
    )


    print(
        f"Known Skills: "
        f"{profile['known_skills']}"
    )


    print(
        f"Target Skills: "
        f"{profile['target_skills']}"
    )


    print(
        "\nTOP 10 RECOMMENDATIONS:"
    )


    for rank, (_, row) in enumerate(

        profile_results.iterrows(),

        start=1

    ):


        print(
            f"\n{rank}. "
            f"{row['course_name']}"
        )


        print(
            f"   Course ID: "
            f"{row['course_id']}"
        )


        print(
            f"   Difficulty: "
            f"{row['difficulty_level']}"
        )


        print(
            f"   Final Score: "
            f"{row['final_score']:.4f}"
        )


        print(
            f"   Content: "
            f"{row['content_score']:.4f}"
        )


        print(
            f"   Collaborative: "
            f"{row['collaborative_score']:.4f}"
        )


        print(
            f"   Skill Gap: "
            f"{row['skill_gap_score']:.4f}"
        )


        print(
            f"   Performance/Difficulty: "
            f"{row['performance_difficulty_score']:.4f}"
        )


        print(
            f"   Why: "
            f"{row['explanation']}"
        )


# ============================================================
# 31. FINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 70
)


print(
    "STEP 18D COMPLETED"
)


print(
    "=" * 70
)


print(
    "\nArchitecture:"
)


print(
    "Stage 1:"
)


print(
    "Content + Collaborative + Skill "
    "candidate generation"
)


print(
    "\nStage 1.5:"
)


print(
    "Skill-Relevance Gate"
)


print(
    "\nStage 2:"
)


print(
    "Personalized hybrid ranking"
)


print(
    "\nFinal ranking formula:"
)


print(
    "40% Skill Gap + "
    "30% Content + "
    "15% Collaborative + "
    "15% Performance/Difficulty"
)


print(
    "\nSkill relevance threshold:"
)


print(
    f"{SKILL_GATE_THRESHOLD:.2f}"
)


print(
    "\nMinimum relevant candidates required "
    "before applying the gate:"
)


print(
    f"{MIN_RELEVANT_COURSES}"
)


print(
    "\nOutput:"
)


print(
    OUTPUT_FILE
)


print(
    "\nIMPORTANT:"
)


print(
    "The demonstration learner goals and target "
    "skills are scenario-based inputs and are "
    "not claimed to come directly from OULAD."
)


print(
    "\nThe synthetic interaction dataset is used "
    "for controlled experimentation and should "
    "not be interpreted as observed real-world "
    "student behavior."
)


print(
    "\nNext step after validating this output:"
)


print(
    "STEP 19 - FORMAL RECOMMENDER EVALUATION"
)