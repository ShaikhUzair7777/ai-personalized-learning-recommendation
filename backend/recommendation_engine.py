from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.interaction_service import (
    get_user_interactions,
    calculate_live_interest,
)

from backend.supabase_client import supabase


class RecommendationEngine:

    def __init__(self):

        print("\n" + "=" * 70)
        print("INITIALIZING RECOMMENDATION ENGINE")
        print("=" * 70)

        self.root = Path(__file__).resolve().parent.parent

        # ==============================================================
        # COURSE CATALOG
        # ==============================================================

        print("\nLoading course catalog...")

        self.courses = pd.read_csv(
            self.root
            / "data"
            / "processed"
            / "courses_ready.csv"
        )

        self.courses["course_id"] = (
            self.courses["course_id"]
            .astype(str)
            .str.strip()
        )

        for column in [
            "course_name",
            "course_description",
            "skills",
            "combined_text",
        ]:
            if column in self.courses.columns:
                self.courses[column] = (
                    self.courses[column]
                    .fillna("")
                    .astype(str)
                )

        # Searchable course representation.
        self.courses["search_text"] = (
            self.courses["course_name"]
            + " "
            + self.courses["skills"]
            + " "
            + self.courses["course_description"]
        ).str.lower()

        self.course_id_to_index = {
            course_id: index
            for index, course_id in enumerate(
                self.courses["course_id"]
            )
        }

        print(
            f"Courses loaded: {len(self.courses):,}"
        )

        # ==============================================================
        # LEARNER PROFILES
        # ==============================================================

        print("\nLoading learner profiles...")

        learner_path = (
            self.root
            / "data"
            / "processed"
            / "learner_features_ready.csv"
        )

        self.learners = pd.read_csv(
            learner_path
        )

        if "id_student" in self.learners.columns:

            self.learners["id_student"] = (
                self.learners["id_student"]
                .astype(int)
            )

            self.learner_index = (
                self.learners
                .set_index("id_student")
            )

        else:

            self.learner_index = pd.DataFrame()

        print(
            f"Learners loaded: {len(self.learners):,}"
        )

        # ==============================================================
        # CONTENT TF-IDF
        # ==============================================================

        print("\nLoading TF-IDF model...")

        self.tfidf_matrix = load_npz(
            self.root
            / "models"
            / "tfidf_matrix.npz"
        )

        self.tfidf_vectorizer = joblib.load(
            self.root
            / "models"
            / "tfidf_vectorizer.joblib"
        )

        print(
            f"TF-IDF matrix: {self.tfidf_matrix.shape}"
        )

        # ==============================================================
        # COLLABORATIVE MODEL
        # ==============================================================

        print("\nLoading collaborative model...")

        self.user_course_matrix = load_npz(
            self.root
            / "models"
            / "user_course_matrix.npz"
        )

        self.user_to_index = joblib.load(
            self.root
            / "models"
            / "user_to_index.joblib"
        )

        self.course_to_index = joblib.load(
            self.root
            / "models"
            / "course_to_index.joblib"
        )

        self.index_to_user = joblib.load(
            self.root
            / "models"
            / "index_to_user.joblib"
        )

        self.index_to_course = joblib.load(
            self.root
            / "models"
            / "index_to_course.joblib"
        )

        self.knn_model = joblib.load(
            self.root
            / "models"
            / "knn_collaborative_model.joblib"
        )

        print(
            f"KNN users: "
            f"{self.user_course_matrix.shape[0]:,}"
        )

        # ==============================================================
        # HYBRID CONFIGURATION
        # ==============================================================

        print("\nHybrid configuration:")

        self.hybrid_config = joblib.load(
            self.root
            / "models"
            / "hybrid_config.joblib"
        )

        self.skill_weight = float(
            self.hybrid_config["skill_weight"]
        )

        self.content_weight = float(
            self.hybrid_config["content_weight"]
        )

        self.collaborative_weight = float(
            self.hybrid_config["collaborative_weight"]
        )

        self.performance_weight = float(
            self.hybrid_config["performance_weight"]
        )

        print(
            f"Skill          : {self.skill_weight}"
        )

        print(
            f"Content        : {self.content_weight}"
        )

        print(
            f"Collaborative  : {self.collaborative_weight}"
        )

        print(
            f"Performance    : {self.performance_weight}"
        )

        # ==============================================================
        # SKILL TF-IDF
        # ==============================================================

        print(
            "\nBuilding skill representation..."
        )

        self.skill_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
            min_df=1,
        )

        self.skill_matrix = (
            self.skill_vectorizer.fit_transform(
                self.courses["skills"]
            )
        )

        print(
            f"Skill TF-IDF matrix: "
            f"{self.skill_matrix.shape}"
        )

        # ==============================================================
        # READY
        # ==============================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "RECOMMENDATION ENGINE READY"
        )

        print(
            "=" * 70
        )

    # ==================================================================
    # NORMALIZATION
    # ==================================================================

    @staticmethod
    def normalize_scores(scores):

        scores = np.asarray(
            scores,
            dtype=float,
        )

        if len(scores) == 0:
            return scores

        min_value = np.nanmin(scores)
        max_value = np.nanmax(scores)

        if (
            max_value - min_value
            < 1e-12
        ):
            return np.zeros_like(
                scores
            )

        return (
            scores - min_value
        ) / (
            max_value - min_value
        )

    # ==================================================================
    # LEARNING DOMAIN NORMALIZATION
    # ==================================================================

    @staticmethod
    def normalize_learning_text(
        text: str
    ) -> str:

        text = str(
            text or ""
        ).lower()

        replacements = {

            "dapps": (
                "dapps decentralized applications "
                "decentralized application"
            ),

            "dapp": (
                "dapps decentralized applications "
                "decentralized application"
            ),

            "dapps developer": (
                "dapps developer decentralized applications "
                "blockchain ethereum smart contract"
            ),

            "dapp developer": (
                "dapp developer decentralized applications "
                "blockchain ethereum smart contract"
            ),

            "hardhat": (
                "hardhat ethereum smart contracts "
                "solidity blockchain decentralized applications"
            ),

            "web3": (
                "web3 blockchain ethereum "
                "decentralized applications smart contracts"
            ),

            "solidity": (
                "solidity smart contract ethereum "
                "blockchain decentralized applications"
            ),

            "smart contract": (
                "smart contract smart contracts "
                "ethereum solidity blockchain"
            ),

            "smart contracts": (
                "smart contract smart contracts "
                "ethereum solidity blockchain"
            ),

            "ethereum": (
                "ethereum blockchain smart contracts "
                "solidity decentralized applications"
            ),

            "blockchain development": (
                "blockchain ethereum smart contracts "
                "solidity decentralized applications"
            ),
        }

        for source, replacement in sorted(
            replacements.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):

            text = text.replace(
                source,
                replacement,
            )

        return text

    # ==================================================================
    # DOMAIN SIGNALS
    # ==================================================================

    def calculate_domain_relevance_scores(
        self,
        goal: str = "",
        known_skills: str = "",
        target_skills: str = "",
    ):

        raw_profile = " ".join(
            [
                goal or "",
                known_skills or "",
                target_skills or "",
            ]
        )

        profile = self.normalize_learning_text(
            raw_profile
        )

        profile_terms = {

            "blockchain": [
                "blockchain",
                "ethereum",
                "smart contract",
                "smart contracts",
                "solidity",
                "dapps",
                "decentralized applications",
                "web3",
                "hardhat",
            ],

            "machine learning": [
                "machine learning",
                "ml",
                "deep learning",
                "neural network",
                "tensorflow",
                "pytorch",
            ],

            "data science": [
                "data science",
                "data analysis",
                "statistics",
                "machine learning",
                "python",
            ],

            "web development": [
                "javascript",
                "react",
                "html",
                "css",
                "frontend",
                "backend",
                "web development",
            ],
        }

        detected_domains = []

        for domain, terms in profile_terms.items():

            if any(
                term in profile
                for term in terms
            ):

                detected_domains.append(
                    domain
                )

        scores = np.zeros(
            len(self.courses),
            dtype=float,
        )

        if not detected_domains:
            return scores

        for domain in detected_domains:

            terms = profile_terms[
                domain
            ]

            domain_mask = np.zeros(
                len(self.courses),
                dtype=bool,
            )

            for term in terms:

                domain_mask |= (
                    self.courses[
                        "search_text"
                    ]
                    .str.contains(
                        term,
                        regex=False,
                        na=False,
                    )
                    .to_numpy()
                )

            scores[
                domain_mask
            ] += 1.0

        return np.clip(
            scores,
            0.0,
            1.0,
        )

    # ==================================================================
    # PERFORMANCE
    # ==================================================================

    def get_performance_score(
        self,
        user_id: Optional[int] = None,
        performance_score: Optional[float] = None,
    ):

        if performance_score is not None:

            return float(
                np.clip(
                    performance_score / 100.0,
                    0.0,
                    1.0,
                )
            )

        if (
            user_id is not None
            and not self.learner_index.empty
        ):

            try:

                if (
                    user_id
                    in self.learner_index.index
                ):

                    row = (
                        self.learner_index.loc[
                            user_id
                        ]
                    )

                    if (
                        "performance_score"
                        in row.index
                    ):

                        value = float(
                            row[
                                "performance_score"
                            ]
                        )

                        if value > 1:
                            value /= 100.0

                        return float(
                            np.clip(
                                value,
                                0.0,
                                1.0,
                            )
                        )

                    if (
                        "average_score"
                        in row.index
                    ):

                        value = float(
                            row[
                                "average_score"
                            ]
                        )

                        if value > 1:
                            value /= 100.0

                        return float(
                            np.clip(
                                value,
                                0.0,
                                1.0,
                            )
                        )

            except Exception:
                pass

        return 0.60

    # ==================================================================
    # PERFORMANCE / DIFFICULTY
    # ==================================================================

    def performance_difficulty_scores(
        self,
        performance_value: float,
    ):

        difficulties = (
            self.courses[
                "difficulty_level"
            ]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        scores = np.full(
            len(self.courses),
            0.50,
            dtype=float,
        )

        low_mask = (
            difficulties.str.contains(
                "beginner|basic|easy",
                regex=True,
            )
        )

        medium_mask = (
            difficulties.str.contains(
                "intermediate",
                regex=True,
            )
        )

        advanced_mask = (
            difficulties.str.contains(
                "advanced|expert",
                regex=True,
            )
        )

        if performance_value < 0.40:

            scores[low_mask] = 1.00
            scores[medium_mask] = 0.65
            scores[advanced_mask] = 0.30

        elif performance_value < 0.70:

            scores[low_mask] = 0.70
            scores[medium_mask] = 1.00
            scores[advanced_mask] = 0.60

        else:

            scores[low_mask] = 0.45
            scores[medium_mask] = 0.75
            scores[advanced_mask] = 1.00

        return scores

    # ==================================================================
    # SKILL SCORE
    # ==================================================================

    def calculate_skill_scores(
        self,
        goal: str = "",
        known_skills: str = "",
        target_skills: str = "",
    ):

        goal = self.normalize_learning_text(
            goal or ""
        )

        known_skills = (
            self.normalize_learning_text(
                known_skills or ""
            )
        )

        target_skills = (
            self.normalize_learning_text(
                target_skills or ""
            )
        )

        target_vector = (
            self.skill_vectorizer.transform(
                [target_skills]
            )
        )

        target_scores = cosine_similarity(
            target_vector,
            self.skill_matrix,
        ).ravel()

        known_vector = (
            self.skill_vectorizer.transform(
                [known_skills]
            )
        )

        known_scores = cosine_similarity(
            known_vector,
            self.skill_matrix,
        ).ravel()

        goal_vector = (
            self.skill_vectorizer.transform(
                [goal]
            )
        )

        goal_scores = cosine_similarity(
            goal_vector,
            self.skill_matrix,
        ).ravel()

        scores = (
            0.70 * target_scores
            + 0.20 * known_scores
            + 0.10 * goal_scores
        )

        return self.normalize_scores(
            scores
        )

    # ==================================================================
    # CONTENT SCORE
    # ==================================================================

    def calculate_content_scores(
        self,
        user_id: Optional[int] = None,
        goal: str = "",
        known_skills: str = "",
        target_skills: str = "",
    ):

        # --------------------------------------------------------------
        # Existing synthetic user
        # --------------------------------------------------------------

        if (
            user_id is not None
            and user_id in self.user_to_index
        ):

            try:

                user_index = (
                    self.user_to_index[
                        user_id
                    ]
                )

                interaction_vector = (
                    self.user_course_matrix[
                        user_index
                    ]
                )

                interacted_indices = (
                    interaction_vector.indices
                )

                if len(
                    interacted_indices
                ) > 0:

                    user_profile = (
                        self.tfidf_matrix[
                            interacted_indices
                        ].mean(axis=0)
                    )

                    scores = cosine_similarity(
                        user_profile,
                        self.tfidf_matrix,
                    ).ravel()

                    return self.normalize_scores(
                        scores
                    )

            except Exception:
                pass

        # --------------------------------------------------------------
        # New / custom / Supabase user
        # --------------------------------------------------------------

        profile_text = " ".join(
            [
                goal or "",
                known_skills or "",
                target_skills or "",
            ]
        )

        profile_text = (
            self.normalize_learning_text(
                profile_text
            )
        )

        if not profile_text:

            return np.zeros(
                len(self.courses)
            )

        profile_vector = (
            self.tfidf_vectorizer.transform(
                [profile_text]
            )
        )

        scores = cosine_similarity(
            profile_vector,
            self.tfidf_matrix,
        ).ravel()

        return self.normalize_scores(
            scores
        )

    # ==================================================================
    # COLLABORATIVE SCORE
    # ==================================================================

    def calculate_collaborative_scores(
        self,
        user_id: Optional[int] = None,
    ):

        scores = np.zeros(
            len(self.courses),
            dtype=float,
        )

        if user_id is None:
            return scores

        if user_id not in self.user_to_index:
            return scores

        try:

            user_index = (
                self.user_to_index[
                    user_id
                ]
            )

            user_vector = (
                self.user_course_matrix[
                    user_index
                ]
            )

            distances, indices = (
                self.knn_model.kneighbors(
                    user_vector,
                    return_distance=True,
                )
            )

            neighbor_indices = (
                indices[0][1:]
            )

            neighbor_distances = (
                distances[0][1:]
            )

            for (
                neighbor_index,
                distance,
            ) in zip(
                neighbor_indices,
                neighbor_distances,
            ):

                similarity = max(
                    0.0,
                    1.0 - float(
                        distance
                    ),
                )

                neighbor_vector = (
                    self.user_course_matrix[
                        neighbor_index
                    ]
                )

                for (
                    course_index,
                    value,
                ) in zip(
                    neighbor_vector.indices,
                    neighbor_vector.data,
                ):

                    if (
                        course_index
                        < len(scores)
                    ):

                        scores[
                            course_index
                        ] += (
                            similarity
                            * float(value)
                        )

            return self.normalize_scores(
                scores
            )

        except Exception:
            return scores

    # ==================================================================
    # LIVE PREFERENCE PROFILE
    # ==================================================================

    def calculate_live_preference_scores(
        self,
        supabase_user_id: Optional[str] = None,
    ):

        scores = np.zeros(
            len(self.courses),
            dtype=float,
        )

        if not supabase_user_id:

            return (
                scores,
                set(),
            )

        try:

            interactions = (
                get_user_interactions(
                    supabase_user_id
                )
            )

            if not interactions:

                return (
                    scores,
                    set(),
                )

            interaction_weights = {
                "viewed": 0.20,
                "clicked": 0.40,
                "started": 0.70,
                "saved": 0.80,
                "completed": 1.00,
            }

            profile_vectors = []
            profile_weights = []

            interacted_courses = set()

            for interaction in interactions:

                course_id = str(
                    interaction.get(
                        "course_id",
                        "",
                    )
                ).strip()

                interaction_type = (
                    interaction.get(
                        "interaction_type",
                        "",
                    )
                )

                if not course_id:
                    continue

                if (
                    course_id
                    not in self.course_id_to_index
                ):
                    continue

                interacted_courses.add(
                    course_id
                )

                weight = (
                    interaction_weights.get(
                        interaction_type,
                        0.0,
                    )
                )

                if weight <= 0:
                    continue

                interaction_value = (
                    interaction.get(
                        "interaction_value",
                        1,
                    )
                )

                try:

                    interaction_value = float(
                        interaction_value
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    interaction_value = 1.0

                weight *= max(
                    0.0,
                    interaction_value,
                )

                if weight <= 0:
                    continue

                course_index = (
                    self.course_id_to_index[
                        course_id
                    ]
                )

                profile_vectors.append(
                    self.tfidf_matrix[
                        course_index
                    ]
                )

                profile_weights.append(
                    weight
                )

            if not profile_vectors:

                return (
                    scores,
                    interacted_courses,
                )

            total_weight = sum(
                profile_weights
            )

            if total_weight <= 0:

                return (
                    scores,
                    interacted_courses,
                )

            weighted_profile = (
                profile_vectors[0]
                * profile_weights[0]
            )

            for (
                vector,
                weight,
            ) in zip(
                profile_vectors[1:],
                profile_weights[1:],
            ):

                weighted_profile = (
                    weighted_profile
                    + vector * weight
                )

            weighted_profile = (
                weighted_profile
                / total_weight
            )

            scores = cosine_similarity(
                weighted_profile,
                self.tfidf_matrix,
            ).ravel()

            scores = self.normalize_scores(
                scores
            )

            return (
                scores,
                interacted_courses,
            )

        except Exception as exc:

            print(
                "Live preference profile error:",
                exc,
            )

            return (
                scores,
                set(),
            )

    # ==================================================================
    # FINAL RECOMMENDATION
    # ==================================================================

    def recommend(
        self,
        user_id: Optional[int] = None,
        goal: str = "",
        known_skills: str = "",
        target_skills: str = "",
        performance_score: Optional[float] = None,
        top_k: int = 10,
        supabase_user_id: Optional[str] = None,
    ):

        # ==============================================================
        # LOAD SUPABASE LEARNING PROFILE
        # ==============================================================

        if supabase_user_id:

            try:

                profile_response = (
                    supabase
                    .table("profiles")
                    .select(
                        "learning_goal, "
                        "known_skills, "
                        "target_skills, "
                        "performance_score"
                    )
                    .eq(
                        "id",
                        supabase_user_id,
                    )
                    .limit(1)
                    .execute()
                )

                profile_data = (
                    profile_response.data
                    or []
                )

                if profile_data:

                    profile = profile_data[0]

                    if not goal:

                        goal = (
                            profile.get(
                                "learning_goal"
                            )
                            or ""
                        )

                    if not known_skills:

                        known_skills = (
                            profile.get(
                                "known_skills"
                            )
                            or ""
                        )

                    if not target_skills:

                        target_skills = (
                            profile.get(
                                "target_skills"
                            )
                            or ""
                        )

                    if (
                        performance_score
                        is None
                    ):

                        performance_score = (
                            profile.get(
                                "performance_score"
                            )
                        )

                    print(
                        "\nLoaded Supabase learning profile:"
                    )

                    print(
                        f"Goal          : {goal}"
                    )

                    print(
                        f"Known skills  : {known_skills}"
                    )

                    print(
                        f"Target skills : {target_skills}"
                    )

                    print(
                        f"Performance   : {performance_score}"
                    )

                else:

                    print(
                        "\nNo Supabase profile found "
                        f"for user {supabase_user_id}"
                    )

            except Exception as exc:

                print(
                    "\nSupabase profile loading error:",
                    exc,
                )

        # ==============================================================
        # PERFORMANCE
        # ==============================================================

        performance_value = (
            self.get_performance_score(
                user_id=user_id,
                performance_score=performance_score,
            )
        )

        performance_scores = (
            self.performance_difficulty_scores(
                performance_value
            )
        )

        # ==============================================================
        # SKILL
        # ==============================================================

        skill_scores = (
            self.calculate_skill_scores(
                goal=goal,
                known_skills=known_skills,
                target_skills=target_skills,
            )
        )

        # ==============================================================
        # CONTENT
        # ==============================================================

        content_scores = (
            self.calculate_content_scores(
                user_id=user_id,
                goal=goal,
                known_skills=known_skills,
                target_skills=target_skills,
            )
        )

        # ==============================================================
        # DOMAIN RELEVANCE
        # ==============================================================

        domain_scores = (
            self.calculate_domain_relevance_scores(
                goal=goal,
                known_skills=known_skills,
                target_skills=target_skills,
            )
        )

        # ==============================================================
        # COLLABORATIVE
        # ==============================================================

        collaborative_scores = (
            self.calculate_collaborative_scores(
                user_id=user_id,
            )
        )

        # ==============================================================
        # HYBRID ML SCORE
        #
        # Locked architecture:
        #
        # 40% Skill
        # 30% Content
        # 15% Collaborative
        # 15% Performance
        # ==============================================================

        hybrid_scores = (
            self.skill_weight
            * skill_scores

            + self.content_weight
            * content_scores

            + self.collaborative_weight
            * collaborative_scores

            + self.performance_weight
            * performance_scores
        )

        # ==============================================================
        # DOMAIN PROTECTION
        #
        # Prevent unrelated courses from ranking highly only because
        # their difficulty happens to match the learner.
        # ==============================================================

        profile_text = (
            self.normalize_learning_text(
                " ".join(
                    [
                        goal or "",
                        known_skills or "",
                        target_skills or "",
                    ]
                )
            )
        )

        has_technical_profile = any(
            term in profile_text
            for term in [
                "blockchain",
                "ethereum",
                "solidity",
                "smart contract",
                "smart contracts",
                "dapps",
                "decentralized applications",
                "web3",
                "hardhat",
                "machine learning",
                "data science",
                "javascript",
                "react",
                "web development",
            ]
        )

        if has_technical_profile:

            relevance_mask = (
                (skill_scores > 0.0)
                | (content_scores > 0.0)
                | (domain_scores > 0.0)
            )

            hybrid_scores = np.where(
                relevance_mask,
                hybrid_scores,
                -np.inf,
            )

        # ==============================================================
        # DOMAIN RELEVANCE BOOST
        # ==============================================================

        hybrid_scores = (
            hybrid_scores
            + 0.10 * domain_scores
        )

        # ==============================================================
        # LIVE SUPABASE FEEDBACK
        # ==============================================================

        (
            live_preference_scores,
            interacted_courses,
        ) = self.calculate_live_preference_scores(
            supabase_user_id=supabase_user_id
        )

        # ==============================================================
        # ADAPTIVE SCORE
        # ==============================================================

        if supabase_user_id:

            final_scores = (
                0.90 * hybrid_scores
                + 0.10 * live_preference_scores
            )

        else:

            final_scores = (
                hybrid_scores.copy()
            )

        # ==============================================================
        # REMOVE ALREADY INTERACTED COURSES
        # ==============================================================

        for course_id in interacted_courses:

            if (
                course_id
                in self.course_id_to_index
            ):

                index = (
                    self.course_id_to_index[
                        course_id
                    ]
                )

                final_scores[index] = (
                    -np.inf
                )

        # ==============================================================
        # TOP K
        # ==============================================================

        top_k = min(
            int(top_k),
            len(self.courses),
        )

        ranked_indices = np.argsort(
            -final_scores
        )[:top_k]

        # ==============================================================
        # RESPONSE
        # ==============================================================

        recommendations = []

        for index in ranked_indices:

            if not np.isfinite(
                final_scores[index]
            ):
                continue

            row = self.courses.iloc[
                index
            ]

            def safe_value(
                column,
            ):

                value = row.get(
                    column
                )

                if pd.isna(value):
                    return None

                return value

            recommendations.append(
                {
                    "course_id": str(
                        row["course_id"]
                    ),

                    "course_name": str(
                        row["course_name"]
                    ),

                    "university": (
                        None
                        if safe_value(
                            "university"
                        )
                        is None
                        else str(
                            safe_value(
                                "university"
                            )
                        )
                    ),

                    "difficulty_level": (
                        None
                        if safe_value(
                            "difficulty_level"
                        )
                        is None
                        else str(
                            safe_value(
                                "difficulty_level"
                            )
                        )
                    ),

                    "course_rating": (
                        None
                        if safe_value(
                            "course_rating"
                        )
                        is None
                        else float(
                            safe_value(
                                "course_rating"
                            )
                        )
                    ),

                    "course_url": (
                        None
                        if safe_value(
                            "course_url"
                        )
                        is None
                        else str(
                            safe_value(
                                "course_url"
                            )
                        )
                    ),

                    "final_score": float(
                        final_scores[index]
                    ),

                    "skill_score": float(
                        skill_scores[index]
                    ),

                    "content_score": float(
                        content_scores[index]
                    ),

                    "collaborative_score": float(
                        collaborative_scores[index]
                    ),

                    "performance_score": float(
                        performance_scores[index]
                    ),
                }
            )

        return recommendations