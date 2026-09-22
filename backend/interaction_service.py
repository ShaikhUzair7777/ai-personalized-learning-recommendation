from collections import defaultdict
from typing import Dict, List

from backend.supabase_client import supabase


INTERACTION_WEIGHTS = {
    "viewed": 0.20,
    "clicked": 0.40,
    "started": 0.70,
    "saved": 0.80,
    "completed": 1.00,
}


def get_user_interactions(user_id: str) -> List[dict]:
    """
    Fetch the authenticated user's course interactions
    from Supabase.
    """

    response = (
        supabase
        .table("course_interactions")
        .select(
            "course_id, interaction_type, interaction_value, created_at"
        )
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    return response.data or []


def calculate_live_interest(user_id: str) -> Dict[str, float]:
    """
    Convert raw interactions into a live interest score
    for each course.

    Multiple interactions with the same course are accumulated,
    then capped at 1.0.
    """

    interactions = get_user_interactions(user_id)

    course_scores = defaultdict(float)

    for interaction in interactions:
        course_id = str(interaction["course_id"])
        interaction_type = interaction["interaction_type"]

        weight = INTERACTION_WEIGHTS.get(interaction_type, 0.0)

        interaction_value = interaction.get("interaction_value", 1)

        try:
            interaction_value = float(interaction_value)
        except (TypeError, ValueError):
            interaction_value = 1.0

        course_scores[course_id] += weight * interaction_value

    # Keep scores in the 0–1 range.
    normalized_scores = {
        course_id: min(score, 1.0)
        for course_id, score in course_scores.items()
    }

    return normalized_scores