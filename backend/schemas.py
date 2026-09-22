from typing import List, Optional

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    course_id: str
    course_name: str
    university: Optional[str] = None
    difficulty_level: Optional[str] = None
    course_rating: Optional[float] = None
    course_url: Optional[str] = None

    final_score: float

    skill_score: float
    content_score: float
    collaborative_score: float
    performance_score: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]


class CustomRecommendationRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    known_skills: str = ""
    target_skills: str = ""

    performance_score: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=50
    )


class HealthResponse(BaseModel):
    status: str
    message: str