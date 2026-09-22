from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    CustomRecommendationRequest,
    HealthResponse,
    RecommendationResponse
)

from backend.recommendation_engine import RecommendationEngine
from backend.auth import get_authenticated_user


app = FastAPI(
    title="AI Personalized Learning Recommendation API",
    description="Hybrid machine learning recommendation API for personalized learning resources.",
    version="1.0.0"
)


# =============================================================
# CORS
# =============================================================

app.add_middleware(
    CORSMiddleware,
   allow_origins=[
    "https://ai-personalized-learning-recommenda.vercel.app",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================
# LOAD ML ENGINE
# =============================================================

print("\nLoading recommendation engine...")

engine = RecommendationEngine()


# =============================================================
# ROOT
# =============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "message": "AI Personalized Learning Recommendation API is running."
    }


# =============================================================
# HEALTH
# =============================================================

@app.get(
    "/api/health",
    response_model=HealthResponse
)
def health():

    return {
        "status": "healthy",
        "message": "FastAPI backend and ML recommendation engine are operational."
    }


# =============================================================
# AUTHENTICATED LEARNER RECOMMENDATIONS
# =============================================================

@app.get(
    "/api/recommendations",
    response_model=RecommendationResponse
)
def get_authenticated_recommendations(
    user=Depends(get_authenticated_user)
):

    try:

        # Supabase UUID is used to access the live user feedback layer.
        supabase_user_id = str(user.id)

        # For now, the ML learner ID is not taken from the frontend.
        # The authenticated Supabase user is used for the adaptive layer.
        recommendations = engine.recommend(
            user_id=None,
            supabase_user_id=supabase_user_id,
            top_k=10
        )

        return {
            "user_id": 0,
            "recommendations": recommendations
        }

    except Exception as e:

        print(f"Recommendation error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Failed to generate recommendations."
        )


# =============================================================
# CUSTOM AUTHENTICATED RECOMMENDATIONS
# =============================================================

@app.post(
    "/api/recommendations/custom",
    response_model=RecommendationResponse
)
def custom_recommendations(
    request: CustomRecommendationRequest,
    user=Depends(get_authenticated_user)
):

    try:

        # Get the verified Supabase user ID from the JWT.
        supabase_user_id = str(user.id)

        recommendations = engine.recommend(
            user_id=None,
            supabase_user_id=supabase_user_id,
            goal=request.goal,
            known_skills=request.known_skills,
            target_skills=request.target_skills,
            performance_score=request.performance_score,
            top_k=request.top_k
        )

        return {
            "user_id": 0,
            "recommendations": recommendations
        }

    except Exception as e:

        print(f"Custom recommendation error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Failed to generate personalized recommendations."
        )