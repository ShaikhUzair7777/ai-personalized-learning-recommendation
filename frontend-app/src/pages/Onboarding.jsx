import { useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  ArrowRight,
  ArrowLeft,
  BrainCircuit,
  Target,
  Sparkles,
  Code2,
  Rocket,
  BarChart3,
} from "lucide-react";

import { supabase } from "../lib/supabase";
import { useAuth } from "../context/AuthContext";
import { getCustomRecommendations } from "../services/api";

import LoadingAI from "../components/LoadingAI";
import ErrorState from "../components/ErrorState";
import AnimatedSection from "../components/AnimatedSection";

function Onboarding() {
  const navigate = useNavigate();
  const { user } = useAuth();

  const [formData, setFormData] = useState({
    goal: "",
    known_skills: "",
    target_skills: "",
    performance_score: 70,
    top_k: 10,
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (event) => {
    const { name, value } = event.target;

    setFormData((previous) => ({
      ...previous,
      [name]: value,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");

    // ---------------------------------------------------------
    // 1. Check authentication
    // ---------------------------------------------------------

    if (!user) {
      setError(
        "You must be logged in to create your learning profile."
      );
      return;
    }

    // ---------------------------------------------------------
    // 2. Validate learning goal
    // ---------------------------------------------------------

    if (!formData.goal.trim()) {
      setError("Please enter your learning goal.");
      return;
    }

    setLoading(true);

    try {
      // -------------------------------------------------------
      // STEP 1 — Generate recommendations from FastAPI
      // -------------------------------------------------------

      const recommendationPayload = {
        goal: formData.goal.trim(),
        known_skills: formData.known_skills.trim(),
        target_skills: formData.target_skills.trim(),
        performance_score: Number(formData.performance_score),
        top_k: Number(formData.top_k),
      };

      console.log(
        "Sending recommendation request:",
        recommendationPayload
      );

      const response = await getCustomRecommendations(
        recommendationPayload
      );

      console.log(
        "Recommendation response:",
        response
      );

      // -------------------------------------------------------
      // STEP 2 — Save / update learning profile in Supabase
      // -------------------------------------------------------

      const { error: profileError } = await supabase
        .from("profiles")
        .upsert(
          {
            id: user.id,
            email: user.email,
            learning_goal: formData.goal.trim(),
            known_skills: formData.known_skills.trim(),
            target_skills: formData.target_skills.trim(),
            performance_score: Number(
              formData.performance_score
            ),
            updated_at: new Date().toISOString(),
          },
          {
            onConflict: "id",
          }
        );

      if (profileError) {
        console.error(
          "Supabase profile error:",
          profileError
        );

        throw new Error(
          "Your recommendations were generated, but your learning profile could not be saved."
        );
      }

      // -------------------------------------------------------
      // STEP 3 — Save recommendation history in Supabase
      // -------------------------------------------------------

      const recommendations =
        response?.recommendations || [];

      const historyRows = recommendations.map(
        (course) => ({
          user_id: user.id,
          course_id: course.course_id,

          final_score: Number(
            course.final_score ?? 0
          ),

          skill_score: Number(
            course.skill_score ?? 0
          ),

          content_score: Number(
            course.content_score ?? 0
          ),

          collaborative_score: Number(
            course.collaborative_score ?? 0
          ),

          performance_score: Number(
            course.performance_score ?? 0
          ),
        })
      );

      console.log(
        "Recommendation history rows:",
        historyRows
      );

      if (historyRows.length > 0) {
        const { error: historyError } =
          await supabase
            .from("recommendation_history")
            .insert(historyRows);

        if (historyError) {
          // History is useful but should not prevent
          // the user from reaching the dashboard.
          console.error(
            "Recommendation history save failed:",
            historyError
          );
        } else {
          console.log(
            `Saved ${historyRows.length} recommendations to Supabase.`
          );
        }
      } else {
        console.warn(
          "FastAPI returned no recommendations."
        );
      }

      // -------------------------------------------------------
      // STEP 4 — Save current session data
      // -------------------------------------------------------

      sessionStorage.setItem(
        "learningProfile",
        JSON.stringify(formData)
      );

      sessionStorage.setItem(
        "recommendations",
        JSON.stringify(response)
      );

      // -------------------------------------------------------
      // STEP 5 — Navigate to dashboard
      // -------------------------------------------------------

      navigate("/dashboard");
    } catch (err) {
      console.error(
        "Onboarding failed:",
        err
      );

      setError(
        err?.message ||
          "Unable to generate recommendations. Please make sure the FastAPI backend is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="onboarding-page">
      {loading && <LoadingAI />}

      <div className="onboarding-glow glow-left"></div>
      <div className="onboarding-glow glow-right"></div>

      <AnimatedSection
        animation="fade-up"
        delay={0}
      >
        <div className="onboarding-container">

          {/* -------------------------------------------------
              HEADER
          ------------------------------------------------- */}

          <div className="onboarding-header">

            <div className="onboarding-icon">
              <BrainCircuit size={28} />
            </div>

            <div className="eyebrow">
              <Sparkles size={14} />
              AI LEARNING PROFILE
            </div>

            <h1>
              Build your
              <span> personalized path.</span>
            </h1>

            <p>
              Tell us where you are today and where you
              want to go. Our hybrid recommendation engine
              will create a learning path around your skills
              and goals.
            </p>
          </div>

          {/* -------------------------------------------------
              FORM
          ------------------------------------------------- */}

          <form
            className="onboarding-form"
            onSubmit={handleSubmit}
          >

            {/* =================================================
                GOAL
            ================================================= */}

            <div className="form-section">

              <div className="form-section-header">

                <div className="form-section-icon">
                  <Target size={18} />
                </div>

                <div>
                  <h2>
                    What's your learning goal?
                  </h2>

                  <p>
                    Describe what you want to achieve.
                  </p>
                </div>

              </div>

              <textarea
                name="goal"
                value={formData.goal}
                onChange={handleChange}
                placeholder="Example: Become a data scientist and build machine learning applications"
                rows="4"
              />

            </div>

            {/* =================================================
                KNOWN SKILLS
            ================================================= */}

            <div className="form-section">

              <div className="form-section-header">

                <div className="form-section-icon">
                  <Code2 size={18} />
                </div>

                <div>
                  <h2>
                    What do you already know?
                  </h2>

                  <p>
                    Separate your current skills with commas.
                  </p>
                </div>

              </div>

              <input
                type="text"
                name="known_skills"
                value={formData.known_skills}
                onChange={handleChange}
                placeholder="Python, SQL, statistics, data analysis"
              />

              <div className="input-hint">
                Example: Python, Java, SQL, HTML, CSS
              </div>

            </div>

            {/* =================================================
                TARGET SKILLS
            ================================================= */}

            <div className="form-section">

              <div className="form-section-header">

                <div className="form-section-icon">
                  <Rocket size={18} />
                </div>

                <div>
                  <h2>
                    What skills do you want to learn?
                  </h2>

                  <p>
                    Tell the AI which skills you want to
                    develop.
                  </p>
                </div>

              </div>

              <input
                type="text"
                name="target_skills"
                value={formData.target_skills}
                onChange={handleChange}
                placeholder="Machine Learning, Deep Learning, AI, Data Visualization"
              />

              <div className="input-hint">
                These skills strongly influence your
                recommendations.
              </div>

            </div>

            {/* =================================================
                PERFORMANCE
            ================================================= */}

            <div className="form-section">

              <div className="form-section-header">

                <div className="form-section-icon">
                  <BarChart3 size={18} />
                </div>

                <div>
                  <h2>
                    How would you rate your current
                    performance?
                  </h2>

                  <p>
                    This helps the system match course
                    difficulty.
                  </p>
                </div>

                <div className="performance-value">
                  {formData.performance_score}
                </div>

              </div>

              <div className="range-container">

                <input
                  type="range"
                  name="performance_score"
                  min="0"
                  max="100"
                  value={formData.performance_score}
                  onChange={handleChange}
                  className="performance-range"
                />

                <div className="range-labels">
                  <span>Beginner</span>
                  <span>Intermediate</span>
                  <span>Advanced</span>
                </div>

              </div>

            </div>

            {/* =================================================
                ERROR
            ================================================= */}

            {error && (
              <ErrorState
                title="Profile Error"
                message={error}
                onRetry={() => setError("")}
                actionText="Dismiss Error"
              />
            )}

            {/* =================================================
                SUBMIT
            ================================================= */}

            <div className="form-submit">

              <button
                type="submit"
                className="generate-button"
                disabled={loading}
              >

                {loading ? (
                  <>
                    <span className="loading-spinner"></span>
                    Building your path...
                  </>
                ) : (
                  <>
                    Generate My Learning Path
                    <ArrowRight size={18} />
                  </>
                )}

              </button>

              <div className="privacy-note">
                <BrainCircuit size={14} />

                Your profile is used only to personalize
                recommendations.
              </div>

            </div>

          </form>

          {/* -------------------------------------------------
              BACK BUTTON
          ------------------------------------------------- */}

          <button
            type="button"
            className="onboarding-back"
            onClick={() => navigate("/")}
          >
            <ArrowLeft size={15} />
            Back to home
          </button>

        </div>
      </AnimatedSection>
    </main>
  );
}

export default Onboarding;