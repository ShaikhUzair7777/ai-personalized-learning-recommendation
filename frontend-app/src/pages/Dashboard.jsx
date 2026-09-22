import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import {
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleGauge,
  Code2,
  GraduationCap,
  Layers3,
  RefreshCw,
  Sparkles,
  Star,
  Target,
  Trophy,
  Zap,
} from "lucide-react";

import { supabase } from "../lib/supabase";
import { useAuth } from "../context/AuthContext";
import {
  getRecommendations,
  getCustomRecommendations,
} from "../services/api";


function Dashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();

  const [profile, setProfile] = useState(null);
  const [recommendations, setRecommendations] = useState([]);

  const [profileLoading, setProfileLoading] = useState(true);
  const [recommendationsLoading, setRecommendationsLoading] =
    useState(true);

  const [error, setError] = useState("");


  // =========================================================
  // LOAD DASHBOARD
  // =========================================================

  useEffect(() => {
    const loadDashboard = async () => {
      if (!user) {
        setProfileLoading(false);
        setRecommendationsLoading(false);
        return;
      }

      try {
        setError("");

        // ---------------------------------------------------
        // 1. Load profile from Supabase
        // ---------------------------------------------------

        const { data, error: profileError } = await supabase
          .from("profiles")
          .select(`
            id,
            email,
            full_name,
            learning_goal,
            known_skills,
            target_skills,
            performance_score,
            created_at,
            updated_at
          `)
          .eq("id", user.id)
          .single();

        if (profileError) {
          throw profileError;
        }

        const savedProfile = {
          id: data.id,
          email: data.email,
          full_name: data.full_name,
          goal: data.learning_goal || "",
          known_skills: data.known_skills || "",
          target_skills: data.target_skills || "",
          performance_score: Number(
            data.performance_score ?? 0
          ),
          created_at: data.created_at,
          updated_at: data.updated_at,
        };

        setProfile(savedProfile);


        // ---------------------------------------------------
        // 2. GET FRESH AUTHENTICATED RECOMMENDATIONS
        // ---------------------------------------------------

        const response = await getRecommendations();

        const items = Array.isArray(
          response?.recommendations
        )
          ? response.recommendations
          : [];

        setRecommendations(items);

        // Optional browser cache.
        // This is NOT used as the primary source anymore.
        sessionStorage.setItem(
          "recommendations",
          JSON.stringify(response)
        );

      } catch (dashboardError) {
        console.error(
          "Dashboard loading failed:",
          dashboardError
        );

        // ---------------------------------------------------
        // 3. FALLBACK: regenerate using saved profile
        // ---------------------------------------------------

        try {
          if (profile) {
            const payload = {
              goal: profile.goal,
              known_skills: profile.known_skills,
              target_skills: profile.target_skills,
              performance_score:
                Number(profile.performance_score),
              top_k: 10,
            };

            const fallbackResponse =
              await getCustomRecommendations(payload);

            const fallbackItems = Array.isArray(
              fallbackResponse?.recommendations
            )
              ? fallbackResponse.recommendations
              : [];

            setRecommendations(fallbackItems);
          } else {
            throw dashboardError;
          }
        } catch (fallbackError) {
          console.error(
            "Fallback recommendation generation failed:",
            fallbackError
          );

          setError(
            "We couldn't load your personalized learning path."
          );
        }
      } finally {
        setProfileLoading(false);
        setRecommendationsLoading(false);
      }
    };

    loadDashboard();

    // Important:
    // Only rerun when the authenticated user changes.
  }, [user]);


  // =========================================================
  // DERIVED VALUES
  // =========================================================

  const knownSkills = useMemo(() => {
    if (!profile?.known_skills) return [];

    return profile.known_skills
      .split(",")
      .map((skill) => skill.trim())
      .filter(Boolean);
  }, [profile]);


  const targetSkills = useMemo(() => {
    if (!profile?.target_skills) return [];

    return profile.target_skills
      .split(",")
      .map((skill) => skill.trim())
      .filter(Boolean);
  }, [profile]);


  const averageRecommendationScore = useMemo(() => {
    if (!recommendations.length) return 0;

    const total = recommendations.reduce(
      (sum, course) =>
        sum + Number(course.final_score ?? 0),
      0
    );

    return Math.round(
      (total / recommendations.length) * 100
    );
  }, [recommendations]);


  // =========================================================
  // REFRESH — LIVE ADAPTIVE RECOMMENDATIONS
  // =========================================================

  const handleRefresh = async () => {
    try {
      setRecommendationsLoading(true);
      setError("");

      // -----------------------------------------------------
      // IMPORTANT:
      // This endpoint uses the authenticated Supabase user.
      // The frontend does NOT send a user ID.
      // -----------------------------------------------------

      const response = await getRecommendations();

      const items = Array.isArray(
        response?.recommendations
      )
        ? response.recommendations
        : [];

      setRecommendations(items);

      sessionStorage.setItem(
        "recommendations",
        JSON.stringify(response)
      );

    } catch (refreshError) {
      console.error(
        "Recommendation refresh failed:",
        refreshError
      );

      setError(
        "Unable to refresh recommendations right now."
      );
    } finally {
      setRecommendationsLoading(false);
    }
  };


  // =========================================================
  // LOADING
  // =========================================================

  if (profileLoading) {
    return (
      <main className="dashboard-page">
        <div className="dashboard-container">
          <DashboardSkeleton />
        </div>
      </main>
    );
  }


  // =========================================================
  // NO PROFILE
  // =========================================================

  if (!profile) {
    return (
      <main className="dashboard-page">
        <div className="dashboard-container">

          <div className="dashboard-empty-state">

            <div className="empty-state-icon">
              <BrainCircuit size={34} />
            </div>

            <span className="section-eyebrow">
              AI PERSONALIZED LEARNING
            </span>

            <h1>
              Build your learning path
            </h1>

            <p>
              Complete your learning profile and let the
              hybrid ML engine generate personalized
              course recommendations.
            </p>

            <Link
              to="/onboarding"
              className="primary-dashboard-button"
            >
              Build My Learning Path
              <ArrowRight size={17} />
            </Link>

          </div>

        </div>
      </main>
    );
  }


  return (
    <main className="dashboard-page">

      {/* =====================================================
          BACKGROUND DECORATION
      ====================================================== */}

      <div className="dashboard-orb dashboard-orb-one" />
      <div className="dashboard-orb dashboard-orb-two" />


      <div className="dashboard-container">

        {/* ===================================================
            HERO
        ==================================================== */}

        <section className="dashboard-hero dashboard-reveal">

          <div className="hero-copy">

            <div className="dashboard-eyebrow">
              <Sparkles size={14} />
              AI PERSONALIZED DASHBOARD
            </div>

            <h1>
              Your learning path,
              <span>
                {" "}intelligently built.
              </span>
            </h1>

            <p>
              Personalized recommendations generated from
              your goals, skills, performance, and our
              hybrid machine learning engine.
            </p>

          </div>


          <div className="hero-actions">

            <button
              type="button"
              className="secondary-dashboard-button"
              onClick={() => navigate("/onboarding")}
            >
              <RefreshCw size={16} />
              Update Profile
            </button>

          </div>

        </section>


        {/* ===================================================
            STATS
        ==================================================== */}

        <section className="dashboard-stats">

          <StatCard
            icon={<Target size={19} />}
            label="Learning Goal"
            value={
              profile.goal ||
              "Personalized learning"
            }
            className="stat-purple"
          />

          <StatCard
            icon={<BarChart3 size={19} />}
            label="Performance"
            value={`${Math.round(
              profile.performance_score
            )}/100`}
            className="stat-blue"
          />

          <StatCard
            icon={<GraduationCap size={19} />}
            label="AI Recommendations"
            value={`${recommendations.length} Courses`}
            className="stat-cyan"
          />

          <StatCard
            icon={<Zap size={19} />}
            label="Recommendation Engine"
            value="Hybrid ML"
            className="stat-green"
          />

        </section>


        {/* ===================================================
            LEARNING PROFILE
        ==================================================== */}

        <section className="dashboard-section dashboard-reveal">

          <div className="section-heading">

            <div>

              <span className="section-eyebrow">
                YOUR LEARNING PROFILE
              </span>

              <h2>
                Skills we're optimizing for
              </h2>

            </div>


            <div className="profile-score-badge">

              <CircleGauge size={15} />

              {Math.round(
                profile.performance_score
              )}
              /100

            </div>

          </div>


          <div className="profile-grid">

            <SkillPanel
              title="Known Skills"
              icon={<Code2 size={17} />}
              skills={knownSkills}
              variant="known"
              emptyText="No known skills added yet."
            />

            <SkillPanel
              title="Target Skills"
              icon={<RocketIcon />}
              skills={targetSkills}
              variant="target"
              emptyText="No target skills added yet."
            />

          </div>

        </section>


        {/* ===================================================
            RECOMMENDATIONS
        ==================================================== */}

        <section className="dashboard-section recommendations-section">

          <div className="section-heading recommendations-heading">

            <div>

              <span className="section-eyebrow">
                AI RECOMMENDATIONS
              </span>

              <h2>
                Recommended for you
              </h2>

              <p>
                Courses selected using your learning profile,
                hybrid ML, and your latest learning activity.
              </p>

            </div>


            <div className="recommendation-actions">

              <div className="engine-badge">
                <BrainCircuit size={15} />
                Hybrid ML
              </div>


              {recommendations.length > 0 && (
                <button
                  type="button"
                  className="icon-refresh-button"
                  onClick={handleRefresh}
                  disabled={recommendationsLoading}
                  title="Refresh recommendations"
                >

                  <RefreshCw
                    size={17}
                    className={
                      recommendationsLoading
                        ? "spin"
                        : ""
                    }
                  />

                </button>
              )}

            </div>

          </div>


          {error && (
            <div className="dashboard-alert">

              <Zap size={17} />

              <span>
                {error}
              </span>

            </div>
          )}


          {recommendationsLoading ? (
            <CourseGridSkeleton />
          ) : recommendations.length > 0 ? (
            <>

              <div className="recommendation-summary">

                <div>

                  <strong>
                    {recommendations.length}
                  </strong>

                  <span>
                    personalized courses
                  </span>

                </div>


                <div className="summary-match">

                  <Trophy size={16} />

                  <span>
                    Avg. AI Match
                  </span>

                  <strong>
                    {averageRecommendationScore}%
                  </strong>

                </div>

              </div>


              <div className="course-grid">

                {recommendations.map(
                  (course, index) => (
                    <CourseCard
                      key={
                        course.course_id ||
                        `${course.course_name}-${index}`
                      }
                      course={course}
                      index={index}
                    />
                  )
                )}

              </div>

            </>
          ) : (
            <EmptyRecommendations />
          )}

        </section>


        {/* ===================================================
            MODEL EXPLANATION
        ==================================================== */}

        <section className="dashboard-section intelligence-section">

          <div className="section-heading">

            <div>

              <span className="section-eyebrow">
                RECOMMENDATION INTELLIGENCE
              </span>

              <h2>
                How LearnAI personalizes your path
              </h2>

            </div>

          </div>


          <div className="model-grid">

            <ModelFactor
              icon={<Target size={18} />}
              title="Skill Matching"
              value="40%"
              description="Matches your target and known skills against course content."
            />

            <ModelFactor
              icon={<Layers3 size={18} />}
              title="Content Relevance"
              value="30%"
              description="Uses course descriptions and skill-related text similarity."
            />

            <ModelFactor
              icon={<BrainCircuit size={18} />}
              title="Collaborative Signal"
              value="15%"
              description="Uses learner-course interaction patterns when available."
            />

            <ModelFactor
              icon={<BarChart3 size={18} />}
              title="Difficulty Fit"
              value="15%"
              description="Helps match course difficulty with your current performance."
            />

          </div>

        </section>


        {/* ===================================================
            FOOTER CTA
        ==================================================== */}

        <section className="dashboard-cta dashboard-reveal">

          <div className="cta-icon">
            <Sparkles size={23} />
          </div>


          <div>

            <span>
              KEEP YOUR PATH PERSONALIZED
            </span>

            <h3>
              Your goals can evolve.
            </h3>

            <p>
              Update your profile anytime to generate
              recommendations for your next learning stage.
            </p>

          </div>


          <button
            type="button"
            className="primary-dashboard-button"
            onClick={() => navigate("/onboarding")}
          >
            Update Learning Path
            <ArrowRight size={17} />
          </button>

        </section>

      </div>

    </main>
  );
}


/* ============================================================
   STAT CARD
============================================================ */

function StatCard({
  icon,
  label,
  value,
  className = "",
}) {
  return (
    <article
      className={`dashboard-stat-card ${className}`}
    >

      <div className="stat-icon">
        {icon}
      </div>

      <div className="stat-content">
        <span>{label}</span>
        <strong>{value}</strong>
      </div>

    </article>
  );
}


/* ============================================================
   SKILL PANEL
============================================================ */

function SkillPanel({
  title,
  icon,
  skills,
  variant,
  emptyText,
}) {
  return (
    <article
      className={`skill-panel skill-panel-${variant}`}
    >

      <div className="skill-panel-header">

        <div className="skill-panel-title">
          {icon}
          <span>{title}</span>
        </div>

        <span className="skill-count">
          {skills.length}
        </span>

      </div>


      {skills.length > 0 ? (
        <div className="skill-list">

          {skills.map((skill) => (
            <span
              className="skill-pill"
              key={skill}
            >
              <CheckCircle2 size={13} />
              {skill}
            </span>
          ))}

        </div>
      ) : (
        <p className="skill-empty">
          {emptyText}
        </p>
      )}

    </article>
  );
}


/* ============================================================
   COURSE CARD
============================================================ */

function CourseCard({
  course,
  index,
}) {
  const score = Math.round(
    Number(course.final_score ?? 0) * 100
  );

  const skillScore = Math.round(
    Number(course.skill_score ?? 0) * 100
  );

  const contentScore = Math.round(
    Number(course.content_score ?? 0) * 100
  );

  const performanceScore = Math.round(
    Number(course.performance_score ?? 0) * 100
  );

  const collaborativeScore = Math.round(
    Number(course.collaborative_score ?? 0) * 100
  );

  const rating =
    course.course_rating !== null &&
    course.course_rating !== undefined
      ? Number(course.course_rating).toFixed(1)
      : "—";

  /*
   * ----------------------------------------------------------
   * EXPLAINABLE AI REASONS
   * ----------------------------------------------------------
   * These explanations are generated from the actual model
   * component scores returned by the recommendation API.
   */

  const recommendationReasons = [];

  if (skillScore >= 80) {
    recommendationReasons.push(
      "Strong match with your known and target skills."
    );
  } else if (skillScore >= 60) {
    recommendationReasons.push(
      "Good alignment with the skills you're developing."
    );
  }

  if (contentScore >= 80) {
    recommendationReasons.push(
      "Highly relevant to your learning goal and course content."
    );
  } else if (contentScore >= 60) {
    recommendationReasons.push(
      "Course content is relevant to your learning direction."
    );
  }

  if (performanceScore >= 80) {
    recommendationReasons.push(
      "Course difficulty fits your current learning level."
    );
  } else if (performanceScore >= 60) {
    recommendationReasons.push(
      "Course difficulty is reasonably aligned with your current level."
    );
  }

  if (collaborativeScore >= 50) {
    recommendationReasons.push(
      "Your learning activity supports this recommendation."
    );
  }

  if (recommendationReasons.length === 0) {
    recommendationReasons.push(
      "Selected by the hybrid recommendation engine based on your learning profile."
    );
  }

  const visibleReasons = recommendationReasons.slice(0, 3);

  return (
    <article
      className="course-card"
      style={{
        "--card-delay": `${index * 70}ms`,
      }}
    >

      <div className="course-card-top">

        <div className="course-number">
          {String(index + 1).padStart(2, "0")}
        </div>

        <div className="course-score">
          <Sparkles size={13} />
          {score}% match
        </div>

      </div>


      <div className="course-card-content">

        <div className="course-icon">
          <GraduationCap size={21} />
        </div>


        <Link
          to={`/course/${course.course_id}`}
          state={{ course }}
          style={{
            textDecoration: "none",
            color: "inherit",
          }}
        >

          <h3>
            {course.course_name ||
              "Personalized Course"}
          </h3>

        </Link>


        {course.university && (
          <p className="course-university">
            {course.university}
          </p>
        )}


        <div className="course-meta">

          {course.course_rating !== null &&
            course.course_rating !== undefined && (
              <span>

                <Star
                  size={13}
                  fill="currentColor"
                />

                {rating}

              </span>
            )}


          {course.difficulty_level && (
            <span>
              {course.difficulty_level}
            </span>
          )}

        </div>


        <div className="course-match">

          <div className="match-header">

            <span>
              AI Match
            </span>

            <strong>
              {score}%
            </strong>

          </div>


          <div className="match-track">

            <div
              className="match-fill"
              style={{
                width: `${Math.min(
                  score,
                  100
                )}%`,
              }}
            />

          </div>

        </div>


        <div className="course-signals">

          <Signal
            label="Skills"
            value={skillScore}
          />

          <Signal
            label="Content"
            value={contentScore}
          />

          <Signal
            label="Difficulty Fit"
            value={performanceScore}
          />

        </div>


        {/* --------------------------------------------------
            WHY RECOMMENDED?
        -------------------------------------------------- */}

        <details className="course-explanation">

          <summary>
            <Sparkles size={14} />
            Why recommended?
          </summary>

          <div className="course-explanation-content">

            {visibleReasons.map(
              (reason, reasonIndex) => (
                <div
                  className="course-reason"
                  key={`${course.course_id}-reason-${reasonIndex}`}
                >
                  <CheckCircle2 size={14} />
                  <span>
                    {reason}
                  </span>
                </div>
              )
            )}

          </div>

        </details>

      </div>


      <div className="course-card-footer">

        <Link
          to={`/course/${course.course_id}`}
          state={{ course }}
          className="course-button"
        >
          View Course
          <ChevronRight size={16} />
        </Link>

      </div>

    </article>
  );
}

/* ============================================================
   SIGNAL
============================================================ */

function Signal({
  label,
  value,
}) {
  return (
    <div className="course-signal">

      <span>
        {label}
      </span>

      <strong>
        {Math.min(value, 100)}%
      </strong>

    </div>
  );
}


/* ============================================================
   MODEL FACTOR
============================================================ */

function ModelFactor({
  icon,
  title,
  value,
  description,
}) {
  return (
    <article className="model-factor">

      <div className="model-factor-icon">
        {icon}
      </div>

      <div className="model-factor-body">

        <div className="model-factor-heading">

          <h3>
            {title}
          </h3>

          <strong>
            {value}
          </strong>

        </div>

        <p>
          {description}
        </p>

      </div>

    </article>
  );
}


/* ============================================================
   EMPTY STATE
============================================================ */

function EmptyRecommendations() {
  return (
    <div className="recommendations-empty">

      <div className="empty-state-icon">
        <BrainCircuit size={30} />
      </div>

      <h3>
        Your recommendations are almost ready.
      </h3>

      <p>
        Complete or update your learning profile to
        generate a personalized course path.
      </p>

      <Link
        to="/onboarding"
        className="primary-dashboard-button"
      >
        Build My Learning Path
        <ArrowRight size={17} />
      </Link>

    </div>
  );
}


/* ============================================================
   LOADING SKELETON
============================================================ */

function DashboardSkeleton() {
  return (
    <div className="dashboard-skeleton">

      <div className="skeleton skeleton-eyebrow" />

      <div className="skeleton skeleton-title" />

      <div className="skeleton skeleton-description" />


      <div className="skeleton-stats">

        {Array.from({ length: 4 }).map(
          (_, index) => (
            <div
              className="skeleton skeleton-stat"
              key={index}
            />
          )
        )}

      </div>


      <div className="skeleton skeleton-section-title" />


      <div className="skeleton-profile">

        <div className="skeleton skeleton-profile-card" />

        <div className="skeleton skeleton-profile-card" />

      </div>


      <div className="skeleton skeleton-section-title" />

      <CourseGridSkeleton />

    </div>
  );
}


function CourseGridSkeleton() {
  return (
    <div className="course-grid">

      {Array.from({ length: 6 }).map(
        (_, index) => (
          <div
            className="course-skeleton"
            key={index}
          >

            <div className="skeleton skeleton-small" />

            <div className="skeleton skeleton-course-icon" />

            <div className="skeleton skeleton-course-title" />

            <div className="skeleton skeleton-course-line" />

            <div className="skeleton skeleton-course-line short" />

            <div className="skeleton skeleton-course-score" />

          </div>
        )
      )}

    </div>
  );
}


/* ============================================================
   ICON
============================================================ */

function RocketIcon() {
  return <RocketFallback size={17} />;
}


function RocketFallback({ size }) {
  return (
    <span style={{ fontSize: size }}>
      🚀
    </span>
  );
}


export default Dashboard;