import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  ExternalLink,
  BrainCircuit,
  BookOpen,
  Star,
  Target,
  Layers3,
  Users,
  BarChart3,
  Sparkles,
  Bookmark,
  Check,
  AlertCircle,
  X,
  LoaderCircle,
  ShieldCheck,
} from "lucide-react";

import AnimatedSection from "../components/AnimatedSection";
import EmptyState from "../components/EmptyState";
import ScoreRing from "../components/ScoreRing";
import ScoreBar from "../components/ScoreBar";

import { supabase } from "../lib/supabase";
import { useAuth } from "../context/AuthContext";

/**
 * Format score value (0 to 1) into percentage string (e.g. 0.7746 -> 77.5%, 1.0 -> 100%, 0 -> 0%).
 */
function formatPercent(val) {
  const num = Number(val || 0) * 100;
  return num % 1 === 0 ? `${num.toFixed(0)}%` : `${num.toFixed(1)}%`;
}

function CourseDetails() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user } = useAuth();

  const course = location.state?.course;

  // Track viewed interaction strictly once per course mount
  const viewedTrackedCourseId = useRef(null);

  const [saved, setSaved] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [trackingStatus, setTrackingStatus] = useState("");
  const [trackingError, setTrackingError] = useState("");

  // Auto-dismiss status messages after 3.5s
  useEffect(() => {
    if (!trackingStatus) return;
    const timer = setTimeout(() => {
      setTrackingStatus("");
    }, 3500);
    return () => clearTimeout(timer);
  }, [trackingStatus]);

  // Auto-dismiss non-critical error messages after 6s
  useEffect(() => {
    if (!trackingError) return;
    const timer = setTimeout(() => {
      setTrackingError("");
    }, 6000);
    return () => clearTimeout(timer);
  }, [trackingError]);

  /*
   * ---------------------------------------------------------
   * CHECK IF COURSE WAS PREVIOUSLY SAVED
   * ---------------------------------------------------------
   */
  useEffect(() => {
    let isMounted = true;
    if (!course?.course_id || !user?.id) return;

    const checkSavedStatus = async () => {
      try {
        const { data, error } = await supabase
          .from("course_interactions")
          .select("id")
          .eq("user_id", user.id)
          .eq("course_id", String(course.course_id))
          .eq("interaction_type", "saved")
          .limit(1);

        if (!error && data && data.length > 0 && isMounted) {
          setSaved(true);
        }
      } catch (err) {
        console.warn("Could not check saved status:", err);
      }
    };

    checkSavedStatus();

    return () => {
      isMounted = false;
    };
  }, [course?.course_id, user?.id]);

  /*
   * ---------------------------------------------------------
   * TRACK COURSE INTERACTION
   * ---------------------------------------------------------
   *
   * Supported interaction types:
   * viewed, clicked, started, completed, saved
   */
  const trackInteraction = useCallback(
    async (interactionType) => {
      if (!user?.id) {
        console.warn("Course interaction not recorded: no authenticated user.");
        setTrackingError("Sign in to track your learning activity.");
        return false;
      }

      if (!course?.course_id) {
        console.warn("Course interaction not recorded: course_id is missing.", course);
        setTrackingError("Course information is incomplete.");
        return false;
      }

      try {
        console.log(`Tracking "${interactionType}" interaction:`, {
          user_id: user.id,
          course_id: String(course.course_id),
          interaction_type: interactionType,
          interaction_value: 1,
        });

        const { error } = await supabase.from("course_interactions").insert({
          user_id: user.id,
          course_id: String(course.course_id),
          interaction_type: interactionType,
          interaction_value: 1,
        });

        if (error) {
          console.error(`Supabase error recording ${interactionType}:`, error);
          setTrackingError(`Unable to save ${interactionType} activity.`);
          return false;
        }

        console.log(`Interaction "${interactionType}" successfully recorded.`);
        setTrackingStatus(
          `${interactionType.charAt(0).toUpperCase()}${interactionType.slice(1)} recorded`
        );
        setTrackingError("");
        return true;
      } catch (err) {
        console.error(`Unexpected error recording ${interactionType}:`, err);
        setTrackingError(`Could not record ${interactionType} activity.`);
        return false;
      }
    },
    [user, course]
  );

  /*
   * ---------------------------------------------------------
   * TRACK COURSE VIEW
   * ---------------------------------------------------------
   * Runs once when this course details page opens for the current course.
   */
  useEffect(() => {
    if (!course?.course_id || !user?.id) {
      return;
    }

    const currentCourseId = String(course.course_id);
    if (viewedTrackedCourseId.current === currentCourseId) {
      return;
    }

    viewedTrackedCourseId.current = currentCourseId;
    trackInteraction("viewed");
  }, [course?.course_id, user?.id, trackInteraction]);

  /*
   * ---------------------------------------------------------
   * COURSE NOT FOUND / EMPTY STATE
   * ---------------------------------------------------------
   */
  if (!course) {
    return (
      <main className="course-details-page">
        <button
          type="button"
          className="details-back"
          onClick={() => navigate("/dashboard")}
          aria-label="Back to Dashboard"
        >
          <ArrowLeft size={16} />
          <span>Back to Dashboard</span>
        </button>

        <AnimatedSection animation="reveal-scale">
          <EmptyState
            title="Course not found"
            description="This course was not available in the current recommendation session."
            ctaText="Back to Dashboard"
            ctaTo="/dashboard"
          />
        </AnimatedSection>
      </main>
    );
  }

  /*
   * ---------------------------------------------------------
   * SCORE VALUES
   * ---------------------------------------------------------
   */
  const finalScore = Number(course.final_score || 0);
  const skillScore = Number(course.skill_score || 0);
  const contentScore = Number(course.content_score || 0);
  const collaborativeScore = Number(course.collaborative_score || 0);
  const performanceScore = Number(course.performance_score || 0);

  /*
   * ---------------------------------------------------------
   * START LEARNING
   * ---------------------------------------------------------
   */
  const handleStartLearning = async () => {
    setTrackingStatus("");
    setTrackingError("");

    if (!course.course_url) {
      setTrackingError("Course link is currently unavailable for this course.");
      return;
    }

    try {
      // 1. Record clicked
      await trackInteraction("clicked");
      // 2. Record started
      await trackInteraction("started");
    } catch (err) {
      console.error("Error during interaction tracking:", err);
    }

    // 3. Open original course URL in new browser tab
    window.open(course.course_url, "_blank", "noopener,noreferrer");
  };

  /*
   * ---------------------------------------------------------
   * SAVE COURSE
   * ---------------------------------------------------------
   */
  const handleSaveCourse = async () => {
    if (!user?.id) {
      setTrackingError("Please log in before saving a course.");
      return;
    }

    if (saved || isSaving) {
      return;
    }

    setIsSaving(true);
    setTrackingError("");

    try {
      const success = await trackInteraction("saved");
      if (success) {
        setSaved(true);
      }
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <main className="course-details-page">
      {/* BACK BUTTON */}
      <button
        type="button"
        className="details-back"
        onClick={() => navigate("/dashboard")}
        aria-label="Back to recommendations"
      >
        <ArrowLeft size={16} />
        <span>Back to recommendations</span>
      </button>

      {/* TRACKING STATUS & NOTIFICATIONS */}
      {(trackingStatus || trackingError) && (
        <div
          className={`tracking-banner ${trackingError ? "error" : "status"}`}
          role="status"
          aria-live="polite"
        >
          <div className="tracking-banner-content">
            {trackingError ? (
              <AlertCircle size={16} className="tracking-banner-icon" />
            ) : (
              <ShieldCheck size={16} className="tracking-banner-icon" />
            )}
            <span>{trackingError || trackingStatus}</span>
          </div>

          <button
            type="button"
            className="tracking-banner-close"
            onClick={() => {
              setTrackingStatus("");
              setTrackingError("");
            }}
            aria-label="Dismiss notification"
          >
            <X size={14} />
          </button>
        </div>
      )}

      <AnimatedSection animation="reveal" delay={0}>
        <section className="course-details-layout">
          {/* ================================================= */}
          {/* LEFT CONTENT */}
          {/* ================================================= */}
          <div className="course-details-main">
            {/* COURSE ICON */}
            <div className="course-details-icon" aria-hidden="true">
              <BookOpen size={36} />
            </div>

            {/* LABEL */}
            <div className="course-details-label">
              AI RECOMMENDED COURSE
            </div>

            {/* TITLE */}
            <h1>{course.course_name}</h1>

            {/* UNIVERSITY */}
            <p className="course-details-university">
              {course.university || "Learning Resource"}
            </p>

            {/* ================================================= */}
            {/* META */}
            {/* ================================================= */}
            <div className="details-meta-row">
              {/* RATING */}
              <div className="details-meta">
                <Star size={16} fill="currentColor" aria-hidden="true" />
                <div>
                  <span>Rating</span>
                  <strong>
                    {course.course_rating !== null &&
                    course.course_rating !== undefined
                      ? Number(course.course_rating).toFixed(1)
                      : "N/A"}
                  </strong>
                </div>
              </div>

              {/* DIFFICULTY */}
              <div className="details-meta">
                <Layers3 size={16} aria-hidden="true" />
                <div>
                  <span>Difficulty</span>
                  <strong>
                    {course.difficulty_level || "All Levels"}
                  </strong>
                </div>
              </div>

              {/* AI MATCH */}
              <div className="details-meta">
                <Sparkles size={16} aria-hidden="true" />
                <div>
                  <span>AI Match</span>
                  <strong>{formatPercent(finalScore)}</strong>
                </div>
              </div>
            </div>

            {/* ================================================= */}
            {/* DESCRIPTION */}
            {/* ================================================= */}
            <div className="details-description">
              <div className="details-section-title">
                <Sparkles size={17} aria-hidden="true" />
                <span>Why this course?</span>
              </div>
              <p>
                This course was selected by the hybrid recommendation engine
                based on your learner profile, target skills, content
                relevance, collaborative signals, and performance compatibility.
              </p>
            </div>

            {/* ================================================= */}
            {/* SCORE BREAKDOWN */}
            {/* ================================================= */}
            <div className="score-breakdown">
              <div className="details-section-title">
                <BrainCircuit size={17} aria-hidden="true" />
                <span>Recommendation Breakdown</span>
              </div>

              <div className="breakdown-grid">
                <ScoreItem
                  icon={<Target size={17} />}
                  label="Skill Relevance"
                  value={skillScore}
                />

                <ScoreItem
                  icon={<BookOpen size={17} />}
                  label="Content Similarity"
                  value={contentScore}
                />

                <ScoreItem
                  icon={<Users size={17} />}
                  label="Collaborative Signal"
                  value={collaborativeScore}
                />

                <ScoreItem
                  icon={<BarChart3 size={17} />}
                  label="Performance Fit"
                  value={performanceScore}
                />
              </div>
            </div>
          </div>

          {/* ================================================= */}
          {/* RIGHT SIDEBAR */}
          {/* ================================================= */}
          <aside className="course-details-sidebar">
            {/* AI MATCH CARD */}
            <div className="ai-match-card">
              <div className="ai-match-ring-wrapper">
                <ScoreRing score={finalScore} size={72} strokeWidth={7} />
              </div>

              <span>PERSONALIZATION SCORE</span>

              <strong>{formatPercent(finalScore)}</strong>

              <ScoreBar score={finalScore} showValue={false} />

              <p>Based on your learning profile.</p>
            </div>

            {/* COURSE ACTION CARD */}
            <div className="course-action-card">
              <h3>Ready to learn?</h3>
              <p>
                Continue to the original course platform.
              </p>

              <div className="course-action-buttons">
                {/* START LEARNING */}
                {course.course_url ? (
                  <button
                    type="button"
                    className="start-learning-button"
                    onClick={handleStartLearning}
                    aria-label={`Start learning ${course.course_name}`}
                  >
                    <span>Start Learning</span>
                    <ExternalLink size={16} aria-hidden="true" />
                  </button>
                ) : (
                  <button
                    type="button"
                    className="start-learning-button disabled"
                    disabled
                    aria-disabled="true"
                  >
                    <span>Course Link Unavailable</span>
                  </button>
                )}

                {/* SAVE COURSE */}
                <button
                  type="button"
                  className={`save-course-button ${saved ? "saved" : ""}`}
                  onClick={handleSaveCourse}
                  disabled={saved || isSaving}
                  aria-label={saved ? "Course Saved" : "Save Course"}
                >
                  {saved ? (
                    <>
                      <Check size={16} aria-hidden="true" />
                      <span>Course Saved</span>
                    </>
                  ) : isSaving ? (
                    <>
                      <LoaderCircle size={16} className="spin" aria-hidden="true" />
                      <span>Saving Course...</span>
                    </>
                  ) : (
                    <>
                      <Bookmark size={16} aria-hidden="true" />
                      <span>Save Course</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* MODEL CARD */}
            <div className="model-card">
              <div className="model-card-header">
                <BrainCircuit size={17} aria-hidden="true" />
                <span>HYBRID ML MODEL</span>
              </div>

              <div className="model-weight-row">
                <span>Skill</span>
                <strong>40%</strong>
              </div>

              <div className="model-weight-row">
                <span>Content</span>
                <strong>30%</strong>
              </div>

              <div className="model-weight-row">
                <span>Collaborative</span>
                <strong>15%</strong>
              </div>

              <div className="model-weight-row">
                <span>Performance</span>
                <strong>15%</strong>
              </div>
            </div>
          </aside>
        </section>
      </AnimatedSection>
    </main>
  );
}

/*
 * ============================================================
 * SCORE ITEM
 * ============================================================
 */
function ScoreItem({ icon, label, value }) {
  const percentage = Math.min(Math.max(Number(value || 0) * 100, 0), 100);

  return (
    <div className="score-item">
      <div className="score-item-top">
        <div className="score-item-label">
          <span className="score-item-icon" aria-hidden="true">
            {icon}
          </span>
          <span>{label}</span>
        </div>

        <strong>{formatPercent(value)}</strong>
      </div>

      <div
        className="score-item-bar"
        role="progressbar"
        aria-valuenow={Math.round(percentage)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          style={{
            width: `${percentage}%`,
          }}
        />
      </div>
    </div>
  );
}

export default CourseDetails;