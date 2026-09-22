import { useState, useEffect, useRef, useCallback } from "react";
import NeuralBrain from "./NeuralBrain";
import { SkipForward } from "lucide-react";

const INTRO_MESSAGES = [
  "Analyzing your learning profile...",
  "Understanding your skills...",
  "Building your personalized path...",
  "Your learning path is ready.",
];

const STAGE_DURATION = 700; // ms per stage
const TOTAL_STAGES = INTRO_MESSAGES.length;
const SESSION_KEY = "learnai_intro_seen";

/**
 * Cinematic dashboard opening animation.
 * Shows a neural brain forming with cycling status text.
 * Skips automatically on repeat visits in the same session.
 * Respects prefers-reduced-motion.
 */
function DashboardIntro({ onComplete }) {
  const [visible, setVisible] = useState(false);
  const [stage, setStage] = useState(0);
  const [fading, setFading] = useState(false);
  const timerRef = useRef(null);

  // Check if we should show intro
  useEffect(() => {
    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    const alreadySeen = sessionStorage.getItem(SESSION_KEY);

    if (prefersReducedMotion || alreadySeen) {
      onComplete();
      return;
    }

    setVisible(true);
    sessionStorage.setItem(SESSION_KEY, "true");
  }, [onComplete]);

  // Progress through stages
  useEffect(() => {
    if (!visible) return;

    timerRef.current = setInterval(() => {
      setStage((prev) => {
        if (prev >= TOTAL_STAGES - 1) {
          clearInterval(timerRef.current);
          // Start fade-out after last message
          setTimeout(() => {
            setFading(true);
            setTimeout(() => {
              onComplete();
            }, 600);
          }, 800);
          return prev;
        }
        return prev + 1;
      });
    }, STAGE_DURATION);

    return () => clearInterval(timerRef.current);
  }, [visible, onComplete]);

  const skip = useCallback(() => {
    clearInterval(timerRef.current);
    setFading(true);
    setTimeout(() => {
      onComplete();
    }, 300);
  }, [onComplete]);

  if (!visible) return null;

  return (
    <div
      className="dashboard-intro-overlay"
      style={{
        opacity: fading ? 0 : 1,
        transition: "opacity 0.6s ease",
      }}
      role="status"
      aria-live="polite"
    >
      <NeuralBrain
        size={Math.min(window.innerWidth * 0.8, 500)}
        intensity={0.3 + (stage / TOTAL_STAGES) * 0.7}
        animate={true}
        className="hero-brain-canvas"
      />

      <div className="intro-text-container">
        <div className="intro-status-text" key={stage}>
          {INTRO_MESSAGES[stage]}
        </div>
      </div>

      <div className="intro-progress">
        {INTRO_MESSAGES.map((_, i) => (
          <div
            key={i}
            className={`intro-progress-dot ${i <= stage ? "active" : ""}`}
          />
        ))}
      </div>

      <button
        className="intro-skip-button"
        onClick={skip}
        aria-label="Skip introduction animation"
      >
        <SkipForward size={14} />
        Skip intro
      </button>
    </div>
  );
}

export default DashboardIntro;
