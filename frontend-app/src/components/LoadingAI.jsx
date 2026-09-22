import { BrainCircuit } from "lucide-react";
import { useEffect, useState } from "react";

const MESSAGES = [
  "Analyzing your learning profile",
  "Matching your skills",
  "Evaluating course relevance",
  "Building your personalized path",
];

function LoadingAI() {
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setMessageIndex((prev) =>
        prev < MESSAGES.length - 1 ? prev + 1 : prev
      );
    }, 1200);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="loading-ai-overlay" role="status" aria-live="polite">
      <div className="loading-ai-brain">
        <BrainCircuit size={36} />
      </div>

      <div className="loading-ai-text">
        {MESSAGES[messageIndex]}
      </div>

      <div className="loading-dots">
        <span></span>
        <span></span>
        <span></span>
      </div>

      <div className="loading-ai-sub">
        This may take a moment
      </div>
    </div>
  );
}

export default LoadingAI;
