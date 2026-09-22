import { BrainCircuit, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";

function EmptyState({
  icon,
  title = "No data available",
  description = "Complete your learning profile to get started.",
  ctaText = "Build My Learning Path",
  ctaTo,
  ctaLink,
}) {
  const destination = ctaTo || ctaLink || "/onboarding";

  return (
    <div className="empty-state-card">
      {icon || <BrainCircuit size={52} />}
      <h1>{title}</h1>
      <p>{description}</p>
      {destination && (
        <Link to={destination} className="primary-button">
          {ctaText}
          <ArrowRight size={17} />
        </Link>
      )}
    </div>
  );
}

export default EmptyState;
