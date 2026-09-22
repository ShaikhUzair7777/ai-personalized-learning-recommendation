import { AlertCircle, RefreshCw } from "lucide-react";
import { Link } from "react-router-dom";

function ErrorState({
  title = "Something went wrong",
  message = "We were unable to process your request. Please ensure the FastAPI backend is running.",
  onRetry,
  actionText = "Try Again",
  backLink = "/onboarding",
}) {
  return (
    <div className="error-state-card" style={{
      background: "rgba(239, 68, 68, 0.05)",
      border: "1px solid rgba(239, 68, 68, 0.2)",
      borderRadius: "16px",
      padding: "2.5rem",
      textAlign: "center",
      maxWidth: "500px",
      margin: "2rem auto"
    }}>
      <div style={{
        width: "56px",
        height: "56px",
        borderRadius: "50%",
        background: "rgba(239, 68, 68, 0.15)",
        color: "#f87171",
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        marginBottom: "1rem"
      }}>
        <AlertCircle size={28} />
      </div>
      <h3 style={{ fontSize: "1.25rem", color: "var(--text-primary)", marginBottom: "0.5rem" }}>{title}</h3>
      <p style={{ color: "var(--text-secondary)", fontSize: "0.925rem", marginBottom: "1.5rem", lineHeight: 1.5 }}>{message}</p>
      
      <div style={{ display: "flex", gap: "1rem", justifyContent: "center" }}>
        {onRetry ? (
          <button onClick={onRetry} className="primary-button" style={{ background: "linear-gradient(135deg, #ef4444, #dc2626)" }}>
            <RefreshCw size={16} />
            {actionText}
          </button>
        ) : backLink ? (
          <Link to={backLink} className="primary-button">
            {actionText}
          </Link>
        ) : null}
      </div>
    </div>
  );
}

export default ErrorState;
