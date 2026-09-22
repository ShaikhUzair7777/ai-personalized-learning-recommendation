function ScoreRing({ score, size = 44, strokeWidth = 4 }) {
  const percentage = Math.min(Math.max(Number(score || 0) * 100, 0), 100);
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  const fontSize =
    size >= 70 ? "1rem" : size >= 56 ? "0.85rem" : "0.7rem";

  return (
    <div
      className="score-ring-wrapper"
      style={{
        width: size,
        height: size,
        position: "relative",
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={strokeWidth}
          fill="transparent"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="url(#scoreGrad)"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          fill="transparent"
          style={{ transition: "stroke-dashoffset 1s ease" }}
        />
        <defs>
          <linearGradient id="scoreGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="var(--accent-cyan, #00f2fe)" />
            <stop offset="100%" stopColor="var(--accent-purple, #7928ca)" />
          </linearGradient>
        </defs>
      </svg>
      <span
        style={{
          position: "absolute",
          fontSize,
          fontWeight: 700,
          color: "var(--text-primary)",
        }}
      >
        {Math.round(percentage)}%
      </span>
    </div>
  );
}

export default ScoreRing;
