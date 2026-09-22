function ScoreBar({ score, label, showValue = true }) {
  const percentage = Math.min(Math.max(Number(score || 0) * 100, 0), 100);

  return (
    <div className="course-score">
      {label && (
        <div className="score-header">
          <span>{label}</span>
          {showValue && <strong>{percentage.toFixed(1)}%</strong>}
        </div>
      )}
      <div className="score-bar">
        <div style={{ width: `${percentage}%` }}></div>
      </div>
    </div>
  );
}

export default ScoreBar;
