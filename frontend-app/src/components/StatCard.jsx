function StatCard({ icon, label, value, color = "purple" }) {
  return (
    <div className="dashboard-stat-card">
      <div className={`stat-icon ${color}`}>{icon}</div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

export default StatCard;
