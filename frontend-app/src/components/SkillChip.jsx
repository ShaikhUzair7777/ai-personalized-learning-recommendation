function SkillChip({ name, type = "known" }) {
  return (
    <span className={`skill-tag ${type}-tag`}>
      {name}
    </span>
  );
}

export default SkillChip;
