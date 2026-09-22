import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowRight,
  BarChart3,
  BookOpen,
  BrainCircuit,
  Target,
  UserRound,
} from "lucide-react";

import AnimatedSection from "../components/AnimatedSection";
import EmptyState from "../components/EmptyState";
import SkillChip from "../components/SkillChip";

function Profile() {
  const [profile, setProfile] = useState(null);

  useEffect(() => {
    const savedProfile = sessionStorage.getItem("learningProfile");
    if (savedProfile) {
      setProfile(JSON.parse(savedProfile));
    }
  }, []);

  if (!profile) {
    return (
      <main className="profile-page">
        <AnimatedSection animation="scale-up">
          <EmptyState
            icon={<UserRound size={52} />}
            title="No learning profile saved"
            description="Your learning profile is stored in the current session after onboarding."
            ctaText="Build My Learning Profile"
            ctaLink="/onboarding"
          />
        </AnimatedSection>
      </main>
    );
  }

  const knownSkills = profile.known_skills
    ?.split(",")
    .map((skill) => skill.trim())
    .filter(Boolean) || [];

  const targetSkills = profile.target_skills
    ?.split(",")
    .map((skill) => skill.trim())
    .filter(Boolean) || [];

  return (
    <main className="profile-page">
      <AnimatedSection animation="fade-down" delay={0}>
        <section className="page-hero profile-hero">
          <div className="eyebrow">
            <UserRound size={14} />
            LEARNER PROFILE
          </div>

          <h1>
            Your learning profile,
            <span> in context.</span>
          </h1>
        </section>
      </AnimatedSection>

      <AnimatedSection animation="fade-up" delay={100}>
        <section className="profile-grid">
          <article className="glass-card profile-card main-profile-card">
            <div className="panel-header">
              <div className="panel-icon violet">
                <Target size={18} />
              </div>
              <div>
                <span className="panel-kicker">Learning Profile</span>
                <h2>Goals and readiness</h2>
              </div>
            </div>

            <div className="profile-field">
              <span className="field-label">Learning goal</span>
              <p className="field-value">{profile.goal || "Not specified yet"}</p>
            </div>

            <div className="profile-field">
              <span className="field-label">Performance score</span>
              <p className="field-value">{profile.performance_score ?? "Not available"}/100</p>
            </div>

            <div className="profile-field">
              <span className="field-label">Recommendation preference</span>
              <p className="field-value">Top {profile.top_k ?? 10} recommendations</p>
            </div>
          </article>

          <article className="glass-card profile-card">
            <div className="panel-header">
              <div className="panel-icon cyan">
                <BrainCircuit size={18} />
              </div>
              <div>
                <span className="panel-kicker">Recommendation Engine</span>
                <h2>How your path is built</h2>
              </div>
            </div>

            <div className="weights-list">
              <div className="weight-row">
                <span>Skill</span>
                <strong>40%</strong>
              </div>
              <div className="weight-row">
                <span>Content</span>
                <strong>30%</strong>
              </div>
              <div className="weight-row">
                <span>Collaborative</span>
                <strong>15%</strong>
              </div>
              <div className="weight-row">
                <span>Performance</span>
                <strong>15%</strong>
              </div>
            </div>
          </article>
        </section>
      </AnimatedSection>

      <AnimatedSection animation="fade-up" delay={200}>
        <section className="profile-grid two-up">
          <article className="glass-card profile-card">
            <div className="panel-header">
              <div className="panel-icon purple">
                <BookOpen size={18} />
              </div>
              <div>
                <span className="panel-kicker">Skill Development</span>
                <h2>Known skills</h2>
              </div>
            </div>

            <div className="skill-cloud">
              {knownSkills.length ? (
                knownSkills.map((skill, index) => (
                  <SkillChip key={`${skill}-${index}`} name={skill} type="known" />
                ))
              ) : (
                <span className="muted-text">No known skills added yet.</span>
              )}
            </div>
          </article>

          <article className="glass-card profile-card">
            <div className="panel-header">
              <div className="panel-icon blue">
                <BarChart3 size={18} />
              </div>
              <div>
                <span className="panel-kicker">Skill Development</span>
                <h2>Target skills</h2>
              </div>
            </div>

            <div className="skill-cloud">
              {targetSkills.length ? (
                targetSkills.map((skill, index) => (
                  <SkillChip key={`${skill}-${index}`} name={skill} type="target" />
                ))
              ) : (
                <span className="muted-text">No target skills added yet.</span>
              )}
            </div>
          </article>
        </section>
      </AnimatedSection>

      <div className="profile-actions">
        <Link to="/onboarding" className="primary-button profile-button">
          Update Profile
          <ArrowRight size={17} />
        </Link>
      </div>
    </main>
  );
}

export default Profile;