import { Link } from "react-router-dom";
import {
  ArrowRight,
  BrainCircuit,
  Sparkles,
  Target,
  TrendingUp,
  BookOpen,
} from "lucide-react";
import AnimatedSection from "../components/AnimatedSection";

function Home() {
  return (
    <main className="home-page">
      <AnimatedSection animation="fade-down" delay={0}>
        <section className="hero-section">
          <div className="hero-glow glow-one"></div>
          <div className="hero-glow glow-two"></div>

          <div className="hero-content">
            <div className="eyebrow">
              <Sparkles size={16} />
              AI-Powered Personalized Learning
            </div>

            <h1>
              Learn smarter.
              <br />
              <span>Grow faster.</span>
            </h1>

            <p className="hero-description">
              Discover learning resources personalized to your skills, goals, performance, and learning behavior using our hybrid machine learning recommendation system.
            </p>

            <div className="hero-actions">
              <Link to="/onboarding" className="primary-button">
                Build My Learning Path
                <ArrowRight size={18} />
              </Link>

              <Link to="/dashboard" className="secondary-button">
                Explore Dashboard
              </Link>
            </div>

            <div className="hero-trust">
              <div>
                <strong>3,400+</strong>
                <span>Learning Resources</span>
              </div>

              <div>
                <strong>AI</strong>
                <span>Hybrid Recommendations</span>
              </div>

              <div>
                <strong>4D</strong>
                <span>Personalization Signals</span>
              </div>
            </div>
          </div>

          <div className="hero-visual">
            <div className="floating-card card-top">
              <div className="mini-icon">
                <Target size={18} />
              </div>

              <div>
                <small>Skill Match</small>
                <strong>94%</strong>
              </div>
            </div>

            <div className="ai-orb">
              <div className="orb-ring ring-one"></div>
              <div className="orb-ring ring-two"></div>

              <div className="orb-core">
                <BrainCircuit size={54} />
              </div>
            </div>

            <div className="floating-card card-bottom">
              <div className="mini-icon">
                <BookOpen size={18} />
              </div>

              <div>
                <small>AI Recommendation</small>
                <strong>Machine Learning</strong>
              </div>

              <TrendingUp size={18} />
            </div>
          </div>
        </section>
      </AnimatedSection>

      <AnimatedSection animation="fade-up" delay={150}>
        <section className="features-section">
          <div className="section-heading">
            <span>HOW IT WORKS</span>

            <h2>
              A learning path built
              <br />
              <span>around you.</span>
            </h2>
          </div>

          <div className="feature-grid">
            <div className="feature-card">
              <div className="feature-number">01</div>

              <div className="feature-icon">
                <Target />
              </div>

              <h3>Tell us your goal</h3>

              <p>
                Define what you want to learn and where you want your skills to take you.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-number">02</div>

              <div className="feature-icon">
                <BrainCircuit />
              </div>

              <h3>AI understands you</h3>

              <p>
                Our hybrid recommendation engine analyzes skills, content, behavior, and performance.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-number">03</div>

              <div className="feature-icon">
                <TrendingUp />
              </div>

              <h3>Keep improving</h3>

              <p>
                Follow personalized recommendations and continuously build the skills you need.
              </p>
            </div>
          </div>
        </section>
      </AnimatedSection>
    </main>
  );
}

export default Home;