import { Link, useNavigate } from "react-router-dom";
import {
  ArrowRight,
  BrainCircuit,
  LogIn,
  LogOut,
  UserRound,
} from "lucide-react";

import { useAuth } from "../context/AuthContext";

function Navbar() {
  const navigate = useNavigate();

  const {
    user,
    isAuthenticated,
    loading,
    signOut,
  } = useAuth();

  const handleLogout = async () => {
    try {
      await signOut();
      navigate("/login");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  return (
    <nav className="navbar">
      <Link to="/" className="brand">
        <div className="brand-icon">
          <BrainCircuit size={22} />
        </div>

        <div>
          <span className="brand-name">
            LearnAI
          </span>

          <span className="brand-subtitle">
            Personalized Learning
          </span>
        </div>
      </Link>

      <div className="nav-links">
        <Link to="/">Home</Link>

        {!loading && isAuthenticated && (
          <>
            <Link to="/dashboard">
              Dashboard
            </Link>

            <Link to="/profile">
              Profile
            </Link>
          </>
        )}
      </div>

      {!loading && (
        <>
          {isAuthenticated ? (
            <div className="nav-user-actions">
              <span className="nav-user">
                <UserRound size={15} />

                <span>
                  {user?.email || "Learner"}
                </span>
              </span>

              <button
                type="button"
                className="nav-button nav-logout"
                onClick={handleLogout}
              >
                Logout
                <LogOut size={16} />
              </button>
            </div>
          ) : (
            <div className="nav-auth-actions">
              <Link
                to="/login"
                className="nav-login"
              >
                <LogIn size={16} />
                Sign in
              </Link>

              <Link
                to="/signup"
                className="nav-button"
              >
                Get Started
                <ArrowRight size={16} />
              </Link>
            </div>
          )}
        </>
      )}
    </nav>
  );
}

export default Navbar;