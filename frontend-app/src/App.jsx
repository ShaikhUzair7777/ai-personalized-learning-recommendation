import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import Navbar from "./components/Navbar";
import ProtectedRoute from "./components/ProtectedRoute";

// ============================================================
// LAZY-LOADED ROUTES
// Each page becomes its own JavaScript chunk.
// ============================================================

const Home = lazy(() => import("./pages/Home"));
const Login = lazy(() => import("./pages/Login"));
const Signup = lazy(() => import("./pages/Signup"));
const Onboarding = lazy(() => import("./pages/Onboarding"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const CourseDetails = lazy(() => import("./pages/CourseDetails"));
const Profile = lazy(() => import("./pages/Profile"));

// ============================================================
// ROUTE LOADING FALLBACK
// Lightweight fallback so Suspense does not pull in another
// component or dependency.
// ============================================================

function RouteLoading() {
  return (
    <div
      style={{
        minHeight: "calc(100vh - 80px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "32px",
        background: "#080a14",
        color: "#a78bfa",
        fontSize: "14px",
        fontWeight: 600,
      }}
      aria-live="polite"
      aria-busy="true"
    >
      Loading...
    </div>
  );
}

// ============================================================
// APP
// ============================================================

function App() {
  return (
    <BrowserRouter>
      <Navbar />

      <Suspense fallback={<RouteLoading />}>
        <Routes>
          {/* ==================================================
              PUBLIC ROUTES
          ================================================== */}

          <Route path="/" element={<Home />} />

          <Route path="/login" element={<Login />} />

          <Route path="/signup" element={<Signup />} />

          {/* ==================================================
              PROTECTED APPLICATION ROUTES
          ================================================== */}

          <Route element={<ProtectedRoute />}>
            <Route
              path="/onboarding"
              element={<Onboarding />}
            />

            <Route
              path="/dashboard"
              element={<Dashboard />}
            />

            <Route
              path="/course/:courseId"
              element={<CourseDetails />}
            />

            <Route
              path="/profile"
              element={<Profile />}
            />
          </Route>
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

export default App;