import axios from "axios";
import { supabase } from "../lib/supabase";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

api.interceptors.request.use(
  async (config) => {
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (session?.access_token) {
        config.headers.Authorization = `Bearer ${session.access_token}`;
      }
    } catch (error) {
      console.error("Failed to get Supabase session:", error);
    }

    return config;
  },
  (error) => Promise.reject(error)
);

export const checkHealth = async () => {
  const response = await api.get("/api/health");
  return response.data;
};

export const getRecommendations = async () => {
  const response = await api.get("/api/recommendations");
  return response.data;
};

export const getCustomRecommendations = async (profile) => {
  const response = await api.post(
    "/api/recommendations/custom",
    profile
  );
  return response.data;
};

export default api;