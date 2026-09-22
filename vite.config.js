import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],

  build: {
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            {
              name: "react-vendor",
              test: /node_modules[\\/](react|react-dom|scheduler)[\\/]/
            },
            {
              name: "router-vendor",
              test: /node_modules[\\/](react-router|react-router-dom)[\\/]/
            },
            {
              name: "supabase-vendor",
              test: /node_modules[\\/]@supabase[\\/]/
            },
            {
              name: "http-vendor",
              test: /node_modules[\\/]axios[\\/]/
            },
            {
              name: "icons-vendor",
              test: /node_modules[\\/]lucide-react[\\/]/
            }
          ]
        }
      }
    }
  }
});
