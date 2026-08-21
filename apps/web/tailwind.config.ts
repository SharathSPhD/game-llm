import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Dark scientific palette
        slate: {
          50: "#f8fafc",
          100: "#f1f5f9",
          200: "#e2e8f0",
          300: "#cbd5e1",
          400: "#94a3b8",
          500: "#64748b",
          600: "#475569",
          700: "#334155",
          800: "#1e293b",
          900: "#0f172a",
          950: "#020617",
        },
      },
      typography: {
        DEFAULT: {
          css: {
            color: "var(--text)",
            "a": {
              color: "var(--accent)",
              textDecoration: "none",
              "&:hover": {
                textDecoration: "underline",
              },
            },
            "code": {
              color: "var(--text-secondary)",
              backgroundColor: "var(--bg-secondary)",
              padding: "0.125rem 0.375rem",
              borderRadius: "0.25rem",
              fontSize: "0.875em",
            },
            "pre code": {
              color: "inherit",
              backgroundColor: "transparent",
              padding: "0",
            },
            "pre": {
              backgroundColor: "var(--bg-secondary)",
              border: "1px solid var(--border)",
            },
          },
        },
      },
    },
  },
  plugins: [],
  darkMode: ["class", '[data-theme="dark"]'],
};

export default config;
