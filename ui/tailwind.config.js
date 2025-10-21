/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        corpus: {
          blue: "#3056d3",
          teal: "#1fb6aa",
          slate: "#1f2937",
        },
      },
    },
  },
  plugins: [],
};
