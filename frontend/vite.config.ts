import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The API's CORS allowlist defaults to exactly this origin.
export default defineConfig({
  plugins: [react()],
  server: { host: "localhost", port: 5173, strictPort: true },
});
