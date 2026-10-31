import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The build is served by `fliptuned serve` from src/fliptuned/web_dist/. Every asset is bundled
// (no CDNs, web fonts or analytics) and nothing is inlined into index.html, so the page works under
// the server's Content-Security-Policy `default-src 'self'`.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../src/fliptuned/web_dist",
    emptyOutDir: true,
    assetsInlineLimit: 0,
    modulePreload: { polyfill: false },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
