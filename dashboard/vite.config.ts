import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vitest/config";

// Relative base: the same build works at /qiskit-qward/dashboard/ on GitHub Pages and locally.
export default defineConfig({
  base: "./",
  plugins: [svelte()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "es2022",
    chunkSizeWarningLimit: 900,
  },
  test: {
    include: ["tests/unit/**/*.test.ts"],
    environment: "node",
  },
});
