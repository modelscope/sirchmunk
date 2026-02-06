/** @type {import('next').NextConfig} */

// When NEXT_BUILD_STATIC=true, produce a fully static export (HTML/CSS/JS)
// that can be served by any static file server (including FastAPI StaticFiles).
// This is used by `sirchmunk init --ui` and `sirchmunk serve --ui`.
const isStaticExport = process.env.NEXT_BUILD_STATIC === "true";

const nextConfig = {
  // Static export mode: generates `out/` directory with pure static files
  ...(isStaticExport && {
    output: "export",
    images: {
      unoptimized: true, // Image Optimization requires a server; disable for static
    },
  }),

  // Move dev indicator to bottom-right corner
  devIndicators: {
    position: "bottom-right",
  },

  // Transpile mermaid and related packages for proper ESM handling
  transpilePackages: ["mermaid"],

  // Turbopack configuration (Next.js 16+ uses Turbopack by default for dev)
  turbopack: {
    resolveAlias: {
      // Fix for mermaid's cytoscape dependency - use CJS version
      cytoscape: "cytoscape/dist/cytoscape.cjs.js",
    },
  },

  // Webpack configuration (used for production builds - next build)
  webpack: (config) => {
    const path = require("path");
    config.resolve.alias = {
      ...config.resolve.alias,
      cytoscape: path.resolve(
        __dirname,
        "node_modules/cytoscape/dist/cytoscape.cjs.js",
      ),
    };
    return config;
  },
};

module.exports = nextConfig;
