// API configuration and utility functions

// Get API base URL from environment variable.
// Priority:
//   1. NEXT_PUBLIC_API_BASE explicitly set (e.g. "http://localhost:8584")
//   2. Empty string → same-origin mode (static export served by FastAPI)
//   3. Fallback: auto-detect from window.location (runtime)
//
// When built with `sirchmunk web init`, NEXT_PUBLIC_API_BASE is set to ""
// so all API calls are relative to the current origin (single-port mode).
export const API_BASE_URL: string = (() => {
  // Env var is embedded at build time by Next.js
  const envBase = process.env.NEXT_PUBLIC_API_BASE;

  // If explicitly set (even to empty string), use it directly.
  // Empty string means same-origin (relative URLs).
  if (envBase !== undefined) {
    return envBase;
  }

  // Not set at all — likely local dev without env configuration.
  // Fall back to same-origin so relative API calls still work.
  if (typeof window !== "undefined") {
    console.warn(
      "NEXT_PUBLIC_API_BASE is not set. Using same-origin mode.",
      "Configure it in .env.local or via start_web.py for cross-origin setups.",
    );
  }
  return "";
})();

/**
 * Construct a full API URL from a path
 * @param path - API path (e.g., '/api/v1/knowledge/list')
 * @returns Full URL (e.g., 'http://localhost:8000/api/v1/knowledge/list')
 */
export function apiUrl(path: string): string {
  // Remove leading slash if present to avoid double slashes
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  // Remove trailing slash from base URL if present
  const base = API_BASE_URL.endsWith("/")
    ? API_BASE_URL.slice(0, -1)
    : API_BASE_URL;

  return `${base}${normalizedPath}`;
}

/**
 * Construct a WebSocket URL from a path
 * @param path - WebSocket path (e.g., '/api/v1/solve')
 * @returns WebSocket URL (e.g., 'ws://localhost:{backend_port}/api/v1/solve')
 * Note: backend_port is configured in config/main.yaml
 */
export function wsUrl(path: string): string {
  // If API_BASE_URL is empty (same-origin), derive ws URL from window.location
  let base: string;
  if (!API_BASE_URL && typeof window !== "undefined") {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    base = `${proto}//${window.location.host}`;
  } else {
    // Security Hardening: Convert http to ws and https to wss.
    base = API_BASE_URL.replace(/^http:/, "ws:").replace(/^https:/, "wss:");
  }

  // Remove leading slash if present to avoid double slashes
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  // Remove trailing slash from base URL if present
  const normalizedBase = base.endsWith("/") ? base.slice(0, -1) : base;

  return `${normalizedBase}${normalizedPath}`;
}
