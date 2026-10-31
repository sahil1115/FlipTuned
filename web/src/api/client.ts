// API access: the per-launch token, the typed client generated from OpenAPI, and errors in the
// server's shape (message → what to do → details).
import createClient from "openapi-fetch";

import type { components, paths } from "./schema";

export type Schemas = components["schemas"];

export const TOKEN_KEY = "fliptuned-token";
export const TOKEN_HEADER = "X-FlipTuned-Token";

/** Move a `#token=...` fragment into sessionStorage and out of the address bar. */
export function captureToken(win: Window = window): string | null {
  const match = win.location.hash.match(/token=([^&]+)/);
  if (match?.[1]) {
    win.sessionStorage.setItem(TOKEN_KEY, decodeURIComponent(match[1]));
    win.history.replaceState(null, "", `${win.location.pathname}#/`);
  }
  return win.sessionStorage.getItem(TOKEN_KEY);
}

export function getToken(): string | null {
  return window.sessionStorage.getItem(TOKEN_KEY);
}

export class ApiError extends Error {
  readonly status: number;
  readonly hint: string | null;
  readonly details: string | null;

  constructor(status: number, message: string, hint: string | null = null, details: string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.hint = hint;
    this.details = details;
  }
}

interface ErrorBody {
  error?: { message?: string; hint?: string | null; details?: string | null };
}

export function toApiError(status: number, body: unknown): ApiError {
  const error = (body as ErrorBody | null)?.error;
  if (status === 401) {
    return new ApiError(
      401,
      error?.message ?? "Missing or wrong access token.",
      "Open the link printed by `fliptuned serve` (each launch has a new token).",
    );
  }
  if (error?.message) {
    return new ApiError(status, error.message, error.hint ?? null, error.details ?? null);
  }
  return new ApiError(status, `The server answered with status ${status}.`, "Try again, or check the server console.");
}

// Same origin as the page (an absolute base also works where relative URLs can't be parsed).
export const api = createClient<paths>({
  baseUrl: typeof window === "undefined" ? "" : window.location.origin,
  fetch: (request) => globalThis.fetch(request), // looked up per call, not captured at import
});
api.use({
  onRequest({ request }) {
    const token = getToken();
    if (token) request.headers.set(TOKEN_HEADER, token);
    return request;
  },
});

/** The data of an openapi-fetch call, or an ApiError. */
export async function unwrap<T>(call: Promise<{ data?: T; error?: unknown; response: Response }>): Promise<T> {
  let result: { data?: T; error?: unknown; response: Response };
  try {
    result = await call;
  } catch {
    throw new ApiError(0, "Can't reach the FlipTuned server.", "Check that `fliptuned serve` is still running.");
  }
  if (!result.response.ok || result.error !== undefined) {
    throw toApiError(result.response.status, result.error);
  }
  return result.data as T;
}

/** fetch() with the token, for uploads, downloads and event streams. */
export async function rawFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set(TOKEN_HEADER, token);
  let response: Response;
  try {
    response = await fetch(new URL(path, window.location.origin), { ...init, headers });
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new ApiError(0, "Can't reach the FlipTuned server.", "Check that `fliptuned serve` is still running.");
  }
  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      // not JSON
    }
    throw toApiError(response.status, body);
  }
  return response;
}
