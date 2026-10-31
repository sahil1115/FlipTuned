// A fake FlipTuned server for screen tests: routes keyed "METHOD /path", answering JSON (or an
// event stream when a handler returns `sse`).
import { vi } from "vitest";

export type Handler = (
  url: URL,
  body: string,
  headers: Headers,
) => { status?: number; body?: unknown; sse?: string };
export interface Call {
  method: string;
  url: string;
  headers: Headers;
  body: string;
}

export function mockServer(routes: Record<string, Handler | unknown>): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
      const request = input instanceof Request ? input : new Request(String(input), init);
      const url = new URL(request.url, "http://127.0.0.1:8765");
      const sent = request.method === "GET" ? "" : await request.clone().text();
      calls.push({ method: request.method, url: url.pathname, headers: request.headers, body: sent });
      const key = `${request.method} ${url.pathname}`;
      const route = routes[key];
      if (route === undefined) return new Response(JSON.stringify({ error: { message: `no mock for ${key}` } }), { status: 404 });
      const { status = 200, body, sse } =
        typeof route === "function" ? (route as Handler)(url, sent, request.headers) : { body: route, sse: undefined };
      if (sse !== undefined) return new Response(sse, { status, headers: { "Content-Type": "text/event-stream" } });
      return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
    }),
  );
  return calls;
}
