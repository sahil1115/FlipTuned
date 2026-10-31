// Hash routes with optional query parameters, e.g. `#/preflight?config=quick&version=2`.

export function routePath(hash: string = window.location.hash): string {
  return hash.replace(/^#/, "").split("?")[0] || "/";
}

export function routeParams(hash: string = window.location.hash): URLSearchParams {
  return new URLSearchParams(hash.split("?")[1] ?? "");
}

export function routeHref(path: string, params: Record<string, string | number | undefined> = {}): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) if (value !== undefined && value !== "") query.set(key, String(value));
  const text = query.toString();
  return `#${path}${text ? `?${text}` : ""}`;
}
