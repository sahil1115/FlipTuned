# web/

The FlipTuned web UI: React 19 + TypeScript + Vite, Tailwind, TanStack Query and a typed API client
generated from the server's OpenAPI schema. `fliptuned serve` serves the build from
`src/fliptuned/web_dist/` (git-ignored); every asset is bundled, nothing loads from the internet.

```text
npm --prefix web ci              # install (Node 22)
npm --prefix web run build       # type-check, then build into src/fliptuned/web_dist/
npm --prefix web run lint
npm --prefix web run test        # Vitest (jsdom)
npm --prefix web run gen:api     # regenerate openapi.json, src/test/train-schema.json, src/api/schema.d.ts
npm --prefix web run check:api   # fails if either is stale (CI runs this)
uv run pytest -m e2e             # Playwright against a real `fliptuned serve` (needs the build)
```

Layout: `src/api/` (client, token, SSE), `src/components/` (UI pieces), `src/lib/` (hooks, form
logic), `src/screens/` (one file per screen). Inline `style` attributes are not allowed (the server's
CSP blocks them; ESLint enforces it).
