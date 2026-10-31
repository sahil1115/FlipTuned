// App shell: navigation (only the screens that exist), token check, global errors, theme.
import { Component, createContext, lazy, Suspense, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

import { ApiError, getToken } from "./api/client";
import { Button, ErrorPanel, Spinner } from "./components/ui";
import { routePath } from "./lib/route";
import { Dashboard } from "./screens/Dashboard";
import { DataSources } from "./screens/DataSources";
import { Dataset } from "./screens/Dataset";
import { Models } from "./screens/Models";
import { Preflight } from "./screens/Preflight";
import { Chat } from "./screens/Chat";
import { Evaluation } from "./screens/Evaluation";
import { Review } from "./screens/Review";
import { Runs } from "./screens/Runs";
import { Settings } from "./screens/Settings";
import { TrainingConfig } from "./screens/TrainingConfig";

// The monitor brings the chart library; it loads when first opened.
const RunMonitor = lazy(() => import("./screens/RunMonitor").then((m) => ({ default: m.RunMonitor })));

export const ROUTES = [
  { path: "/", label: "Dashboard", element: () => <Dashboard /> },
  { path: "/sources", label: "Data sources", element: () => <DataSources /> },
  { path: "/review", label: "Review", element: () => <Review /> },
  { path: "/dataset", label: "Dataset", element: () => <Dataset /> },
  { path: "/models", label: "Models", element: () => <Models /> },
  { path: "/config", label: "Training config", element: () => <TrainingConfig /> },
  { path: "/preflight", label: "Preflight", element: () => <Preflight /> },
  { path: "/monitor", label: "Run monitor", element: () => <RunMonitor /> },
  { path: "/evaluation", label: "Evaluation", element: () => <Evaluation /> },
  { path: "/chat", label: "Chat", element: () => <Chat /> },
  { path: "/runs", label: "Runs", element: () => <Runs /> },
  { path: "/settings", label: "Settings", element: () => <Settings /> },
] as const;

function currentPath(): string {
  const path = routePath();
  return ROUTES.some((r) => r.path === path) ? path : "/";
}

// --------------------------------------------------------------------------- global errors

const ErrorContext = createContext<(error: unknown) => void>(() => {});

/** Show an error in the banner at the top of the page. */
export function useReportError(): (error: unknown) => void {
  return useContext(ErrorContext);
}

class Boundary extends Component<{ children: ReactNode }, { error: unknown }> {
  state = { error: null as unknown };
  static getDerivedStateFromError(error: unknown) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div className="space-y-2">
          <ErrorPanel error={this.state.error} />
          <Button variant="secondary" onClick={() => this.setState({ error: null })}>
            Try again
          </Button>
        </div>
      );
    }
    return this.props.children;
  }
}

// --------------------------------------------------------------------------- theme

const THEME_KEY = "fliptuned-theme";

function useTheme(): [boolean, () => void] {
  const [dark, setDark] = useState(() => {
    let saved: string | null = null;
    try {
      saved = window.localStorage.getItem(THEME_KEY);
    } catch {
      // storage unavailable: follow the OS
    }
    return saved ? saved === "dark" : window.matchMedia?.("(prefers-color-scheme: dark)").matches === true;
  });
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);
  const toggle = useCallback(() => {
    setDark((value) => {
      try {
        window.localStorage.setItem(THEME_KEY, value ? "light" : "dark");
      } catch {
        // not saved; still switches for this visit
      }
      return !value;
    });
  }, []);
  return [dark, toggle];
}

// --------------------------------------------------------------------------- shell

export function App() {
  const [path, setPath] = useState(currentPath);
  const [, setHash] = useState(() => window.location.hash); // re-render on query changes too
  const [errors, setErrors] = useState<unknown[]>([]);
  const [dark, toggleTheme] = useTheme();
  const report = useCallback((error: unknown) => setErrors((list) => [...list, error]), []);

  useEffect(() => {
    const onHash = () => {
      setPath(currentPath());
      setHash(window.location.hash);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  if (!getToken()) {
    return (
      <main className="mx-auto max-w-xl p-8">
        <h1 className="mb-4 text-2xl font-semibold">FlipTuned</h1>
        <ErrorPanel
          error={
            new ApiError(
              401,
              "This page needs the access token from `fliptuned serve`.",
              "Open the link the command printed; each launch has a new token.",
            )
          }
        />
        <p className="mt-3 text-sm">
          Open the link the <code>fliptuned serve</code> command printed (it ends with <code>#token=…</code>). Each
          launch has a new token.
        </p>
      </main>
    );
  }

  const route = ROUTES.find((r) => r.path === path) ?? ROUTES[0];
  return (
    <ErrorContext.Provider value={report}>
      <a
        href="#main"
        onClick={(e) => {
          e.preventDefault();
          document.getElementById("main")?.focus();
        }}
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-10 focus:rounded focus:bg-white focus:p-2 dark:focus:bg-slate-900"
      >
        Skip to content
      </a>
      <header className="border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-4 px-4 py-3">
          <span className="text-lg font-semibold">FlipTuned</span>
          <nav aria-label="Main" className="flex flex-wrap gap-1">
            {ROUTES.map((r) => (
              <a
                key={r.path}
                href={`#${r.path}`}
                aria-current={r.path === route.path ? "page" : undefined}
                className={`rounded-md px-3 py-1.5 text-sm ${
                  r.path === route.path
                    ? "bg-sky-100 font-medium text-sky-900 dark:bg-slate-800 dark:text-sky-200"
                    : "hover:bg-slate-100 dark:hover:bg-slate-800"
                }`}
              >
                {r.label}
              </a>
            ))}
          </nav>
          <span className="ml-auto text-xs text-slate-600 dark:text-slate-400">
            Runs on this computer only · no retrieval at chat time
          </span>
          <Button variant="secondary" onClick={toggleTheme} aria-pressed={dark}>
            {dark ? "Light theme" : "Dark theme"}
          </Button>
        </div>
      </header>
      <main id="main" tabIndex={-1} className="mx-auto max-w-6xl space-y-4 px-4 py-6 outline-none">
        {errors.map((error, index) => (
          <ErrorPanel key={index} error={error} onDismiss={() => setErrors((list) => list.filter((_, i) => i !== index))} />
        ))}
        <h1 className="text-2xl font-semibold">{route.label}</h1>
        <Boundary key={route.path}>
          <Suspense fallback={<Spinner />}>{route.element()}</Suspense>
        </Boundary>
      </main>
    </ErrorContext.Provider>
  );
}
