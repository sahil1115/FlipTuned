// Data sources: allowed folders, browse or upload, scan, confirm the column mapping, ingest.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api, rawFetch, type Schemas, unwrap } from "../api/client";
import { useReportError } from "../App";
import { JobProgress } from "../components/JobProgress";
import { Badge, Button, Card, CountList, ErrorPanel, Field, inputClass, Spinner, Stat } from "../components/ui";
import {
  buildIngestConfig,
  FIELD_HELP,
  initialMapping,
  MAPPING_FIELDS,
  type Mapping,
  tabularKey,
  validateMapping,
  validateSourceName,
} from "../lib/ingest";
import { useLatestJob, useSettings } from "../lib/queries";

export function DataSources() {
  const [scan, setScan] = useState<Schemas["ScanResult"] | null>(null);
  return (
    <div className="space-y-4">
      <AllowedFolders />
      <div className="grid gap-4 lg:grid-cols-2">
        <Browser onScanned={setScan} />
        <Upload onScanned={setScan} />
      </div>
      {scan ? <ScanAndIngest key={scan.root + scan.files_total} scan={scan} /> : <LatestIngest />}
    </div>
  );
}

function AllowedFolders() {
  const settings = useSettings();
  const client = useQueryClient();
  const report = useReportError();
  const [path, setPath] = useState("");
  const save = useMutation({
    mutationFn: (roots: string[]) => unwrap(api.PUT("/api/settings", { body: { allowed_roots: roots } })),
    onSuccess: () => {
      setPath("");
      client.invalidateQueries({ queryKey: ["settings"] });
      client.invalidateQueries({ queryKey: ["browse"] });
    },
    onError: report,
  });
  const roots = settings.data?.allowed_roots ?? [];
  return (
    <Card title="Allowed folders">
      <p className="mb-2 text-sm text-slate-600 dark:text-slate-400">
        FlipTuned reads data only from these folders (and from files you upload).
      </p>
      {settings.error ? <ErrorPanel error={settings.error} /> : null}
      <ul className="mb-3 space-y-1 text-sm">
        {roots.map((root) => (
          <li key={root} className="flex items-center gap-2">
            <code>{root}</code>
            <Button
              variant="ghost"
              aria-label={`Remove ${root}`}
              onClick={() => save.mutate(roots.filter((r) => r !== root))}
            >
              Remove
            </Button>
          </li>
        ))}
        {!roots.length && settings.data && <li>No folders yet: add the folder that holds your Q&A files.</li>}
      </ul>
      <form
        className="flex flex-wrap items-end gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (path.trim()) save.mutate([...roots, path.trim()]);
        }}
      >
        <div className="min-w-64 flex-1">
          <Field label="Add a folder" htmlFor="add-root" hint="The full path, as shown by your file manager.">
            <input id="add-root" className={inputClass} value={path} onChange={(e) => setPath(e.target.value)} />
          </Field>
        </div>
        <Button type="submit" disabled={!path.trim() || save.isPending}>
          Add folder
        </Button>
      </form>
    </Card>
  );
}

function useScan(onScanned: (scan: Schemas["ScanResult"]) => void) {
  const report = useReportError();
  const client = useQueryClient();
  return useMutation({
    mutationFn: (args: { path: string; name?: string }) => unwrap(api.POST("/api/scan", { body: args })),
    onSuccess: (result) => {
      client.invalidateQueries({ queryKey: ["sources"] });
      onScanned(result);
    },
    onError: report,
  });
}

function Browser({ onScanned }: { onScanned: (scan: Schemas["ScanResult"]) => void }) {
  const [path, setPath] = useState<string | null>(null);
  const listing = useQuery({
    queryKey: ["browse", path],
    queryFn: () => unwrap(api.GET("/api/fs/browse", { params: { query: { path: path ?? undefined } } })),
  });
  const scan = useScan(onScanned);
  const data = listing.data;
  return (
    <Card
      title="Browse"
      actions={
        data?.path ? (
          <Button onClick={() => scan.mutate({ path: data.path! })} disabled={scan.isPending}>
            {scan.isPending ? "Scanning…" : "Scan this folder"}
          </Button>
        ) : null
      }
    >
      <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
        <Button variant="secondary" onClick={() => setPath(null)} disabled={path === null}>
          Allowed folders
        </Button>
        {data?.parent && (
          <Button variant="secondary" onClick={() => setPath(data.parent ?? null)}>
            Up
          </Button>
        )}
        {data?.path && <code className="break-all">{data.path}</code>}
      </div>
      {listing.error ? (
        <ErrorPanel error={listing.error} />
      ) : !data ? (
        <Spinner />
      ) : data.entries.length === 0 ? (
        <p className="text-sm">{path === null ? "No allowed folders yet." : "This folder is empty."}</p>
      ) : (
        <ul className="max-h-72 space-y-0.5 overflow-y-auto text-sm" aria-label="Folder contents">
          {data.entries.map((entry) => (
            <li key={entry.path} className="flex items-center gap-2">
              {entry.kind === "folder" ? (
                <button
                  type="button"
                  className="text-left text-sky-800 underline dark:text-sky-300"
                  onClick={() => setPath(entry.path)}
                >
                  📁 {entry.name}
                </button>
              ) : (
                <span>
                  📄 {entry.name}{" "}
                  {entry.supported ? <Badge tone="good">readable</Badge> : <Badge>not used</Badge>}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function Upload({ onScanned }: { onScanned: (scan: Schemas["ScanResult"]) => void }) {
  const [name, setName] = useState("uploads");
  const [files, setFiles] = useState<File[]>([]);
  const report = useReportError();
  const scan = useScan(onScanned);
  const nameError = validateSourceName(name);
  const upload = useMutation({
    mutationFn: async () => {
      let folder = "";
      for (const file of files) {
        const response = await rawFetch(
          `/api/sources/${encodeURIComponent(name.trim())}/uploads?filename=${encodeURIComponent(file.name)}&overwrite=true`,
          { method: "POST", body: file, headers: { "Content-Type": "application/octet-stream" } },
        );
        folder = ((await response.json()) as Schemas["UploadResult"]).folder;
      }
      return folder;
    },
    onSuccess: (folder) => scan.mutate({ path: folder, name: name.trim() }),
    onError: report,
  });
  return (
    <Card title="Upload files">
      <form
        className="space-y-3"
        onSubmit={(e) => {
          e.preventDefault();
          if (!nameError && files.length) upload.mutate();
        }}
      >
        <Field label="Data source name" htmlFor="upload-name" error={nameError}>
          <input id="upload-name" className={inputClass} value={name} onChange={(e) => setName(e.target.value)} />
        </Field>
        <Field label="Files" htmlFor="upload-files" hint="CSV, XLSX, Markdown or TXT. Copied into the app home.">
          <input
            id="upload-files"
            type="file"
            multiple
            accept=".csv,.xlsx,.md,.txt"
            className="block text-sm"
            onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
          />
        </Field>
        <Button type="submit" disabled={!!nameError || !files.length || upload.isPending || scan.isPending}>
          {upload.isPending ? "Uploading…" : "Upload and scan"}
        </Button>
      </form>
    </Card>
  );
}

function ScanAndIngest({ scan }: { scan: Schemas["ScanResult"] }) {
  const [name, setName] = useState(scan.name);
  const [overwrite, setOverwrite] = useState(false);
  const [mappings, setMappings] = useState<Record<string, Mapping>>(() =>
    Object.fromEntries(scan.tabular.map((file) => [tabularKey(file), initialMapping(file.suggested_mapping)])),
  );
  const [jobId, setJobId] = useState<string | null>(null);
  const report = useReportError();
  const client = useQueryClient();
  const problems = scan.tabular.flatMap((file) =>
    validateMapping(mappings[tabularKey(file)] ?? {}, file.headers).map((p) => `${tabularKey(file)}: ${p}`),
  );
  const nameError = validateSourceName(name);
  const readable = scan.tabular.length + (scan.text_files?.length ?? 0);
  const ingest = useMutation({
    mutationFn: () =>
      unwrap(api.POST("/api/ingest", { body: { config: buildIngestConfig(name, scan, mappings), overwrite } })),
    onSuccess: (job) => {
      setJobId(job.id);
      client.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: report,
  });

  return (
    <div className="space-y-4">
      <Card title={`Scan of ${scan.root}`}>
        <dl className="mb-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Files" value={scan.files_total} />
          <Stat label="Readable" value={scan.supported_files} />
          <Stat label="Not used" value={scan.unsupported_files.length} />
          <Stat label="Encoding problems" value={scan.encoding_problems.length} />
        </dl>
        <CountList counts={Object.fromEntries(Object.entries(scan.by_type).map(([ext, s]) => [ext, s.count]))} />
        {scan.unsupported_files.length > 0 && (
          <details className="mt-2 text-sm">
            <summary className="cursor-pointer">Files FlipTuned won't read ({scan.unsupported_files.length})</summary>
            <ul className="ml-4 list-disc">
              {scan.unsupported_files.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
          </details>
        )}
        {scan.encoding_problems.length > 0 && (
          <ul className="mt-2 text-sm text-amber-800 dark:text-amber-300">
            {scan.encoding_problems.map((p) => (
              <li key={p.path}>
                {p.path}: not valid UTF-8 near byte {p.byte_offset}; its records will be quarantined.
              </li>
            ))}
          </ul>
        )}
      </Card>

      {scan.tabular.map((file) => (
        <MappingCard
          key={tabularKey(file)}
          file={file}
          mapping={mappings[tabularKey(file)] ?? {}}
          onChange={(m) => setMappings((all) => ({ ...all, [tabularKey(file)]: m }))}
        />
      ))}
      {(scan.text_files?.length ?? 0) > 0 && (
        <Card title="Markdown and text files">
          <p className="mb-1 text-sm">Q&A pairs are read with the standard delimiters (Q: / A:).</p>
          <ul className="ml-4 list-disc text-sm">
            {(scan.text_files ?? []).map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </Card>
      )}

      <Card title="Ingest">
        <form
          className="space-y-3"
          onSubmit={(e) => {
            e.preventDefault();
            if (!problems.length && !nameError && readable) ingest.mutate();
          }}
        >
          <Field label="Data source name" htmlFor="ingest-name" error={nameError}>
            <input id="ingest-name" className={inputClass} value={name} onChange={(e) => setName(e.target.value)} />
          </Field>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} />
            Replace records from an earlier ingest of this source
          </label>
          {problems.length > 0 && (
            <ul role="alert" className="ml-4 list-disc text-sm text-rose-700 dark:text-rose-300">
              {problems.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          )}
          {!readable && <p className="text-sm">No readable files in this folder.</p>}
          <Button type="submit" disabled={!!problems.length || !!nameError || !readable || ingest.isPending}>
            Ingest {readable} file{readable === 1 ? "" : "s"}
          </Button>
        </form>
        {jobId && (
          <div className="mt-3">
            <IngestJob jobId={jobId} />
          </div>
        )}
      </Card>
    </div>
  );
}

function MappingCard({
  file,
  mapping,
  onChange,
}: {
  file: Schemas["TabularFile"];
  mapping: Mapping;
  onChange: (mapping: Mapping) => void;
}) {
  const id = tabularKey(file).replace(/[^A-Za-z0-9]/g, "-");
  return (
    <Card title={`Columns of ${tabularKey(file)}`}>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {MAPPING_FIELDS.map((field) => (
          <Field key={field} label={field} htmlFor={`${id}-${field}`} hint={FIELD_HELP[field]}>
            <select
              id={`${id}-${field}`}
              className={inputClass}
              value={mapping[field] ?? ""}
              onChange={(e) => onChange({ ...mapping, [field]: e.target.value || undefined })}
            >
              <option value="">{field === "question" || field === "answer" ? "— choose —" : "(none)"}</option>
              {file.headers.map((header) => (
                <option key={header} value={header}>
                  {header}
                </option>
              ))}
            </select>
          </Field>
        ))}
      </div>
    </Card>
  );
}

interface IngestCounts {
  records_total: number;
  by_status: Record<string, number>;
  quality_flags: Record<string, number>;
  privacy_findings: number;
  excluded: number;
}

function IngestJob({ jobId }: { jobId: string }) {
  return (
    <JobProgress jobId={jobId} label="Ingest">
      {(state) => {
        const counts = (state.outcome?.result as { counts?: IngestCounts } | null)?.counts;
        return counts ? <IngestResult counts={counts} /> : null;
      }}
    </JobProgress>
  );
}

function IngestResult({ counts }: { counts: IngestCounts }) {
  const client = useQueryClient();
  useEffect(() => {
    void client.invalidateQueries({ queryKey: ["sources"] });
  }, [client]);
  return (
    <div className="space-y-2 text-sm">
      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Records" value={counts.records_total} />
        <Stat label="Privacy findings" value={counts.privacy_findings} />
        <Stat label="Excluded" value={counts.excluded} />
      </dl>
      <div>
        By status: <CountList counts={counts.by_status} />
      </div>
      <div>
        Quality flags: <CountList counts={counts.quality_flags} />
      </div>
      <p>
        Next: <a className="text-sky-800 underline dark:text-sky-300" href="#/review">review the records</a>.
      </p>
    </div>
  );
}

function LatestIngest() {
  const job = useLatestJob("ingest");
  if (!job) return null;
  return (
    <Card title="Latest ingest">
      <IngestJob jobId={job.id} />
    </Card>
  );
}
