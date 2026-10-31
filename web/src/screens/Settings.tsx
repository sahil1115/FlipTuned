// Settings: allowed folders, app home (read-only), chat idle timeout, upload limit, port, developer
// mode, and this machine's doctor report with install suggestions.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, type Schemas, unwrap } from "../api/client";
import { useReportError } from "../App";
import { Badge, Button, Card, ErrorPanel, Field, inputClass, Spinner, Stat } from "../components/ui";
import { useSettings } from "../lib/queries";

type Update = Partial<Pick<Schemas["Settings"], "allowed_roots" | "idle_timeout_min" | "upload_max_mb" | "dev_mode" | "port">>;

export function Settings() {
  const settings = useSettings();
  const save = useSave();
  if (settings.error) return <ErrorPanel error={settings.error} />;
  if (!settings.data) return <Spinner />;
  return (
    <div className="space-y-4">
      <Folders settings={settings.data} save={save} />
      <Options key={JSON.stringify(settings.data)} settings={settings.data} save={save} />
      <Doctor />
    </div>
  );
}

function useSave() {
  const client = useQueryClient();
  const report = useReportError();
  return useMutation({
    mutationFn: (update: Update) => unwrap(api.PUT("/api/settings", { body: update })),
    onSuccess: (saved) => client.setQueryData(["settings"], saved),
    onError: report,
  });
}

type Save = ReturnType<typeof useSave>;

function Folders({ settings, save }: { settings: Schemas["Settings"]; save: Save }) {
  const [folder, setFolder] = useState("");
  return (
    <Card title="Allowed folders">
      <div className="space-y-3 text-sm">
        <p>
          The app may browse and read data only inside these folders (plus its own uploads folder). Data sources are
          picked from here.
        </p>
        {settings.allowed_roots.length === 0 ? (
          <p>No folder allowed yet.</p>
        ) : (
          <ul className="space-y-1" aria-label="Allowed folders">
            {settings.allowed_roots.map((root) => (
              <li key={root} className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-xs">{root}</span>
                <Button
                  variant="ghost"
                  aria-label={`Remove ${root}`}
                  onClick={() => save.mutate({ allowed_roots: settings.allowed_roots.filter((r) => r !== root) })}
                >
                  Remove
                </Button>
              </li>
            ))}
          </ul>
        )}
        <form
          className="flex flex-wrap items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (folder.trim())
              save.mutate(
                { allowed_roots: [...settings.allowed_roots, folder.trim()] },
                { onSuccess: () => setFolder("") },
              );
          }}
        >
          <Field label="Folder to allow" htmlFor="set-folder" hint="A full path, as your system writes it.">
            <input id="set-folder" className={`${inputClass} md:w-[28rem]`} value={folder} onChange={(e) => setFolder(e.target.value)} />
          </Field>
          <Button type="submit" disabled={!folder.trim() || save.isPending}>
            Allow
          </Button>
        </form>
        <dl>
          <Stat label="App home (read-only; set with --home or FLIPTUNED_HOME)" value={<span className="font-mono text-sm">{settings.app_home}</span>} />
        </dl>
      </div>
    </Card>
  );
}

function numberError(text: string, min: number, max: number, integer: boolean): string | null {
  const value = Number(text);
  if (text.trim() === "" || Number.isNaN(value)) return "Enter a number.";
  if (integer && !Number.isInteger(value)) return "Use a whole number.";
  if (value < min || value > max) return `Use a value from ${min} to ${max}.`;
  return null;
}

function Options({ settings, save }: { settings: Schemas["Settings"]; save: Save }) {
  const [idle, setIdle] = useState(String(settings.idle_timeout_min));
  const [upload, setUpload] = useState(String(settings.upload_max_mb));
  const [port, setPort] = useState(String(settings.port));
  const [dev, setDev] = useState(settings.dev_mode);
  const errors = {
    idle: numberError(idle, 0.5, 1440, false),
    upload: numberError(upload, 1, 4096, true),
    port: numberError(port, 1024, 65535, true),
  };
  const invalid = Object.values(errors).some(Boolean);
  const current = Number(window.location.port) || null;
  return (
    <Card title="Options">
      <form
        className="space-y-3 text-sm"
        onSubmit={(e) => {
          e.preventDefault();
          if (!invalid)
            save.mutate({ idle_timeout_min: Number(idle), upload_max_mb: Number(upload), port: Number(port), dev_mode: dev });
        }}
      >
        <div className="grid gap-3 md:grid-cols-3">
          <Field label="Chat idle timeout (minutes)" htmlFor="set-idle" hint="A chat model is released after this long without a message." error={errors.idle}>
            <input id="set-idle" className={inputClass} inputMode="decimal" value={idle} onChange={(e) => setIdle(e.target.value)} />
          </Field>
          <Field label="Largest upload (MB)" htmlFor="set-upload" error={errors.upload}>
            <input id="set-upload" className={inputClass} inputMode="numeric" value={upload} onChange={(e) => setUpload(e.target.value)} />
          </Field>
          <Field
            label="Port"
            htmlFor="set-port"
            hint={`Used from the next \`fliptuned serve\` (the next free port if it's taken; --port overrides it).${current ? ` This launch uses ${current}.` : ""}`}
            error={errors.port}
          >
            <input id="set-port" className={inputClass} inputMode="numeric" value={port} onChange={(e) => setPort(e.target.value)} />
          </Field>
        </div>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={dev} onChange={(e) => setDev(e.target.checked)} />
          Developer mode (shows tiny test-only models, whose results mean nothing)
        </label>
        <div className="flex items-center gap-3">
          <Button type="submit" disabled={invalid || save.isPending}>
            Save
          </Button>
          {save.isSuccess && <span role="status">Saved.</span>}
        </div>
      </form>
    </Card>
  );
}

function Doctor() {
  const doctor = useQuery({ queryKey: ["doctor"], queryFn: () => unwrap(api.GET("/api/doctor")), staleTime: 60_000 });
  if (doctor.error) return <ErrorPanel error={doctor.error} />;
  if (!doctor.data) return <Spinner label="Checking this machine…" />;
  const d = doctor.data;
  const m = d.machine;
  return (
    <Card title="This machine (doctor)">
      <div className="space-y-4 text-sm">
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="System" value={`${m.os_version} (${m.arch})`} />
          <Stat label="Python" value={`${m.python_version}${m.emulated ? " (emulated!)" : ""}`} />
          <Stat label="CPU" value={`${m.cpu_name} · ${m.cpu_count_logical} threads`} />
          <Stat label="Memory" value={`${m.ram_gb} GB`} />
          <Stat
            label="Accelerator"
            value={
              (m.nvidia_gpus ?? []).length
                ? (m.nvidia_gpus ?? []).map((g) => `${g.name}${g.memory_total_gb ? ` (${g.memory_total_gb} GB)` : ""}`).join(", ")
                : m.apple_chip ?? "none (CPU)"
            }
          />
          <Stat label="Backend in use" value={d.backend.selected ? `${d.backend.selected.backend} on ${d.backend.selected.device}` : "none"} />
          <Stat label="FlipTuned" value={d.fliptuned_version} />
        </dl>
        {d.backend.selected && <p>Why: {d.backend.selected.reason}</p>}

        <section aria-label="Install suggestions" className="space-y-1">
          <h3 className="font-semibold">Install suggestions</h3>
          {d.backend.install_command ? (
            <p>
              The install command for this machine: <code className="rounded bg-slate-100 px-1 dark:bg-slate-800">{d.backend.install_command}</code>
            </p>
          ) : (
            <p>No install command applies to this machine.</p>
          )}
          {(d.backend.suggestions ?? []).length ? (
            <ul className="list-disc pl-5">
              {(d.backend.suggestions ?? []).map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          ) : (
            <p>Nothing to install: the best backend for this machine is available.</p>
          )}
        </section>

        <section aria-label="Backends" className="space-y-1">
          <h3 className="font-semibold">Backends</h3>
          <ul className="space-y-1">
            {d.backend.options.map((o) => (
              <li key={`${o.backend}-${o.device}`} className="flex flex-wrap items-center gap-2">
                <Badge tone={o.available ? "good" : "neutral"}>{o.available ? "available" : "not here"}</Badge>
                <span className="font-medium">
                  {o.backend} on {o.device}
                </span>
                <span className="text-slate-600 dark:text-slate-400">{o.reason}</span>
              </li>
            ))}
          </ul>
        </section>

        {[...(d.warnings ?? []), ...(d.backend.warnings ?? [])].length > 0 && (
          <section aria-label="Doctor warnings">
            <h3 className="font-semibold">Warnings</h3>
            <ul className="list-disc pl-5 text-amber-800 dark:text-amber-300">
              {[...(d.warnings ?? []), ...(d.backend.warnings ?? [])].map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          </section>
        )}

        <details>
          <summary className="cursor-pointer">Packages, disks and caches</summary>
          <div className="mt-2 grid gap-3 md:grid-cols-2">
            <table className="text-left text-xs" aria-label="Packages">
              <tbody>
                {Object.entries(d.packages).map(([name, version]) => (
                  <tr key={name}>
                    <td className="pr-3">{name}</td>
                    <td>{version ?? "not installed"}</td>
                  </tr>
                ))}
                <tr>
                  <td className="pr-3">torch build</td>
                  <td>{d.torch.installed ? `${d.torch.build ?? "?"}${d.torch.cuda_available ? ` (CUDA ${d.torch.cuda_runtime})` : ""}` : "not installed"}</td>
                </tr>
              </tbody>
            </table>
            <div className="space-y-1 text-xs">
              {d.disks.map((disk) => (
                <p key={disk.label}>
                  {disk.label}: {disk.path} — {disk.free_gb != null ? `${disk.free_gb} GB free` : "free space unknown"}
                </p>
              ))}
              <p>Hugging Face cache: {d.hf_cache}</p>
              <p>
                App home: {d.app_home} (from {d.app_home_source})
              </p>
            </div>
          </div>
        </details>
      </div>
    </Card>
  );
}
