// Column mapping and the ingest config built from a scan (pure functions, unit-tested).
import type { Schemas } from "../api/client";

export const MAPPING_FIELDS = ["question", "answer", "id", "tags", "category", "status", "group", "thread"] as const;
export type MappingField = (typeof MAPPING_FIELDS)[number];
export type Mapping = Partial<Record<MappingField, string>>;

export const FIELD_HELP: Record<MappingField, string> = {
  question: "The question (required).",
  answer: "The approved answer (required).",
  id: "Your own ID for the row (optional).",
  tags: "Comma-separated tags (optional).",
  category: "A category, used to group results (optional).",
  status: "approved / pending_review / rejected (optional; empty means pending review).",
  group: "Rows that belong together stay in one split (optional).",
  thread: "Ticket or thread ID; also keeps rows together (optional).",
};

export function initialMapping(suggested: Record<string, string | null | undefined>): Mapping {
  const mapping: Mapping = {};
  for (const field of MAPPING_FIELDS) {
    const column = suggested[field];
    if (column) mapping[field] = column;
  }
  return mapping;
}

/** Problems with a mapping, in plain language; empty when it's usable. */
export function validateMapping(mapping: Mapping, headers: string[]): string[] {
  const problems: string[] = [];
  if (!mapping.question) problems.push("Pick the column that holds the question.");
  if (!mapping.answer) problems.push("Pick the column that holds the answer.");
  const used = new Map<string, MappingField>();
  for (const field of MAPPING_FIELDS) {
    const column = mapping[field];
    if (!column) continue;
    if (!headers.includes(column)) problems.push(`Column "${column}" (${field}) isn't in this file.`);
    const other = used.get(column);
    if (other) problems.push(`Column "${column}" is used for both ${other} and ${field}.`);
    used.set(column, field);
  }
  return problems;
}

export function validateSourceName(name: string): string | null {
  const trimmed = name.trim();
  if (!trimmed) return "Give the data source a name.";
  if (trimmed.length > 100) return "Keep the name under 100 characters.";
  return null;
}

/** `root` + a path relative to it (POSIX style from the server), in the root's native style. */
export function joinPath(root: string, relative: string): string {
  const windows = /^[A-Za-z]:\\/.test(root) || root.includes("\\");
  const sep = windows ? "\\" : "/";
  const rel = windows ? relative.replace(/\//g, "\\") : relative;
  return root.endsWith(sep) ? root + rel : root + sep + rel;
}

export function tabularKey(file: Schemas["TabularFile"]): string {
  return file.sheet ? `${file.path}#${file.sheet}` : file.path;
}

export function buildIngestConfig(
  name: string,
  scan: Schemas["ScanResult"],
  mappings: Record<string, Mapping>,
): { name: string; sources: Record<string, unknown>[] } {
  const sources: Record<string, unknown>[] = scan.tabular.map((file) => {
    const columns = Object.fromEntries(
      Object.entries(mappings[tabularKey(file)] ?? {}).filter(([, column]) => Boolean(column)),
    );
    return {
      path: joinPath(scan.root, file.path),
      columns,
      ...(file.sheet ? { sheet: file.sheet } : {}),
    };
  });
  for (const path of scan.text_files ?? []) sources.push({ path: joinPath(scan.root, path) });
  return { name: name.trim(), sources };
}
