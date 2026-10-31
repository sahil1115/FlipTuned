// Config fields from the server's JSON Schema (generated from the Pydantic config models): labels,
// descriptions, defaults and limits come from there, so the UI never restates them.

export interface JsonSchema {
  $ref?: string;
  $defs?: Record<string, JsonSchema>;
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  enum?: string[];
  const?: unknown;
  anyOf?: JsonSchema[];
  items?: JsonSchema;
  properties?: Record<string, JsonSchema>;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  minLength?: number;
}

export type FieldKind = "integer" | "number" | "boolean" | "string" | "enum" | "list";

export interface SchemaField {
  path: string; // dotted, e.g. "training.max_epochs"
  group: string; // path of the object that holds it ("" for top level)
  title: string;
  description: string;
  kind: FieldKind;
  options: string[];
  defaultValue: unknown;
  nullable: boolean;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  minLength?: number;
}

export interface SchemaGroup {
  path: string;
  description: string;
}

/** Resolve a local `$ref`, keeping the referring node's own description and default. */
export function deref(root: JsonSchema, node: JsonSchema): JsonSchema {
  if (!node.$ref) return node;
  const name = node.$ref.replace(/^#\/\$defs\//, "");
  const target = root.$defs?.[name] ?? {};
  const own = { ...node };
  delete own.$ref;
  return { ...target, ...own };
}

function kindOf(node: JsonSchema): { kind: FieldKind; options: string[]; nullable: boolean } | null {
  if (node.enum) return { kind: "enum", options: node.enum, nullable: false };
  if (node.anyOf) {
    const nullable = node.anyOf.some((n) => n.type === "null");
    const rest = node.anyOf.filter((n) => n.type !== "null");
    if (rest.some((n) => n.const === "auto") && rest.some((n) => n.type === "array")) {
      return { kind: "list", options: [], nullable };
    }
    if (rest.length === 1) {
      const inner = kindOf(rest[0]!);
      return inner ? { ...inner, nullable } : null;
    }
    return null;
  }
  if (node.type === "integer" || node.type === "number" || node.type === "boolean" || node.type === "string") {
    return { kind: node.type, options: [], nullable: false };
  }
  return null;
}

/** Every editable leaf field, in schema order. Constant fields (like `kind`) are left out. */
export function schemaFields(root: JsonSchema, node: JsonSchema = root, prefix = ""): SchemaField[] {
  const out: SchemaField[] = [];
  for (const [key, raw] of Object.entries(node.properties ?? {})) {
    const prop = deref(root, raw);
    const path = prefix ? `${prefix}.${key}` : key;
    if (prop.const !== undefined) continue;
    if (prop.type === "object" && prop.properties) {
      out.push(...schemaFields(root, prop, path));
      continue;
    }
    const kind = kindOf(prop);
    if (!kind) continue;
    out.push({
      path,
      group: prefix,
      title: prop.title ?? key,
      description: prop.description ?? "",
      ...kind,
      defaultValue: prop.default,
      minimum: prop.minimum,
      maximum: prop.maximum,
      exclusiveMinimum: prop.exclusiveMinimum,
      exclusiveMaximum: prop.exclusiveMaximum,
      minLength: prop.minLength,
    });
  }
  return out;
}

/** Object-valued properties (sections), with their descriptions. */
export function schemaGroups(root: JsonSchema, node: JsonSchema = root, prefix = ""): SchemaGroup[] {
  const out: SchemaGroup[] = [];
  for (const [key, raw] of Object.entries(node.properties ?? {})) {
    const prop = deref(root, raw);
    if (prop.type === "object" && prop.properties) {
      const path = prefix ? `${prefix}.${key}` : key;
      out.push({ path, description: prop.description ?? "" }, ...schemaGroups(root, prop, path));
    }
  }
  return out;
}

function limits(field: SchemaField, value: number): string | null {
  const { minimum, maximum, exclusiveMinimum, exclusiveMaximum } = field;
  const above = minimum !== undefined ? `at least ${minimum}` : exclusiveMinimum !== undefined ? `above ${exclusiveMinimum}` : null;
  const below = maximum !== undefined ? `at most ${maximum}` : exclusiveMaximum !== undefined ? `below ${exclusiveMaximum}` : null;
  const bad =
    (minimum !== undefined && value < minimum) ||
    (exclusiveMinimum !== undefined && value <= exclusiveMinimum) ||
    (maximum !== undefined && value > maximum) ||
    (exclusiveMaximum !== undefined && value >= exclusiveMaximum);
  if (!bad) return null;
  return `Use a value ${[above, below].filter(Boolean).join(" and ")}.`;
}

/** Parse what the user typed into the field's type, or say what's wrong with it. */
export function parseField(field: SchemaField, raw: string): { value?: unknown; error?: string } {
  const text = raw.trim();
  switch (field.kind) {
    case "integer": {
      if (!/^-?\d+$/.test(text)) return { error: "Use a whole number." };
      const value = Number(text);
      const error = limits(field, value);
      return error ? { error } : { value };
    }
    case "number": {
      const value = Number(text);
      if (!text || !Number.isFinite(value)) return { error: "Use a number (for example 0.0001 or 1e-4)." };
      const error = limits(field, value);
      return error ? { error } : { value };
    }
    case "boolean":
      if (text === "true" || text === "false") return { value: text === "true" };
      return { error: "Choose yes or no." };
    case "enum":
      return field.options.includes(text) ? { value: text } : { error: `Choose one of ${field.options.join(", ")}.` };
    case "list": {
      if (text === "auto") return { value: "auto" };
      const items = text.split(",").map((s) => s.trim());
      if (!text || items.some((s) => !/^[\w.]+$/.test(s))) {
        return { error: "Write 'auto' or layer names separated by commas (for example q_proj, v_proj)." };
      }
      return { value: items };
    }
    case "string":
      if (field.minLength !== undefined && raw.length < field.minLength) return { error: "This can't be empty." };
      return { value: raw };
  }
}

/** A value as the text the user would type for it. */
export function formatValue(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}

export function getPath(data: unknown, path: string): unknown {
  return path.split(".").reduce<unknown>((node, key) => {
    return node !== null && typeof node === "object" ? (node as Record<string, unknown>)[key] : undefined;
  }, data);
}

export function setPath(data: Record<string, unknown>, path: string, value: unknown): void {
  const keys = path.split(".");
  let node = data;
  for (const key of keys.slice(0, -1)) {
    if (typeof node[key] !== "object" || node[key] === null) node[key] = {};
    node = node[key] as Record<string, unknown>;
  }
  node[keys[keys.length - 1]!] = value;
}
