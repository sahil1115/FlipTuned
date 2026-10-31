// Shows record text exactly as stored: prose wraps, fenced code blocks keep every character and
// show their language. Nothing is reformatted.

export interface Segment {
  kind: "text" | "code";
  content: string;
  language?: string;
}

const OPEN = /^ {0,3}(`{3,}|~{3,})[ \t]*([^`\s]*)/;
const CLOSE = /^ {0,3}(`{3,}|~{3,})[ \t]*$/;

export function splitFences(text: string): Segment[] {
  const segments: Segment[] = [];
  let prose: string[] = [];
  let code: string[] | null = null;
  let fence = "";
  let language = "";
  const flushProse = () => {
    if (prose.length) segments.push({ kind: "text", content: prose.join("\n") });
    prose = [];
  };
  for (const raw of text.split("\n")) {
    const line = raw.replace(/\r$/, "");
    if (code === null) {
      const open = OPEN.exec(line);
      if (open?.[1]) {
        flushProse();
        code = [];
        fence = open[1];
        language = open[2] ?? "";
      } else {
        prose.push(raw);
      }
      continue;
    }
    const close = CLOSE.exec(line);
    if (close?.[1] && close[1][0] === fence[0] && close[1].length >= fence.length) {
      segments.push({ kind: "code", content: code.join("\n"), language });
      code = null;
    } else {
      code.push(raw);
    }
  }
  if (code !== null) segments.push({ kind: "code", content: code.join("\n"), language });
  flushProse();
  return segments;
}

export function CodeText({ text }: { text: string }) {
  return (
    <div className="space-y-2 text-sm">
      {splitFences(text).map((segment, index) =>
        segment.kind === "code" ? (
          <figure key={index} className="rounded-md border border-slate-200 dark:border-slate-700">
            {segment.language && (
              <figcaption className="border-b border-slate-200 px-2 py-0.5 text-xs text-slate-600 dark:border-slate-700 dark:text-slate-400">
                {segment.language}
              </figcaption>
            )}
            <pre className="overflow-x-auto bg-slate-100 p-2 dark:bg-slate-800">
              <code>{segment.content}</code>
            </pre>
          </figure>
        ) : (
          <p key={index} className="whitespace-pre-wrap break-words">
            {segment.content}
          </p>
        ),
      )}
    </div>
  );
}
