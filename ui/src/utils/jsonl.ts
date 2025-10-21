import type { CorpusFile, Segment } from "../types";

function buildCorpus(segments: Segment[]): CorpusFile {
  const duration = segments.reduce((acc, seg) => Math.max(acc, seg.end), 0);
  const analyzerSummary = segments[0]?.analyzer ?? {};
  return {
    segments,
    duration,
    analyzerSummary,
  };
}

export async function parseJsonlFile(file: File): Promise<CorpusFile> {
  const text = await file.text();
  const trimmed = text.trim();

  if (!trimmed) {
    return { segments: [], duration: 0, analyzerSummary: {} };
  }

  // Attempt to parse as JSON (array or object)
  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) {
      return buildCorpus(parsed as Segment[]);
    }
    if (parsed && Array.isArray(parsed.segments)) {
      const corpus = buildCorpus(parsed.segments as Segment[]);
      if (typeof parsed.duration === "number") {
        corpus.duration = parsed.duration;
      }
      if (parsed.analyzerSummary && typeof parsed.analyzerSummary === "object") {
        corpus.analyzerSummary = parsed.analyzerSummary as Record<string, string>;
      }
      return corpus;
    }
  } catch (error) {
    // fall back to JSONL parsing
    console.debug("Failed JSON parse, falling back to JSONL", error);
  }

  const segments: Segment[] = [];

  for (const rawLine of trimmed.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    try {
      const parsedLine = JSON.parse(line);
      segments.push(parsedLine as Segment);
    } catch (error) {
      console.warn("Skipping malformed JSONL line", { error, line });
    }
  }

  return buildCorpus(segments);
}

export function formatSeconds(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds)) return "--:--";
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${seconds}`;
}
