import type { CorpusFile, Segment, WordFeature } from "../types";

function normalizeWordFeature(raw: any, fallbackStart: number, fallbackEnd: number): WordFeature | null {
  const text = typeof raw?.text === "string" ? raw.text : String(raw?.text ?? "").trim();
  if (!text) {
    return null;
  }
  const start = typeof raw?.start === "number" ? raw.start : fallbackStart;
  const endValue = typeof raw?.end === "number" ? raw.end : fallbackEnd;
  const end = endValue > start ? endValue : fallbackEnd;
  const pitchCurve = Array.isArray(raw?.pitch_curve)
    ? raw.pitch_curve.filter((v: unknown) => typeof v === "number" && Number.isFinite(v))
    : [];

  return {
    text,
    kana: typeof raw?.kana === "string" ? raw.kana : undefined,
    accent: typeof raw?.accent === "string" ? raw.accent : undefined,
    start,
    end,
    pitch_mean: typeof raw?.pitch_mean === "number" ? raw.pitch_mean : undefined,
    pitch_curve: pitchCurve,
    loudness: typeof raw?.loudness === "number" ? raw.loudness : undefined,
    tempo: typeof raw?.tempo === "number" ? raw.tempo : undefined,
    valence: typeof raw?.valence === "number" ? raw.valence : undefined,
    arousal: typeof raw?.arousal === "number" ? raw.arousal : undefined,
  };
}

function normalizeSegment(raw: any): Segment {
  const segment = raw as Segment;
  const wordsRaw = Array.isArray((raw as any).words) ? (raw as any).words : [];
  const words: WordFeature[] = [];
  for (const word of wordsRaw) {
    const normalized = normalizeWordFeature(word, segment.start, segment.end);
    if (normalized) {
      words.push(normalized);
    }
  }
  return {
    ...segment,
    words,
  };
}

function buildCorpus(segments: Segment[]): CorpusFile {
  const normalizedSegments = segments.map((seg) => normalizeSegment(seg));
  const duration = normalizedSegments.reduce((acc, seg) => Math.max(acc, seg.end), 0);
  const analyzerSummary = normalizedSegments[0]?.analyzer ?? {};
  return {
    segments: normalizedSegments,
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
