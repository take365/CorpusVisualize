export interface HighlightSpan {
  startChar: number;
  endChar: number;
  tag: string;
}

export interface EmotionScores {
  [label: string]: number;
}

export interface DialectScores {
  [region: string]: number;
}

export interface WordFeature {
  text: string;
  kana?: string;
  accent?: string;
  start: number;
  end: number;
  pitch_mean?: number;
  pitch_curve?: number[];
  loudness?: number;
  tempo?: number;
  valence?: number;
  arousal?: number;
}

export interface Segment {
  id: string;
  conversation_id: string;
  source_file: string;
  start: number;
  end: number;
  speaker: string;
  text: string;
  emotion?: EmotionScores;
  pitch?: number[];
  loudness?: number;
  tempo?: number;
  dialect?: DialectScores;
  highlights?: HighlightSpan[];
  words?: WordFeature[];
  analyzer?: Record<string, string>;
}

export interface CorpusFile {
  segments: Segment[];
  duration: number;
  analyzerSummary: Record<string, string>;
}
