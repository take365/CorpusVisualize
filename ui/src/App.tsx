import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import {
  Upload,
  Settings2,
  Wand2,
  BarChart3,
  FileBarChart,
  Gauge,
  Languages,
  Sparkles,
  Mic,
  Square,
  PlayCircle,
  Download,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { CorpusFile, Segment } from "./types";
import { formatSeconds, parseJsonlFile } from "./utils/jsonl";
import { downloadJson } from "./utils/download";

declare global {
  interface Window {
    webkitSpeechRecognition?: any;
    SpeechRecognition?: any;
    webkitAudioContext?: typeof AudioContext;
  }
}

type InputMode = "import" | "mic";

type FeatureToggleKey = "emotion" | "pitch" | "loudness" | "tempo" | "dialect" | "lexicon";

type FeatureToggle = Record<FeatureToggleKey, boolean>;

interface MicDraftSegment {
  id: string;
  start: number;
  end: number;
  text: string;
  rms: number;
}

const moduleOrder = [
  "asr",
  "diarization",
  "emotion",
  "pitch",
  "loudness",
  "tempo",
  "dialect",
  "lexicon",
] as const;

const featureKeys: FeatureToggleKey[] = ["emotion", "pitch", "loudness", "tempo", "dialect", "lexicon"];

const initialFeatureToggles: FeatureToggle = {
  emotion: true,
  pitch: true,
  loudness: true,
  tempo: true,
  dialect: true,
  lexicon: true,
};

const DIALECT_KEYS = ["kansai", "kanto", "tohoku", "kyushu", "hokkaido"] as const;
const DIALECT_UNIFORM = DIALECT_KEYS.reduce<Record<string, number>>((acc, key) => {
  acc[key] = 1 / DIALECT_KEYS.length;
  return acc;
}, {});

function unique<T>(values: T[]): T[] {
  return Array.from(new Set(values));
}

const Card = ({
  title,
  icon,
  actions,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
}) => (
  <section className="bg-white/80 backdrop-blur border border-slate-200 rounded-2xl p-5 shadow-sm">
    <header className="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
      <div className="flex items-center gap-2 text-slate-700">
        {icon}
        <h2 className="text-base font-semibold">{title}</h2>
      </div>
      {actions}
    </header>
    {children}
  </section>
);

const ModuleBadge = ({ label, value }: { label: string; value?: string }) => (
  <span className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">
    <Settings2 size={12} className="text-slate-400" />
    {label}
    {value ? <span className="font-semibold text-slate-700">{value}</span> : <span className="text-slate-400">n/a</span>}
  </span>
);

function useEmotionKeys(segments: Segment[]): string[] {
  return useMemo(() => {
    const all = segments.flatMap((seg) => Object.keys(seg.emotion ?? {}));
    return unique(all);
  }, [segments]);
}

function useDialectKeys(segments: Segment[]): string[] {
  return useMemo(() => {
    const all = segments.flatMap((seg) => Object.keys(seg.dialect ?? {}));
    return unique(all);
  }, [segments]);
}

function EmotionAreaChart({ segments, keys }: { segments: Segment[]; keys: string[] }) {
  const data = useMemo(
    () =>
      segments.map((seg) => ({
        id: seg.id,
        label: `${seg.speaker}  ${formatSeconds(seg.start)}-${formatSeconds(seg.end)}`,
        ...seg.emotion,
      })),
    [segments],
  );

  if (!keys.length || !data.length) {
    return <p className="text-sm text-slate-500">感情スコアが見つかりませんでした。</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="label" hide />
        <YAxis domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} width={30} />
        <Tooltip formatter={(value: number) => value.toFixed(2)} />
        <Legend verticalAlign="top" height={36} />
        {keys.map((key, index) => (
          <Area
            key={key}
            type="monotone"
            dataKey={key}
            stackId="emotion"
            stroke={`hsl(${index * 60}, 70%, 45%)`}
            fill={`hsl(${index * 60}, 70%, 60%)`}
            fillOpacity={0.65}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}

function PitchLineChart({ segments }: { segments: Segment[] }) {
  const pitchPoints = useMemo(() => {
    const points: Array<{ time: number; value: number; speaker: string }> = [];
    segments.forEach((segment) => {
      const values = segment.pitch ?? [];
      if (!values.length) return;
      const duration = segment.end - segment.start;
      values.forEach((value, idx) => {
        const ratio = idx / Math.max(1, values.length - 1);
        points.push({
          time: segment.start + ratio * duration,
          value,
          speaker: segment.speaker,
        });
      });
    });
    return points.sort((a, b) => a.time - b.time);
  }, [segments]);

  if (!pitchPoints.length) {
    return <p className="text-sm text-slate-500">ピッチデータが見つかりませんでした。</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={pitchPoints}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="time" tickFormatter={(value) => formatSeconds(value as number)} minTickGap={24} />
        <YAxis unit=" Hz" domain={[0, "dataMax"]} width={60} />
        <Tooltip formatter={(value: number) => `${Math.round(value)} Hz`} labelFormatter={(value) => formatSeconds(value as number)} />
        <Legend />
        <Line type="monotone" dataKey="value" name="F0" stroke="#6366f1" dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function LoudnessTempoChart({ segments }: { segments: Segment[] }) {
  const data = useMemo(
    () =>
      segments.map((seg) => ({
        id: seg.id,
        label: `${seg.speaker} ${formatSeconds(seg.start)}`,
        loudness: seg.loudness ?? 0,
        tempo: seg.tempo ?? 0,
      })),
    [segments],
  );

  if (!data.length) {
    return <p className="text-sm text-slate-500">音量/テンポデータが見つかりませんでした。</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="label" hide />
        <YAxis yAxisId="left" domain={[0, 1]} width={32} />
        <YAxis yAxisId="right" orientation="right" width={40} />
        <Tooltip />
        <Legend />
        <Bar dataKey="loudness" yAxisId="left" fill="#0ea5e9" name="音量(RMS)" radius={[6, 6, 0, 0]} />
        <Bar dataKey="tempo" yAxisId="right" fill="#14b8a6" name="テンポ(chars/sec)" radius={[6, 6, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function DialectChart({ segments, keys }: { segments: Segment[]; keys: string[] }) {
  const data = useMemo(
    () =>
      segments.map((seg) => ({
        id: seg.id,
        label: `${seg.speaker} ${formatSeconds(seg.start)}`,
        ...seg.dialect,
      })),
    [segments],
  );

  if (!keys.length) {
    return <p className="text-sm text-slate-500">方言スコアが見つかりませんでした。</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} stackOffset="expand">
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="label" hide />
        <YAxis tickFormatter={(value) => `${Math.round((value ?? 0) * 100)}%`} width={50} />
        <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
        <Legend />
        {keys.map((key, index) => (
          <Bar key={key} dataKey={key} stackId="dialect" fill={`hsl(${index * 72},65%,55%)`} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

function Timeline({ duration, segments }: { duration: number; segments: Segment[] }) {
  if (!segments.length) {
    return <p className="text-sm text-slate-500">セグメントがまだ読み込まれていません。</p>;
  }

  return (
    <div className="relative h-20 rounded-2xl border border-slate-200 bg-white overflow-hidden">
      {segments.map((segment) => {
        const width = ((segment.end - segment.start) / duration) * 100;
        const left = (segment.start / duration) * 100;
        const color = segment.speaker === "A" ? "bg-blue-500/70" : "bg-emerald-500/70";
        return (
          <div key={segment.id} className="absolute inset-y-0 flex flex-col" style={{ width: `${width}%`, left: `${left}%` }}>
            <div className={`${color} h-1.5`} />
            <div className="px-2 pt-1 text-[11px] text-slate-700 truncate">
              <strong className="mr-1 text-slate-900">{segment.speaker}</strong>
              {segment.text}
            </div>
            <span className="mt-auto pb-1 pr-2 self-end text-[10px] text-slate-500">
              {formatSeconds(segment.start)}–{formatSeconds(segment.end)}
            </span>
          </div>
        );
      })}
      <div className="absolute bottom-1 left-2 text-[10px] uppercase tracking-wide text-slate-400">{`Total ${formatSeconds(duration)}`}</div>
    </div>
  );
}

function HighlightList({ segments }: { segments: Segment[] }) {
  const rows = useMemo(() => {
    const items: Array<{ id: string; text: string; tag: string; speaker: string }> = [];
    segments.forEach((segment) => {
      segment.highlights?.forEach((hl) => {
        items.push({
          id: `${segment.id}-${hl.startChar}-${hl.endChar}`,
          text: segment.text.slice(hl.startChar, hl.endChar),
          tag: hl.tag,
          speaker: segment.speaker,
        });
      });
    });
    return items;
  }, [segments]);

  if (!rows.length) {
    return <p className="text-sm text-slate-500">語彙ハイライトは検出されませんでした。</p>;
  }

  return (
    <ul className="space-y-2">
      {rows.map((row) => (
        <li key={row.id} className="flex items-start gap-3 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm">
          <span className="mt-0.5 rounded bg-indigo-100 px-2 py-0.5 text-xs font-semibold text-indigo-700">{row.tag}</span>
          <div>
            <p className="font-medium text-slate-800">{row.text}</p>
            <p className="text-xs text-slate-500">Speaker {row.speaker}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}

function SpeakerSummary({ segments }: { segments: Segment[] }) {
  const stats = useMemo(() => {
    const bySpeaker = new Map<string, { duration: number; count: number; loudness: number; tempo: number }>();
    segments.forEach((segment) => {
      const bucket = bySpeaker.get(segment.speaker) ?? { duration: 0, count: 0, loudness: 0, tempo: 0 };
      bucket.duration += segment.end - segment.start;
      bucket.count += 1;
      bucket.loudness += segment.loudness ?? 0;
      bucket.tempo += segment.tempo ?? 0;
      bySpeaker.set(segment.speaker, bucket);
    });
    return Array.from(bySpeaker.entries()).map(([speaker, value]) => ({
      speaker,
      duration: value.duration,
      count: value.count,
      loudness: value.loudness / Math.max(1, value.count),
      tempo: value.tempo / Math.max(1, value.count),
    }));
  }, [segments]);

  if (!stats.length) {
    return <p className="text-sm text-slate-500">話者統計が計算できませんでした。</p>;
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {stats.map((row) => (
        <div key={row.speaker} className="rounded-xl border border-slate-200 bg-slate-50/60 p-4">
          <p className="text-sm font-semibold text-slate-700">Speaker {row.speaker}</p>
          <dl className="mt-2 space-y-1 text-sm text-slate-600">
            <div className="flex justify-between"><dt>合計時間</dt><dd>{formatSeconds(row.duration)}</dd></div>
            <div className="flex justify-between"><dt>セグメント数</dt><dd>{row.count}</dd></div>
            <div className="flex justify-between"><dt>平均音量</dt><dd>{row.loudness.toFixed(2)}</dd></div>
            <div className="flex justify-between"><dt>平均テンポ</dt><dd>{row.tempo.toFixed(2)} chars/sec</dd></div>
          </dl>
        </div>
      ))}
    </div>
  );
}

function getSpeechRecognitionConstructor(): any {
  if (typeof window === "undefined") return null;
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function getAudioContext(): AudioContext | null {
  if (typeof window === "undefined") return null;
  const Ctor = window.AudioContext || window.webkitAudioContext;
  return Ctor ? new Ctor() : null;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<InputMode>("import");
  const [toggles, setToggles] = useState<FeatureToggle>(initialFeatureToggles);
  const [data, setData] = useState<CorpusFile | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [lastSource, setLastSource] = useState<InputMode | null>(null);

  const [micDrafts, setMicDrafts] = useState<MicDraftSegment[]>([]);
  const micDraftsRef = useRef<MicDraftSegment[]>([]);
  useEffect(() => {
    micDraftsRef.current = micDrafts;
  }, [micDrafts]);

  const [isRecording, setIsRecording] = useState(false);
  const [interimTranscript, setInterimTranscript] = useState<string>("");
  const [micError, setMicError] = useState<string | null>(null);

  const micStateRef = useRef({
    recognition: null as any,
    mediaStream: null as MediaStream | null,
    audioContext: null as AudioContext | null,
    analyser: null as AnalyserNode | null,
    source: null as MediaStreamAudioSourceNode | null,
    rafId: 0,
    amplitudeSamples: [] as number[],
    startTime: 0,
    lastSegmentEnd: 0,
  });

  const speechSupported = typeof window !== "undefined" && Boolean(getSpeechRecognitionConstructor());

  const handleFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = await parseJsonlFile(file);
      setData(parsed);
      setFileName(file.name);
      setError(null);
      setLastSource("import");
    } catch (err) {
      console.error(err);
      setError("JSON/JSONL の読み込みに失敗しました。フォーマットをご確認ください。");
    }
  };

  const startRecording = async () => {
    if (!speechSupported) {
      setMicError("Web Speech API に非対応のブラウザです。Chrome または Edge をご利用ください。");
      return;
    }
    if (isRecording) return;

    const RecognitionCtor = getSpeechRecognitionConstructor();
    if (!RecognitionCtor) {
      setMicError("Web Speech API が利用できません。");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = getAudioContext();
      const analyser = audioContext?.createAnalyser() ?? null;
      let source: MediaStreamAudioSourceNode | null = null;

      if (audioContext && analyser) {
        analyser.fftSize = 2048;
        source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
      }

      const recognition = new RecognitionCtor();
      recognition.lang = "ja-JP";
      recognition.continuous = true;
      recognition.interimResults = true;

      micStateRef.current = {
        recognition,
        mediaStream: stream,
        audioContext: audioContext ?? null,
        analyser,
        source,
        rafId: 0,
        amplitudeSamples: [],
        startTime: performance.now(),
        lastSegmentEnd: 0,
      };

      setMicDrafts([]);
      setInterimTranscript("");
      setMicError(null);
      setError(null);
      setIsRecording(true);

      const buffer = analyser ? new Float32Array(analyser.fftSize) : null;
      const pumpAmplitude = () => {
        const { analyser: currentAnalyser } = micStateRef.current;
        if (!buffer || !currentAnalyser) return;
        currentAnalyser.getFloatTimeDomainData(buffer);
        let sum = 0;
        for (let i = 0; i < buffer.length; i += 1) {
          const value = buffer[i];
          sum += value * value;
        }
        const rms = Math.sqrt(sum / buffer.length);
        const normalized = Math.min(1, rms * 4);
        micStateRef.current.amplitudeSamples.push(Number.isFinite(normalized) ? normalized : 0);
        micStateRef.current.rafId = requestAnimationFrame(pumpAmplitude);
      };
      if (analyser && buffer) {
        pumpAmplitude();
      }

      recognition.onresult = (event: any) => {
        const { resultIndex, results } = event;
        for (let i = resultIndex; i < results.length; i += 1) {
          const result = results[i];
          const transcript = result[0]?.transcript?.trim();
          if (!transcript) continue;

          if (result.isFinal) {
            const now = performance.now();
            const start = micStateRef.current.lastSegmentEnd;
            const end = (now - micStateRef.current.startTime) / 1000;
            const samples = micStateRef.current.amplitudeSamples;
            const rms = samples.length ? samples.reduce((acc, v) => acc + v, 0) / samples.length : 0;
            micStateRef.current.amplitudeSamples = [];
            micStateRef.current.lastSegmentEnd = end;

            const newDraft: MicDraftSegment = {
              id: `mic_${Date.now()}_${micDraftsRef.current.length}`,
              start,
              end,
              text: transcript,
              rms,
            };

            setMicDrafts((prev) => [...prev, newDraft]);
            setInterimTranscript("");
          } else {
            setInterimTranscript(transcript);
          }
        }
      };

      recognition.onerror = (event: any) => {
        console.error("Speech recognition error", event);
        setMicError("音声認識でエラーが発生しました。");
      };

      recognition.onend = () => {
        setIsRecording(false);
      };

      recognition.start();
    } catch (error) {
      console.error(error);
      setMicError("マイクへのアクセスに失敗しました。ブラウザ設定をご確認ください。");
      stopRecording();
    }
  };

  const stopRecording = () => {
    setIsRecording(false);
    const state = micStateRef.current;
    if (state.recognition) {
      try {
        state.recognition.onresult = null;
        state.recognition.stop();
      } catch (error) {
        console.warn("Failed to stop recognition", error);
      }
    }
    if (state.mediaStream) {
      state.mediaStream.getTracks().forEach((track) => track.stop());
    }
    if (state.rafId) {
      cancelAnimationFrame(state.rafId);
    }
    if (state.source) {
      try {
        state.source.disconnect();
      } catch (error) {
        console.warn("Failed to disconnect source", error);
      }
    }
    if (state.audioContext) {
      try {
        state.audioContext.close();
      } catch (error) {
        console.warn("Failed to close audio context", error);
      }
    }
    micStateRef.current = {
      recognition: null,
      mediaStream: null,
      audioContext: null,
      analyser: null,
      source: null,
      rafId: 0,
      amplitudeSamples: [],
      startTime: 0,
      lastSegmentEnd: 0,
    };
  };

  useEffect(() => () => stopRecording(), []);

  const handleAnalyzeMic = () => {
    if (!micDrafts.length) {
      setMicError("録音結果がありません。");
      return;
    }

    const conversationId = `mic_${Math.floor(Date.now() / 1000)}`;
    const analyzerSummary: Record<string, string> = {
      asr: "webspeech",
      diarization: "none",
      emotion: "dummy",
      pitch: "none",
      loudness: "rms",
      tempo: "chars_per_sec",
      dialect: "uniform",
      lexicon: "none",
      language: "ja-JP",
    };

    const segments: Segment[] = micDrafts.map((draft, idx) => {
      const duration = Math.max(0.5, draft.end - draft.start);
      const tempo = draft.text.length / duration;
      return {
        id: `${conversationId}_s${idx.toString().padStart(3, "0")}`,
        conversation_id: conversationId,
        source_file: "mic_capture",
        start: draft.start,
        end: draft.end,
        speaker: "A",
        text: draft.text,
        emotion: { neutral: 1 },
        pitch: [],
        loudness: Number(draft.rms.toFixed(3)),
        tempo: Number(tempo.toFixed(2)),
        dialect: { ...DIALECT_UNIFORM },
        highlights: [],
        analyzer: analyzerSummary,
      };
    });

    const duration = segments.reduce((acc, seg) => Math.max(acc, seg.end), 0);

    setData({ segments, duration, analyzerSummary });
    setLastSource("mic");
    setError(null);
    setMicError(null);
  };

  const handleDownloadMicJson = () => {
    if (!data || lastSource !== "mic") return;
    downloadJson("segments.json", {
      segments: data.segments,
      duration: data.duration,
      analyzerSummary: data.analyzerSummary,
    });
  };

  const segments = data?.segments ?? [];
  const duration = data?.duration ?? 0;
  const emotionKeys = useEmotionKeys(segments);
  const dialectKeys = useDialectKeys(segments);

  const renderFeatureToggles = data ? (
    <div className="flex flex-wrap gap-2 text-xs text-slate-500">
      {featureKeys.map((key) => (
        <button
          key={key}
          onClick={() => setToggles((prev) => ({ ...prev, [key]: !prev[key] }))}
          className={`rounded-full border px-3 py-1 transition ${
            toggles[key] ? "border-slate-700 bg-slate-800 text-white" : "border-slate-300 bg-white text-slate-500"
          }`}
        >
          {key}
        </button>
      ))}
    </div>
  ) : null;

  const micTabDisabled = !speechSupported;
  const micInstructions = "Chrome / Edge でお試しください。HTTPS 環境が必須です。";

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-100 via-white to-slate-200 pb-20">
      <header className="mx-auto flex max-w-6xl flex-col gap-6 px-6 pt-10">
        <div className="flex items-center gap-3 text-slate-500">
          <BarChart3 size={18} />
          <span className="text-sm uppercase tracking-[0.2em] text-slate-500">CorpusVisualize</span>
        </div>
        <div className="flex flex-col gap-3">
          <h1 className="text-3xl font-bold text-slate-800 sm:text-4xl">会話コーパスを多面的に可視化するアナリティクスビュー</h1>
          <p className="max-w-3xl text-base text-slate-600">
            パイプラインが出力した `segments.jsonl` を読み込むか、Web Speech API で録音した音声を簡易解析して感情・韻律・語彙など複数の特徴量を一画面で俯瞰できます。
            Whisper Medium やその他モジュールの設定値も自動で反映します。
          </p>
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setActiveTab("import")}
            className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition ${
              activeTab === "import"
                ? "bg-slate-900 text-white shadow"
                : "bg-white text-slate-600 border border-slate-200"
            }`}
          >
            <Upload size={16} /> Import JSON
          </button>
          <button
            onClick={() => !micTabDisabled && setActiveTab("mic")}
            className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition ${
              activeTab === "mic"
                ? "bg-indigo-600 text-white shadow"
                : "bg-white text-slate-600 border border-slate-200"
            } ${micTabDisabled ? "opacity-50 cursor-not-allowed" : ""}`}
          >
            <Mic size={16} /> Mic (Web Speech API)
          </button>
        </div>

        {activeTab === "import" ? (
          <div className="space-y-3">
            <label
              htmlFor="jsonl-upload"
              className="flex w-full max-w-xl cursor-pointer flex-col gap-3 rounded-2xl border-2 border-dashed border-slate-300 bg-white/80 px-6 py-5 text-slate-600 transition hover:border-slate-400"
            >
              <div className="flex items-center gap-3">
                <Upload size={20} className="text-slate-500" />
                <div>
                  <p className="text-sm font-semibold">`segments.jsonl / segments.json` をアップロード</p>
                  <p className="text-xs text-slate-500">CorpusVisualize pipeline の出力をドラッグ＆ドロップ、またはクリックで選択</p>
                  {fileName ? <p className="mt-1 text-xs text-slate-500">読み込み済み: {fileName}</p> : null}
                </div>
              </div>
              <input id="jsonl-upload" type="file" accept=".jsonl,.json" className="hidden" onChange={handleFile} />
            </label>
            {error ? <p className="text-sm text-rose-600">{error}</p> : null}
          </div>
        ) : (
          <div className="space-y-4 rounded-2xl border border-indigo-200 bg-white/80 p-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-semibold text-slate-700">ブラウザ内で音声認識（ja-JP）</p>
                <p className="text-xs text-slate-500">{micInstructions}</p>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={startRecording}
                  disabled={isRecording || micTabDisabled}
                  className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition ${
                    isRecording ? "bg-slate-200 text-slate-500 cursor-not-allowed" : "bg-indigo-600 text-white shadow"
                  }`}
                >
                  <PlayCircle size={16} /> 録音開始
                </button>
                <button
                  onClick={stopRecording}
                  disabled={!isRecording}
                  className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition ${
                    isRecording ? "bg-rose-500 text-white shadow" : "bg-white text-slate-500 border border-slate-200"
                  }`}
                >
                  <Square size={16} /> 停止
                </button>
              </div>
            </div>

            {micError ? <p className="text-sm text-rose-600">{micError}</p> : null}
            {isRecording ? <p className="text-sm text-indigo-600">録音中…発話が終わったら停止してください。</p> : null}
            {interimTranscript ? (
              <p className="rounded-xl border border-indigo-200 bg-indigo-50 px-3 py-2 text-sm text-indigo-700">
                {interimTranscript}
              </p>
            ) : null}

            <div className="space-y-2">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Segments (draft)</p>
              <div className="max-h-48 overflow-y-auto rounded-xl border border-slate-200 bg-slate-50">
                {micDrafts.length ? (
                  <ul className="divide-y divide-slate-200 text-sm">
                    {micDrafts.map((draft, idx) => (
                      <li key={draft.id} className="flex items-start gap-3 px-3 py-2">
                        <span className="rounded bg-slate-800 px-2 py-0.5 text-xs font-semibold text-white">{idx + 1}</span>
                        <div>
                          <p className="font-medium text-slate-800">{draft.text}</p>
                          <p className="text-xs text-slate-500">
                            {formatSeconds(draft.start)}–{formatSeconds(draft.end)} ・ 推定音量 {draft.rms.toFixed(2)}
                          </p>
                        </div>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="px-3 py-4 text-sm text-slate-500">まだセグメントはありません。録音してみてください。</p>
                )}
              </div>
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                onClick={handleAnalyzeMic}
                disabled={micDrafts.length === 0}
                className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition ${
                  micDrafts.length === 0 ? "bg-slate-200 text-slate-500 cursor-not-allowed" : "bg-emerald-600 text-white shadow"
                }`}
              >
                <Wand2 size={16} /> 解析を実行
              </button>
              <button
                onClick={handleDownloadMicJson}
                disabled={!data || lastSource !== "mic"}
                className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition ${
                  !data || lastSource !== "mic"
                    ? "bg-slate-200 text-slate-500 cursor-not-allowed"
                    : "bg-white text-slate-600 border border-slate-200 hover:border-slate-300"
                }`}
              >
                <Download size={16} /> JSON ダウンロード
              </button>
            </div>
          </div>
        )}
      </header>

      <main className="mx-auto mt-10 flex max-w-6xl flex-col gap-6 px-6">
        {renderFeatureToggles}

        {data ? (
          <>
            <Card
              title="タイムライン"
              icon={<FileBarChart size={18} className="text-blue-500" />}
              actions={<span className="text-xs text-slate-500">合計 {formatSeconds(duration)} ・ {segments.length} セグメント</span>}
            >
              <Timeline duration={duration} segments={segments} />
            </Card>

            <div className="grid gap-6 lg:grid-cols-2">
              {toggles.emotion && (
                <Card title="感情分布" icon={<Gauge size={18} className="text-rose-500" />}>
                  <EmotionAreaChart segments={segments} keys={emotionKeys} />
                </Card>
              )}

              {toggles.pitch && (
                <Card title="ピッチトラッキング" icon={<Wand2 size={18} className="text-indigo-500" />}>
                  <PitchLineChart segments={segments} />
                </Card>
              )}

              {(toggles.loudness || toggles.tempo) && (
                <Card title="音量 / テンポ" icon={<BarChart3 size={18} className="text-sky-500" />}>
                  <LoudnessTempoChart segments={segments} />
                </Card>
              )}

              {toggles.dialect && (
                <Card title="方言スコア" icon={<Languages size={18} className="text-emerald-500" />}>
                  <DialectChart segments={segments} keys={dialectKeys} />
                </Card>
              )}
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              {toggles.lexicon && (
                <Card title="語彙ハイライト" icon={<Sparkles size={18} className="text-amber-500" />}>
                  <HighlightList segments={segments} />
                </Card>
              )}
              <Card title="話者別統計" icon={<BarChart3 size={18} className="text-slate-500" />}>
                <SpeakerSummary segments={segments} />
              </Card>
            </div>

            <Card title="設定スナップショット" icon={<Settings2 size={18} className="text-slate-500" />}>
              <div className="flex flex-wrap gap-2">
                {moduleOrder.map((moduleKey) => (
                  <ModuleBadge key={moduleKey} label={moduleKey} value={data.analyzerSummary[moduleKey]} />
                ))}
              </div>
            </Card>
          </>
        ) : (
          <Card title="サンプルビュー" icon={<Wand2 size={18} className="text-indigo-500" />}>
            <p className="text-sm text-slate-500">
              まだデータが読み込まれていません。インポートまたはマイク録音から解析を実行すると、チャートが表示されます。
            </p>
          </Card>
        )}
      </main>
    </div>
  );
}
