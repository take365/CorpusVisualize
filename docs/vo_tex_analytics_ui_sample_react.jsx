import React, { useMemo, useRef, useState } from "react";
import { Upload, Settings2, Wand2, Play, Pause } from "lucide-react";
import {
  Line,
  LineChart,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  CartesianGrid,
  Legend,
  AreaChart,
  Area,
} from "recharts";

// -------------------------------------------------------------
// VoTexAnalytics — Conversation Visualization Demo UI
// Single-file React component. Paste into a Vite React-TS project.
// Requires: tailwindcss, recharts, lucide-react
// -------------------------------------------------------------

// Types
type Segment = {
  id: string;
  start: number; // sec
  end: number;   // sec
  speaker: string; // "A" | "B"
  text: string;
  emotion?: { neutral: number; joy: number; anger: number; sad: number };
  pitch?: number[]; // simplified F0 curve per second inside segment
  loudness?: number; // RMS (0-1)
  tempo?: number;    // chars/sec (mock)
  dialect?: { kansai: number; kanto: number; tohoku: number; kyushu: number; hokkaido: number };
  highlights?: Array<{ startChar: number; endChar: number; tag: string }>; // for text highlight
};

type Toggles = {
  showEmotion: boolean;
  showPitch: boolean;
  showLoudness: boolean;
  showTempo: boolean;
  showDialect: boolean;
  showLexicon: boolean;
};

const defaultToggles: Toggles = {
  showEmotion: true,
  showPitch: true,
  showLoudness: true,
  showTempo: true,
  showDialect: true,
  showLexicon: true,
};

// --- Method selections for each analysis module ---
type MethodSelections = {
  emotion: "ser_w2v" | "ser_kushinada" | "ser_fusion";
  pitch: "praat" | "pyworld" | "jtobi_rule";
  loudness: "rms" | "lufs" | "a_weighted";
  tempo: "asr_chars" | "syllable_rate" | "vad_ratio";
  dialect: "lexicon_tfidf" | "llm_rag" | "audio_prosody";
  lexicon: "pos_dict" | "formality" | "valence";
};

const defaultMethods: MethodSelections = {
  emotion: "ser_w2v",
  pitch: "praat",
  loudness: "rms",
  tempo: "asr_chars",
  dialect: "lexicon_tfidf",
  lexicon: "pos_dict",
};

// Utility
function seededRandom(seed: number) {
  return function () {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };
}

function secondsToClock(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60).toString().padStart(2, "0");
  return `${m}:${sec}`;
}

// Mock processing: generate 8 segments across ~80 sec for two speakers
function mockProcessFile(fileName: string): { duration: number; segments: Segment[] } {
  const rand = seededRandom(fileName.length);
  const segs: Segment[] = [];
  let t = 0;
  for (let i = 0; i < 8; i++) {
    const dur = 6 + Math.floor(rand() * 6); // 6-11s
    const speaker = i % 2 === 0 ? "A" : "B";
    const textPoolA = [
      "なるほど、それは面白いですね。",
      "そうなんですね、ありがとうございます。",
      "ちなみに、次はどうしましょうか。",
    ];
    const textPoolB = [
      "えっと、こちらの案ですが…",
      "はい、では進めてみます。",
      "なるほど、もう少し検討します。",
    ];
    const text = speaker === "A" ? textPoolA[Math.floor(rand() * textPoolA.length)] : textPoolB[Math.floor(rand() * textPoolB.length)];

    const joy = Math.round(rand() * 60) / 100;
    const anger = Math.round(rand() * 30) / 100;
    const sad = Math.round(rand() * 25) / 100;
    const neutral = Math.max(0, 1 - joy - anger - sad);

    const loud = 0.4 + rand() * 0.5;
    const tempo = 3 + rand() * 4;

    const dists = [rand(), rand(), rand(), rand(), rand()];
    const sum = dists.reduce((a, b) => a + b, 0);

    const pitchLen = dur;
    const base = speaker === "A" ? 210 : 185;
    const pitch = Array.from({ length: pitchLen }, (_, k) => base + Math.sin(k / 1.5) * (8 + rand() * 10) + rand() * 5);

    const highlights = Math.random() < 0.5 ? [{ startChar: 0, endChar: Math.min(6, text.length), tag: "dialect-ish" }] : [];

    segs.push({
      id: `s${i}`,
      start: t,
      end: t + dur,
      speaker,
      text,
      emotion: { neutral, joy, anger, sad },
      loudness: Number(loud.toFixed(2)),
      tempo: Number(tempo.toFixed(2)),
      pitch,
      dialect: {
        kansai: dists[0] / sum,
        kanto: dists[1] / sum,
        tohoku: dists[2] / sum,
        kyushu: dists[3] / sum,
        hokkaido: dists[4] / sum,
      },
      highlights,
    });

    t += dur + (1 + Math.floor(rand() * 2)); // add a small pause 1-2s
  }

  return { duration: t, segments: segs };
}

const Tag: React.FC<{ label: string; color?: string }> = ({ label, color = "bg-indigo-100 text-indigo-700" }) => (
  <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${color}`}>{label}</span>
);

const Card: React.FC<{ title: string; actions?: React.ReactNode; children: React.ReactNode }> = ({ title, actions, children }) => (
  <div className="bg-white/70 backdrop-blur border border-gray-200 rounded-2xl p-4 shadow-sm">
    <div className="flex items-center justify-between mb-3">
      <h3 className="text-sm font-semibold text-gray-700">{title}</h3>
      {actions}
    </div>
    {children}
  </div>
);

const ToggleRow: React.FC<{ toggles: Toggles; setToggles: React.Dispatch<React.SetStateAction<Toggles>> }> = ({ toggles, setToggles }) => (
  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
    {([
      ["showEmotion", "感情"],
      ["showPitch", "アクセント/ピッチ"],
      ["showLoudness", "音量"],
      ["showTempo", "テンポ"],
      ["showDialect", "訛り（方言）"],
      ["showLexicon", "語彙（品位/語感）"],
    ] as const).map(([key, label]) => (
      <label key={key} className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={toggles[key]}
          onChange={(e) => setToggles((t) => ({ ...t, [key]: e.target.checked }))}
        />
        <span>{label}</span>
      </label>
    ))}
  </div>
);

const WaveTimeline: React.FC<{ duration: number; segments: Segment[] }> = ({ duration, segments }) => {
  return (
    <div className="w-full h-16 relative rounded-xl bg-gradient-to-r from-slate-50 to-slate-100 border border-slate-200 overflow-hidden">
      {segments.map((s) => {
        const left = (s.start / duration) * 100;
        const width = ((s.end - s.start) / duration) * 100;
        const color = s.speaker === "A" ? "bg-sky-400/60" : "bg-emerald-400/60";
        return (
          <div key={s.id} className="absolute top-0 h-full" style={{ left: `${left}%`, width: `${width}%` }}>
            <div className={`h-1.5 ${color}`}></div>
            <div className="px-2 pt-2 text-xs text-slate-700 truncate">{s.speaker}: {s.text}</div>
            <div className="absolute bottom-1 right-2 text-[10px] text-slate-500">{secondsToClock(s.start)}–{secondsToClock(s.end)}</div>
          </div>
        );
      })}
    </div>
  );
};

const EmotionChart: React.FC<{ segments: Segment[] }> = ({ segments }) => {
  const data = segments.map((s) => ({
    name: s.id,
    start: s.start,
    joy: s.emotion?.joy ?? 0,
    anger: s.emotion?.anger ?? 0,
    sad: s.emotion?.sad ?? 0,
    neutral: s.emotion?.neutral ?? 0,
  }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={data} margin={{ left: 8, right: 8, top: 10, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="start" tickFormatter={(v) => `${Math.round(v)}s`} />
        <YAxis domain={[0, 1]} />
        <Tooltip formatter={(v: any) => (typeof v === "number" ? v.toFixed(2) : v)} />
        <Legend />
        <Area type="monotone" dataKey="joy" stackId="1" fillOpacity={0.35} />
        <Area type="monotone" dataKey="anger" stackId="1" fillOpacity={0.35} />
        <Area type="monotone" dataKey="sad" stackId="1" fillOpacity={0.35} />
        <Area type="monotone" dataKey="neutral" stackId="1" fillOpacity={0.15} />
      </AreaChart>
    </ResponsiveContainer>
  );
};

const PitchChart: React.FC<{ segments: Segment[] }> = ({ segments }) => {
  // Build a simple time series across segments
  const series = useMemo(() => {
    const arr: { t: number; f0: number; spk: string }[] = [];
    segments.forEach((s) => {
      const step = (s.end - s.start) / (s.pitch?.length || 1);
      s.pitch?.forEach((f0, i) => {
        arr.push({ t: s.start + i * step, f0, spk: s.speaker });
      });
    });
    return arr;
  }, [segments]);

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={series} margin={{ left: 8, right: 8, top: 10, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="t" tickFormatter={(v) => `${Math.round(v)}s`} />
        <YAxis domain={[120, 280]} tickFormatter={(v) => `${v}Hz`} />
        <Tooltip formatter={(v: any) => (typeof v === "number" ? `${Math.round(v)} Hz` : v)} />
        <Legend />
        <Line type="monotone" dataKey="f0" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
};

const LoudTempoChart: React.FC<{ segments: Segment[] }> = ({ segments }) => {
  const data = segments.map((s) => ({ name: `${s.speaker}-${s.id}`, loud: s.loudness ?? 0, tempo: s.tempo ?? 0 }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ left: 8, right: 8, top: 10, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" hide />
        <YAxis yAxisId="left" domain={[0, 1]} />
        <YAxis yAxisId="right" orientation="right" />
        <Tooltip />
        <Legend />
        <Bar yAxisId="left" dataKey="loud" fillOpacity={0.5} />
        <Bar yAxisId="right" dataKey="tempo" fillOpacity={0.5} />
      </BarChart>
    </ResponsiveContainer>
  );
};

const DialectBars: React.FC<{ segments: Segment[] }> = ({ segments }) => {
  const data = segments.map((s) => ({
    name: s.id,
    kansai: s.dialect?.kansai ?? 0,
    kanto: s.dialect?.kanto ?? 0,
    tohoku: s.dialect?.tohoku ?? 0,
    kyushu: s.dialect?.kyushu ?? 0,
    hokkaido: s.dialect?.hokkaido ?? 0,
  }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ left: 8, right: 8, top: 10, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis domain={[0, 1]} />
        <Tooltip />
        <Legend />
        <Bar dataKey="kansai" stackId="a" fillOpacity={0.6} />
        <Bar dataKey="kanto" stackId="a" fillOpacity={0.6} />
        <Bar dataKey="tohoku" stackId="a" fillOpacity={0.6} />
        <Bar dataKey="kyushu" stackId="a" fillOpacity={0.6} />
        <Bar dataKey="hokkaido" stackId="a" fillOpacity={0.6} />
      </BarChart>
    </ResponsiveContainer>
  );
};

const LexiconList: React.FC<{ segments: Segment[] }> = ({ segments }) => (
  <div className="space-y-2">
    {segments.map((s) => (
      <div key={s.id} className="text-sm p-2 rounded-lg bg-slate-50 border border-slate-200">
        <div className="flex items-center justify-between">
          <span className="font-medium text-slate-700">{s.speaker} — Seg {s.id}</span>
          <div className="flex gap-1">
            {s.highlights?.map((h, i) => (
              <Tag key={i} label={h.tag} />
            ))}
          </div>
        </div>
        <p className="mt-1">
          {s.text.split("").map((ch, idx) => {
            const hit = s.highlights?.some((h) => idx >= h.startChar && idx < h.endChar);
            return (
              <span key={idx} className={hit ? "bg-yellow-200/70 rounded px-0.5" : ""}>{ch}</span>
            );
          })}
        </p>
      </div>
    ))}
  </div>
);

const VoTexConversationViz: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [toggles, setToggles] = useState<Toggles>(defaultToggles);
  const [methods, setMethods] = useState<MethodSelections>(defaultMethods);
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const MethodSelect: React.FC<{ value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }> = ({ value, onChange, options }) => (
    <select
      className="border rounded-lg px-2 py-1 text-xs bg-white"
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  );

  const data = useMemo(() => (file ? mockProcessFile(file.name) : null), [file]);

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-white to-slate-50 text-slate-900">
      <header className="max-w-6xl mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold">VoTexAnalytics — Conversation Visualizer</h1>
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <Settings2 className="w-4 h-4" /> サンプルUI
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 pb-24 space-y-8">
        {/* Uploader & Settings */}
        <Card
          title="入力（会話音声ファイル）＆ 表示設定"
          actions={<Tag label="Demo" color="bg-emerald-100 text-emerald-700" />}
        >
          <div className="flex flex-col md:flex-row gap-4 md:items-center">
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              <span className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-slate-900 text-white text-sm">
                <Upload className="w-4 h-4" /> 音声ファイルを選択
              </span>
              {file && <span className="text-slate-600 text-sm">{file.name}</span>}
            </label>

            <div className="flex items-center gap-2 ml-auto">
              <button
                className="inline-flex items-center gap-2 px-3 py-2 rounded-xl border text-sm"
                onClick={() => setPlaying((p) => !p)}
              >
                {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />} 再生
              </button>
              <button className="inline-flex items-center gap-2 px-3 py-2 rounded-xl border text-sm">
                <Wand2 className="w-4 h-4" /> サンプル生成
              </button>
            </div>
          </div>

          <div className="mt-4">
            <ToggleRow toggles={toggles} setToggles={setToggles} />
          </div>
        </Card>

        {/* Timeline */}
        <Card title="会話タイムライン（話者分離＆発話）">
          {data ? (
            <div className="space-y-2">
              <WaveTimeline duration={data.duration} segments={data.segments} />
              <div className="flex justify-end text-xs text-slate-500">合計 {Math.round(data.duration)} 秒</div>
            </div>
          ) : (
            <div className="text-sm text-slate-500">音声を選択すると、話者ごとの発話区間がここに表示されます。</div>
          )}
        </Card>

        {/* Visual Blocks */}
        {data && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {toggles.showEmotion && (
              <Card title="感情（区間ごとの分布）" actions={
              <MethodSelect
                value={methods.emotion}
                onChange={(v) => setMethods((m) => ({ ...m, emotion: v as MethodSelections["emotion"] }))}
                options={[
                  { value: "ser_w2v", label: "SER: wav2vec2" },
                  { value: "ser_kushinada", label: "SER: くしなだ" },
                  { value: "ser_fusion", label: "SER: Fusion" },
                ]}
              />
            }>
                <EmotionChart segments={data.segments} />
              </Card>
            )}

            {toggles.showPitch && (
              <Card title="アクセント / ピッチ曲線（簡易）" actions={
              <MethodSelect
                value={methods.pitch}
                onChange={(v) => setMethods((m) => ({ ...m, pitch: v as MethodSelections["pitch"] }))}
                options={[
                  { value: "praat", label: "Praat/Parselmouth" },
                  { value: "pyworld", label: "pyWORLD" },
                  { value: "jtobi_rule", label: "J-ToBI rule" },
                ]}
              />
            }>
                <PitchChart segments={data.segments} />
              </Card>
            )}

            {toggles.showLoudness && (
              <Card title="音量（RMS）とテンポ（chars/sec）" actions={
              <div className="flex gap-2">
                <MethodSelect
                  value={methods.loudness}
                  onChange={(v) => setMethods((m) => ({ ...m, loudness: v as MethodSelections["loudness"] }))}
                  options={[
                    { value: "rms", label: "RMS" },
                    { value: "lufs", label: "LUFS(R128)" },
                    { value: "a_weighted", label: "A-weighted" },
                  ]}
                />
                <MethodSelect
                  value={methods.tempo}
                  onChange={(v) => setMethods((m) => ({ ...m, tempo: v as MethodSelections["tempo"] }))}
                  options={[
                    { value: "asr_chars", label: "ASR chars/sec" },
                    { value: "syllable_rate", label: "Syllable rate" },
                    { value: "vad_ratio", label: "VAD ratio" },
                  ]}
                />
              </div>
            }>
                <LoudTempoChart segments={data.segments} />
              </Card>
            )}

            {toggles.showDialect && (
              <Card title="訛り（方言）らしさの積み上げ" actions={
              <MethodSelect
                value={methods.dialect}
                onChange={(v) => setMethods((m) => ({ ...m, dialect: v as MethodSelections["dialect"] }))}
                options={[
                  { value: "lexicon_tfidf", label: "Lexicon TF-IDF" },
                  { value: "llm_rag", label: "LLM+RAG" },
                  { value: "audio_prosody", label: "Audio prosody" },
                ]}
              />
            }>
                <DialectBars segments={data.segments} />
              </Card>
            )}

            {toggles.showLexicon && (
              <Card title="語彙・表現（ハイライト）" actions={
              <MethodSelect
                value={methods.lexicon}
                onChange={(v) => setMethods((m) => ({ ...m, lexicon: v as MethodSelections["lexicon"] }))}
                options={[
                  { value: "pos_dict", label: "POS+辞書" },
                  { value: "formality", label: "品位/文体レベル" },
                  { value: "valence", label: "語感(Valence)" },
                ]}
              />
            }>
                <LexiconList segments={data.segments} />
              </Card>
            )}
          </div>
        )}

        {/* Developer Self-Tests */}
        {data && (
          <Card title="Developer Self-Tests">
            <ul className="list-disc pl-5 text-sm text-slate-600 space-y-1">
              <li>segments length: <b>{data.segments.length}</b></li>
              <li>has emotion on all: <b>{data.segments.every(s => !!s.emotion).toString()}</b></li>
              <li>dialect probs sum≈1 (first): <b>{Object.values(data.segments[0].dialect ?? {}).reduce((a,b)=>a+b,0).toFixed(3)}</b></li>
              <li>pitch curve present (first): <b>{(data.segments[0].pitch?.length ?? 0) > 0 ? "yes" : "no"}</b></li>
            </ul>
          </Card>
        )}
      </main>

      <audio ref={audioRef} className="hidden" />
    </div>
  );
};

export default VoTexConversationViz;
