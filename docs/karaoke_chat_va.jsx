import React from "react";

/**
 * KaraokeChatVA — Chat-only demo that visualizes Valence–Arousal (color),
 * Loudness (font size), Tempo (letter spacing), and Pitch (vertical offset)
 * on a chat-style transcript. Includes a simple play/pause timeline that
 * highlights the current word like karaoke.
 *
 * Tech: React + TailwindCSS (no external deps)
 */

// -----------------------------
// Types
// -----------------------------
export type Word = {
  t0: number; // start time (sec)
  t1: number; // end time (sec)
  text: string;
  f0?: number; // pitch (Hz)
  loud?: number; // 0..1
  tempo?: number; // chars/sec for this word window
  valence?: number; // -1..+1
  arousal?: number; // -1..+1
};

export type Utterance = {
  id: string;
  speaker: "A" | "B";
  words: Word[];
  t0: number; // utterance start sec
  t1: number; // utterance end sec
};

// -----------------------------
// Mock conversation (2 speakers, ~20s) — stronger per-word variation for demo
// -----------------------------
const convo: Utterance[] = (() => {
  const mk = (
    id: string,
    speaker: "A" | "B",
    t0: number,
    text: string,
    va: [number, number], // base valence, arousal per utterance
    baseF0: number, // speaker median pitch
    baseLoud: number, // 0..1
    baseTempo: number // chars/sec
  ): Utterance => {
    const tokens = text.split(/\s+/).filter(Boolean);
    const dur = Math.max(1.5, tokens.join(" ").length / Math.max(baseTempo, 2));
    const t1 = t0 + dur;
    const words: Word[] = [];
    let cur = t0;

    tokens.forEach((tok, i) => {
      const wDur = dur / tokens.length; // demo: equal split
      const phase = i / Math.max(1, tokens.length - 1);
      const jitter = (Math.random() - 0.5);

      // Pitch: wider swing + speaker bias
      const f0 = baseF0 + Math.sin(i / 0.8) * 20 + (speaker === "A" ? 14 : -14) + jitter * 8;

      // Loudness: pulsating envelope + jitter
      const loud = clamp(baseLoud + 0.35 * Math.sin(phase * 2 * Math.PI) + 0.15 * jitter, 0, 1);

      // Tempo: speaker style + envelope (fast A, calmer B)
      const tempo = clamp(
        baseTempo + (speaker === "A" ? 2 : -1) + 4 * Math.sin(phase * 2 * Math.PI + (speaker === "A" ? 0 : Math.PI)) + 1.5 * jitter,
        2, 12
      );

      // Valence/Arousal: orbit around base with envelope
      const valence = clamp(va[0] + 0.5 * Math.sin(phase * 2 * Math.PI + 0.7) + 0.25 * jitter, -1, 1);
      const arousal = clamp(va[1] + 0.6 * Math.cos(phase * 2 * Math.PI - 0.4) + 0.25 * jitter, -1, 1);

      words.push({
        t0: cur,
        t1: cur + wDur,
        text: tok,
        f0,
        loud,
        tempo,
        valence,
        arousal,
      });
      cur += wDur;
    });
    return { id, speaker, words, t0, t1 };
  };

  // A: lively/excited, B: calmer — then roles flip
  const u1 = mk(
    "u1",
    "A",
    0,
    "いや これは ほんまに 面白い ですね ちょっと 早めに 進めたい です",
    [0.55, 0.65],
    215,
    0.75,
    8.5
  );

  const u2 = mk(
    "u2",
    "B",
    u1.t1 + 0.4,
    "落ち着いて まずは 手順を 確認 しましょう それから 次の ステップに 行きます",
    [0.15, 0.25],
    185,
    0.35,
    5
  );

  const u3 = mk(
    "u3",
    "A",
    u2.t1 + 0.5,
    "了解です じゃあ この案を 試して もし 無理なら 代替を 出します",
    [0.35, 0.45],
    215,
    0.6,
    7.5
  );

  const u4 = mk(
    "u4",
    "B",
    u3.t1 + 0.5,
    "いいですね では 今日の 夕方までに まとめて 共有 します",
    [0.3, 0.35],
    185,
    0.5,
    6.5
  );

  return [u1, u2, u3, u4];
})();

// Speaker median F0 for relative offset (could be computed from data)
const speakerMedianF0: Record<string, number> = { A: 205, B: 190 };

// -----------------------------
// Mapping functions
// -----------------------------
function clamp(n: number, a: number, b: number) {
  return Math.max(a, Math.min(b, n));
}

// Map (valence, arousal) → HSL color string
function vaToHsl(val = 0, aro = 0) {
  const angle = Math.atan2(aro, val); // -PI..PI → direction
  const hue = (angle * 180 / Math.PI + 360) % 360; // 0..360
  const mag = clamp(Math.hypot(val, aro), 0, 1); // 0..1 → intensity
  const sat = 35 + 50 * mag; // 35..85
  const light = 50; // keep constant for legibility
  return `hsl(${hue} ${sat}% ${light}%)`;
}

// Relative F0 (Hz) to vertical offset (px)
function f0ToY(f0: number | undefined, speaker: "A" | "B") {
  if (!f0) return 0;
  const base = speakerMedianF0[speaker];
  const rel = clamp(f0 - base, -120, 120); // widen window for demo
  return (rel / 120) * 10; // map to -10..+10 px (more visible)
}

// Loudness (0..1) → font-size (px)
function loudToFont(loud = 0.5) {
  // 13..28px for a stronger visual swing
  return 13 + 15 * clamp(loud, 0, 1);
}

// Tempo (chars/sec) → letter-spacing (em)
function tempoToLS(tempo = 4) {
  // Map tempo (2..12 chars/sec) to letter-spacing (slow→wide, fast→tight)
  const t = clamp(tempo, 2, 12);
  const ls = 0.35 - (t - 2) * (0.33 / 10); // 0.35..0.02
  return `${clamp(ls, 0.02, 0.35)}em`;
}

// -----------------------------
// Components
// -----------------------------
const WordSpan: React.FC<{ w: Word; active: boolean; speaker: "A" | "B" }>
  = ({ w, active, speaker }) => {
    const color = vaToHsl(w.valence ?? 0, w.arousal ?? 0);
    const y = f0ToY(w.f0, speaker);
    const fs = loudToFont(w.loud);
    const ls = tempoToLS(w.tempo);

    return (
      <span
        className={`inline-block transition-[transform,color] duration-100 will-change-transform ${active ? "font-semibold underline" : ""}`}
        style={{
          color,
          transform: `translateY(${y}px)`,
          fontSize: `${fs}px`,
          letterSpacing: ls,
        }}
      >
        {w.text}
        <span> </span>
      </span>
    );
  };

const Bubble: React.FC<{ u: Utterance; now: number }>
  = ({ u, now }) => {
    const isA = u.speaker === "A";
    return (
      <div className={`max-w-[72%] rounded-2xl px-4 py-3 mb-3 shadow-sm ${isA ? "self-start bg-sky-50 border border-sky-100" : "self-end bg-emerald-50 border border-emerald-100"}`}>
        {u.words.map((w, i) => {
          const active = now >= w.t0 && now < w.t1;
          return <WordSpan key={i} w={w} active={active} speaker={u.speaker} />;
        })}
      </div>
    );
  };

// Transport controls (no audio, time simulated)
const Transport: React.FC<{ now: number; setNow: (t: number) => void; maxT: number; playing: boolean; setPlaying: (b: boolean) => void; speed: number; setSpeed: (x: number) => void; }>
  = ({ now, setNow, maxT, playing, setPlaying, speed, setSpeed }) => {
    return (
      <div className="flex items-center gap-3 py-2 px-3 bg-white/80 rounded-xl border">
        <button onClick={() => setPlaying(!playing)} className="px-3 py-1 rounded-lg bg-black text-white text-sm">
          {playing ? "Pause" : "Play"}
        </button>
        <input type="range" min={0} max={maxT} step={0.05}
               value={now}
               onChange={(e) => setNow(parseFloat(e.target.value))}
               className="w-64"/>
        <div className="text-xs w-20 tabular-nums text-right">{now.toFixed(2)}s</div>
        <label className="text-xs ml-2">Speed</label>
        <select className="text-xs border rounded px-2 py-1"
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}>
          <option value={0.75}>0.75×</option>
          <option value={1}>1.0×</option>
          <option value={1.25}>1.25×</option>
          <option value={1.5}>1.5×</option>
        </select>
      </div>
    );
  };

// Root demo component
const KaraokeChatVA: React.FC = () => {
  const maxT = convo[convo.length - 1].t1 + 0.25;
  const [now, setNow] = React.useState(0);
  const [playing, setPlaying] = React.useState(true);
  const [speed, setSpeed] = React.useState(1);

  React.useEffect(() => {
    let id = 0; let last = performance.now();
    const tick = () => {
      const t = performance.now();
      const dt = (t - last) / 1000 * speed;
      last = t;
      if (playing) {
        setNow((cur) => {
          const n = cur + dt;
          if (n >= maxT) return 0; // loop
          return n;
        });
      }
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [playing, speed, maxT]);

  // Auto-scroll to the currently active utterance
  const containerRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const activeIdx = convo.findIndex(u => now >= u.t0 && now < u.t1);
    const el = containerRef.current?.querySelector(`#utt-${activeIdx}`);
    if (el && el instanceof HTMLElement) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [now]);

  return (
    <div className="w-full h-full p-4 bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="max-w-3xl mx-auto flex flex-col gap-4">
        <h1 className="text-xl font-bold">Chat Visualization — Valence / Arousal × Loudness × Tempo × Pitch</h1>
        <Transport now={now} setNow={setNow} maxT={maxT} playing={playing} setPlaying={setPlaying} speed={speed} setSpeed={setSpeed} />
        <div ref={containerRef} className="h-[60vh] overflow-auto rounded-2xl p-4 bg-white border">
          {convo.map((u, i) => (
            <div id={`utt-${i}`} key={u.id} className="flex flex-col">
              <div className={`text-[10px] uppercase tracking-wide mb-1 ${u.speaker === "A" ? "text-sky-600" : "text-emerald-600"}`}>Speaker {u.speaker}</div>
              <Bubble u={u} now={now} />
            </div>
          ))}
        </div>
        <p className="text-xs text-slate-500">
          色＝Valence/Arousal（方向と強さ）、サイズ＝音量、字間＝テンポ、上下＝ピッチ。現在語は<strong>太字＋下線</strong>。
        </p>
      </div>
    </div>
  );
};

export default KaraokeChatVA;

// -----------------------------
// Dev micro-tests (do not affect UI) — run once on module load
// -----------------------------
(function devTests() {
  try {
    // clamp
    console.assert(clamp(-2, 0, 1) === 0, "clamp low bound");
    console.assert(clamp(2, 0, 1) === 1, "clamp high bound");

    // tempoToLS should return a string with 'em' and be within [0.02em, 0.35em]
    const lsSlow = tempoToLS(2);
    const lsFast = tempoToLS(12);
    console.assert(/em$/.test(lsSlow) && /em$/.test(lsFast), "tempoToLS unit em");

    // vaToHsl returns hsl string
    const c = vaToHsl(0.5, 0.2);
    console.assert(/^hsl\(/.test(c), "vaToHsl format");
  } catch { /* noop for production */ }
})();
