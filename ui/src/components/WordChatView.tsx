import { useEffect, useMemo, useRef, useState } from "react";
import { Play, Pause, RotateCw, AlertTriangle } from "lucide-react";

import type { Segment, WordFeature } from "../types";
import { formatSeconds } from "../utils/jsonl";

type WordChatViewProps = {
  segments: Segment[];
  duration: number;
  audioUrl?: string | null;
};

type ChatSegment = {
  id: string;
  speaker: string;
  start: number;
  end: number;
  words: WordFeature[];
};

type SpeakerPitchStats = Record<string, number>;

type NullableAudio = HTMLAudioElement | null;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function valenceArousalToColor(valence = 0, arousal = 0): string {
  const angle = Math.atan2(arousal, valence);
  const hue = ((angle * 180) / Math.PI + 360) % 360;
  const magnitude = clamp(Math.hypot(valence, arousal), 0, 1);
  const saturation = 35 + 50 * magnitude;
  const lightness = 52;
  return `hsl(${hue} ${saturation}% ${lightness}%)`;
}

function loudnessToFontSize(loudness?: number): string {
  const value = clamp(loudness ?? 0.5, 0, 1);
  return `${14 + 12 * value}px`;
}

function tempoToLetterSpacing(tempo?: number): string {
  if (!tempo || !Number.isFinite(tempo)) {
    return "0.2em";
  }
  const clamped = clamp(tempo, 2, 12);
  const spacing = 0.35 - (clamped - 2) * (0.33 / 10);
  return `${clamp(spacing, 0.02, 0.35)}em`;
}

function pitchToOffset(
  pitchMean: number | undefined,
  speaker: string,
  speakerStats: SpeakerPitchStats,
): string {
  if (!pitchMean || !Number.isFinite(pitchMean)) {
    return "0px";
  }
  const baseline = speakerStats[speaker] ?? pitchMean;
  const relative = clamp(pitchMean - baseline, -150, 150);
  return `${(relative / 150) * 12}px`;
}

function buildChatSegments(segments: Segment[]): ChatSegment[] {
  return segments
    .filter((seg) => Array.isArray(seg.words) && seg.words.length > 0)
    .map((seg) => ({
      id: seg.id,
      speaker: seg.speaker,
      start: seg.start,
      end: seg.end,
      words: seg.words as WordFeature[],
    }));
}

function computeSpeakerPitchStats(chatSegments: ChatSegment[]): SpeakerPitchStats {
  const stats: Record<string, number[]> = {};
  chatSegments.forEach((segment) => {
    segment.words.forEach((word) => {
      if (typeof word.pitch_mean === "number" && Number.isFinite(word.pitch_mean)) {
        stats[segment.speaker] = stats[segment.speaker] ?? [];
        stats[segment.speaker].push(word.pitch_mean);
      }
    });
  });

  const medians: SpeakerPitchStats = {};
  Object.entries(stats).forEach(([speaker, values]) => {
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
    medians[speaker] = median;
  });

  return medians;
}

const WordChatView = ({ segments, duration, audioUrl }: WordChatViewProps) => {
  const chatSegments = useMemo(() => buildChatSegments(segments), [segments]);
  const speakerStats = useMemo(() => computeSpeakerPitchStats(chatSegments), [chatSegments]);

  const fallbackMax = Math.max(duration, chatSegments.at(-1)?.end ?? 0);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [volume, setVolume] = useState(1);
  const [audioDuration, setAudioDuration] = useState<number | null>(null);

  const audioRef = useRef<NullableAudio>(null);
  const hasAudio = Boolean(audioUrl);

  useEffect(() => {
    setCurrentTime(0);
    setPlaying(hasAudio ? false : true);
    setAudioDuration(null);
    if (hasAudio && audioRef.current) {
      audioRef.current.currentTime = 0;
    }
  }, [audioUrl, hasAudio, segments]);

  useEffect(() => {
    if (!hasAudio || !audioUrl) {
      return;
    }
    const audio = audioRef.current ?? new Audio();
    audioRef.current = audio;
    audio.src = audioUrl;
    audio.load();
    audio.playbackRate = speed;
    audio.volume = volume;

    const handleLoaded = () => {
      if (Number.isFinite(audio.duration)) {
        setAudioDuration(audio.duration);
      }
    };
    const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
    const handlePlay = () => setPlaying(true);
    const handlePause = () => setPlaying(false);
    const handleEnded = () => {
      setPlaying(false);
      audio.currentTime = 0;
      setCurrentTime(0);
    };

    audio.addEventListener("loadedmetadata", handleLoaded);
    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("ended", handleEnded);

    return () => {
      audio.pause();
      audio.removeEventListener("loadedmetadata", handleLoaded);
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("ended", handleEnded);
    };
  }, [audioUrl, hasAudio, speed, volume]);

  useEffect(() => {
    if (!audioRef.current) return;
    audioRef.current.playbackRate = speed;
  }, [speed]);

  useEffect(() => {
    if (!audioRef.current) return;
    audioRef.current.volume = volume;
  }, [volume]);


  useEffect(() => {
    if (!hasAudio && audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  }, [hasAudio]);
  useEffect(() => {
    if (hasAudio) return;
    let animationId = 0;
    let lastTime = performance.now();

    const tick = () => {
      const now = performance.now();
      const delta = ((now - lastTime) / 1000) * speed;
      lastTime = now;
      setCurrentTime((prev) => {
        if (!playing) return prev;
        const next = prev + delta;
        if (next >= fallbackMax) {
          return 0;
        }
        return next;
      });
      animationId = requestAnimationFrame(tick);
    };

    animationId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animationId);
  }, [hasAudio, playing, speed, fallbackMax]);

  const containerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const activeIndex = chatSegments.findIndex(
      (segment) => currentTime >= segment.start && currentTime < segment.end,
    );
    if (activeIndex >= 0) {
      const element = containerRef.current?.querySelector(`#chat-seg-${activeIndex}`);
      if (element instanceof HTMLElement) {
        element.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  }, [currentTime, chatSegments]);

  const playbackMax = hasAudio ? audioDuration ?? fallbackMax : fallbackMax;

  const togglePlayback = () => {
    if (hasAudio) {
      const audio = audioRef.current;
      if (!audio) return;
      if (audio.paused) {
        audio.play().catch(() => setPlaying(false));
      } else {
        audio.pause();
      }
    } else {
      setPlaying((prev) => !prev);
    }
  };

  const resetPlayback = () => {
    if (hasAudio && audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setCurrentTime(0);
      setPlaying(false);
    } else {
      setCurrentTime(0);
    }
  };

  const handleSeek = (value: number) => {
    if (hasAudio && audioRef.current) {
      audioRef.current.currentTime = value;
    } else {
      setCurrentTime(value);
    }
  };

  if (!chatSegments.length) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white/80 p-6 text-sm text-slate-500">
        <div className="flex items-center gap-2 text-slate-600">
          <AlertTriangle size={16} />
          <span>このデータには語単位の情報が含まれていません。最新版のパイプラインで JSON を再生成してください。</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-200 bg-white/80 px-4 py-3 text-sm">
        <button
          onClick={togglePlayback}
          className="flex items-center gap-1 rounded-full bg-slate-900 px-4 py-1.5 text-xs font-semibold text-white transition hover:bg-slate-700"
        >
          {playing ? (
            <>
              <Pause size={14} /> 一時停止
            </>
          ) : (
            <>
              <Play size={14} /> 再生
            </>
          )}
        </button>
        <button
          onClick={resetPlayback}
          className="flex items-center gap-1 rounded-full border border-slate-200 px-3 py-1.5 text-xs text-slate-600 transition hover:border-slate-300"
        >
          <RotateCw size={14} /> 冒頭に戻る
        </button>
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-500">Speed</label>
          <select
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
            className="rounded border border-slate-200 px-2 py-1 text-xs"
          >
            <option value={0.75}>0.75×</option>
            <option value={1}>1.0×</option>
            <option value={1.25}>1.25×</option>
            <option value={1.5}>1.5×</option>
          </select>
        </div>
        {hasAudio ? (
          <div className="flex items-center gap-2">
            <label className="text-xs text-slate-500">Volume</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={volume}
              onChange={(e) => setVolume(parseFloat(e.target.value))}
              className="h-1 w-24"
            />
          </div>
        ) : (
          <span className="text-xs text-slate-400">音声ファイル未設定: 内部タイマーで再生</span>
        )}
        <div className="flex flex-1 items-center gap-2">
          <input
            type="range"
            min={0}
            max={playbackMax}
            step={0.05}
            value={clamp(currentTime, 0, playbackMax)}
            onChange={(e) => handleSeek(parseFloat(e.target.value))}
            className="flex-1"
          />
          <span className="w-16 text-right text-xs tabular-nums text-slate-500">{formatSeconds(currentTime)}</span>
        </div>
      </div>

      <div
        ref={containerRef}
        className="h-[60vh] overflow-auto rounded-2xl border border-slate-200 bg-white p-4"
      >
        <div className="flex flex-col gap-4">
          {chatSegments.map((segment, idx) => (
            <div key={segment.id} id={`chat-seg-${idx}`} className="flex flex-col">
              <div
                className={`mb-1 text-[10px] uppercase tracking-wide ${
                  segment.speaker === "A" ? "text-sky-600" : "text-emerald-600"
                }`}
              >
                Speaker {segment.speaker} ・ {formatSeconds(segment.start)}
              </div>
              <div
                className={`max-w-[75%] rounded-2xl px-4 py-3 shadow-sm ${
                  segment.speaker === "A"
                    ? "self-start border border-sky-100 bg-sky-50"
                    : "self-end border border-emerald-100 bg-emerald-50"
                }`}
              >
                {segment.words.map((word, wordIdx) => {
                  const isActive = currentTime >= word.start && currentTime < word.end;
                  const color = valenceArousalToColor(word.valence ?? 0, word.arousal ?? 0);
                  const fontSize = loudnessToFontSize(word.loudness);
                  const letterSpacing = tempoToLetterSpacing(word.tempo);
                  const translateY = pitchToOffset(word.pitch_mean, segment.speaker, speakerStats);
                  const loudRatio = clamp(word.loudness ?? 0, 0, 1);

                  return (
                    <span
                      key={`${segment.id}-${wordIdx}`}
                      className={`mr-2 inline-flex flex-col transition-[transform,color,font-size] duration-100 ${
                        isActive ? "font-semibold underline underline-offset-4" : ""
                      }`}
                      style={{
                        color,
                        fontSize,
                        letterSpacing,
                        transform: `translateY(${translateY})`,
                      }}
                    >
                      <span>{word.text}</span>
                      {word.kana && word.kana !== word.text ? (
                        <span className="text-[10px] text-slate-500">{word.kana}</span>
                      ) : null}
                      {word.accent ? (
                        <span className="text-[9px] text-slate-400">{word.accent}</span>
                      ) : null}
                      <span className="mt-1 inline-block h-1 w-16 rounded-full bg-slate-200">
                        <span
                          className="block h-full rounded-full bg-indigo-300"
                          style={{ width: `${loudRatio * 100}%` }}
                        />
                      </span>
                    </span>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
      <p className="text-xs text-slate-500">
        色: Valence/Arousal／サイズ: 音量／字間: テンポ／上下位置: ピッチ平均。語毎のメーターバーで瞬間的な音量の強弱も確認できます。
      </p>
    </div>
  );
};

export default WordChatView;
