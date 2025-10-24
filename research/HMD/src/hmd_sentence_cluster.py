#!/usr/bin/env python3
"""HMD Dialogue Segment Clustering & Visualization Pipeline.

Input resources (per dialogue):
  - segments.jsonl  : WhisperX (or equivalent) segments with `start`, `end`, `text`.
  - stage2_discourse.json : LLM discourse summary (used for expected speaker count).
  - <dialogue>.wav  : Original audio (mono or stereo).

Outputs under data/<dialogue>/:
  features/   -> Per segment feature vectors (JSONL).
  embeddings/ -> 2D projections (PCA/UMAP).
  clusters/   -> Cluster assignments for each projection strategy.
  viz/        -> Self-contained Plotly/WaveSurfer HTML dashboard.

Feature families implemented:
  - speaker_embedding (speechbrain ECAPA-TDNN, 192 dims)
  - prosody (F0, energy, speech-rate heuristics)
  - acoustic_stats (MFCC mean/std, delta features, ZCR, flatness)
  - spectral_series_pca64 (mel mean/std → PCA64)
  - environment (noise heuristics & spectral slope)

Clustering strategies (per feature family):
  - DBSCAN on high-dimensional (PCA≤50) representation
  - DBSCAN on PCA 2D projection
  - DBSCAN on UMAP 2D projection

The expected cluster count is derived from stage2_discourse.json. If DBSCAN
fails to reach the requested cluster count, the closest configuration is
retained; when DBSCAN yields no clusters, we fall back to KMeans.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import librosa
import pyworld
import umap
from speechbrain.inference.speaker import EncoderClassifier


# 置き換え例（幅をフレーム数以下の奇数にクランプ）
def _safe_delta(x, width=9, mode="interp"):
    T = x.shape[1]
    if T < 3:
        return np.zeros_like(x)
    # 幅を T 以下の奇数へ
    w = min(width, T if T % 2 == 1 else T - 1)
    if w < 3:
        w = 3
    # mode='interp' は「幅 > T」を許さないので、必要なら 'nearest' に変更
    use_mode = mode
    if use_mode == "interp" and w > T:
        use_mode = "nearest"
    return librosa.feature.delta(x, width=w, mode=use_mode)

@dataclass
class Segment:
    seg_id: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def load_segments(path: Path) -> List[Segment]:
    segments: List[Segment] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            segments.append(
                Segment(
                    seg_id=idx,
                    start=float(obj.get("start", 0.0)),
                    end=float(obj.get("end", 0.0)),
                    text=str(obj.get("text", "")),
                )
            )
    return segments


def load_stage2(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def derive_expected_clusters(stage2: Dict[str, Any]) -> int:
    predicted = str(stage2.get("predicted_type", "")).lower()
    speakers = stage2.get("speakers") or []
    if predicted == "monologue":
        return 1
    if predicted == "dialogue":
        if len(speakers) >= 2:
            return 2
        return 2
    if predicted == "multi_party":
        return max(3, len(speakers)) or 3
    return max(2, len(speakers)) or 2


def ensure_mono(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        return wav
    if wav.ndim == 2:
        return np.mean(wav, axis=1)
    raise ValueError("Unsupported audio shape")


def load_audio(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path)
    wav = ensure_mono(wav)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    wav = wav.astype(np.float32)
    return wav, sr


def slice_audio(wav: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    s = max(0, int(math.floor(start * sr)))
    e = min(len(wav), int(math.ceil(end * sr)))
    if e <= s:
        return np.zeros(int(0.01 * sr), dtype=np.float32)
    return wav[s:e]


def safe_log10(x: float) -> float:
    return math.log10(max(x, 1e-12))


def load_ground_truth(path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                speaker, _ = line.split(":", 1)
                speaker = speaker.strip()
            else:
                speaker = line.strip()
            mapping[idx] = speaker
            idx += 1
    return mapping


class FeatureBank:
    """Lazy feature extractors for each modality."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._speaker_model: EncoderClassifier | None = None

    @property
    def speaker_model(self) -> EncoderClassifier:
        if self._speaker_model is None:
            self._speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
            )
        return self._speaker_model

    def speaker_embedding(self, wav: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(wav).float().unsqueeze(0)
        with torch.no_grad():
            emb = self.speaker_model.encode_batch(tensor)
        return emb.squeeze(0).cpu().numpy()

    def prosody(self, wav: np.ndarray, text: str, duration: float) -> Dict[str, float]:
        sr = self.sample_rate
        wav64 = wav.astype(np.float64)
        if len(wav64) < int(0.02 * sr):
            wav64 = np.pad(wav64, (0, int(0.02 * sr) - len(wav64)))

        _f0, t = pyworld.dio(wav64, sr)
        f0 = pyworld.stonemask(wav64, _f0, t, sr)
        voiced = f0[f0 > 0]
        f0_mean = float(np.mean(voiced)) if voiced.size else 0.0
        f0_std = float(np.std(voiced)) if voiced.size else 0.0
        f0_range = float(np.max(voiced) - np.min(voiced)) if voiced.size else 0.0

        rms = librosa.feature.rms(y=wav, frame_length=512, hop_length=256)
        rms_vals = rms.flatten()
        energy_mean = float(np.mean(rms_vals)) if rms_vals.size else 0.0
        energy_std = float(np.std(rms_vals)) if rms_vals.size else 0.0
        energy_range = float(np.max(rms_vals) - np.min(rms_vals)) if rms_vals.size else 0.0

        num_chars = max(1, len(text.strip()))
        speech_rate = float(num_chars / max(duration, 1e-3))

        return {
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "f0_range": f0_range,
            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "energy_range": energy_range,
            "speech_rate": speech_rate,
        }


    def acoustic_stats(self, wav: np.ndarray) -> Dict[str, float]:
        sr = self.sample_rate
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13)
        #delta = librosa.feature.delta(mfcc)
        delta = _safe_delta(mfcc, width=9, mode="interp")
        #deltadelta = librosa.feature.delta(mfcc, order=2)
        deltadelta = _safe_delta(delta, width=9, mode="interp")
        feats = np.concatenate([mfcc, delta, deltadelta], axis=0)
        stats: Dict[str, float] = {}
        for i, row in enumerate(feats, start=1):
            stats[f"mfcc_stat_{i}_mean"] = float(np.mean(row))
            stats[f"mfcc_stat_{i}_std"] = float(np.std(row))

        zcr = librosa.feature.zero_crossing_rate(y=wav)
        flatness = librosa.feature.spectral_flatness(y=wav)
        stats["zcr_mean"] = float(np.mean(zcr)) if zcr.size else 0.0
        stats["zcr_std"] = float(np.std(zcr)) if zcr.size else 0.0
        stats["flatness_mean"] = float(np.mean(flatness)) if flatness.size else 0.0
        stats["flatness_std"] = float(np.std(flatness)) if flatness.size else 0.0
        return stats

    def spectral_series(self, wav: np.ndarray) -> np.ndarray:
        sr = self.sample_rate
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=sr,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            fmin=30,
            fmax=sr // 2,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel + 1e-9)
        means = np.mean(mel_db, axis=1)
        stds = np.std(mel_db, axis=1)
        return np.concatenate([means, stds])  # 160 dims

    def environment(self, wav: np.ndarray) -> Dict[str, float]:
        sr = self.sample_rate
        frame_rms = librosa.feature.rms(y=wav, frame_length=512, hop_length=256).flatten()
        if frame_rms.size == 0:
            frame_rms = np.array([1e-6], dtype=np.float32)
        segment_rms = float(np.mean(frame_rms))
        noise_floor = float(np.percentile(frame_rms, 10))
        snr = 20.0 * (safe_log10(segment_rms + 1e-6) - safe_log10(noise_floor + 1e-6))

        # Spectral slope: linear fit of log magnitude vs log frequency
        spectrum = np.abs(np.fft.rfft(wav * np.hanning(len(wav))))
        freqs = np.fft.rfftfreq(len(wav), d=1.0 / sr)
        valid = spectrum > 0
        if np.count_nonzero(valid) < 2:
            slope = 0.0
        else:
            x = np.log(freqs[valid] + 1e-6)
            y = np.log(spectrum[valid])
            slope = float(np.polyfit(x, y, 1)[0])

        # Decay proxy: front vs back RMS difference
        half = max(1, len(frame_rms) // 2)
        first = float(np.mean(frame_rms[:half]))
        last = float(np.mean(frame_rms[-half:]))
        decay = 20.0 * (safe_log10(first + 1e-6) - safe_log10(last + 1e-6))

        flatness = librosa.feature.spectral_flatness(y=wav).flatten()
        flatness_mean = float(np.mean(flatness)) if flatness.size else 0.0

        return {
            "snr": snr,
            "noise_floor_db": 20.0 * safe_log10(noise_floor + 1e-6),
            "spectral_slope": slope,
            "spectral_decay": decay,
            "reverb_proxy": flatness_mean,
        }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_pca(data: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    max_components = max(1, min(n_components, data.shape[0], data.shape[1]))
    if data.shape[1] <= max_components:
        return data, PCA(n_components=data.shape[1])
    pca = PCA(n_components=max_components, random_state=42)
    return pca.fit_transform(data), pca


def dbscan_grid(
    data: np.ndarray,
    expected_clusters: int,
    eps_candidates: Iterable[float],
    min_samples_candidates: Iterable[int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    best_labels = None
    best_score = (float("inf"), float("inf"))
    best_meta: Dict[str, Any] = {}

    for eps in eps_candidates:
        for min_samples in min_samples_candidates:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
            unique = sorted({c for c in labels if c != -1})
            n_clusters = len(unique)
            noise_ratio = float(np.mean(labels == -1))
            cluster_diff = abs(n_clusters - expected_clusters)
            score = (cluster_diff, noise_ratio)
            if best_labels is None or score < best_score:
                best_score = score
                best_labels = labels
                best_meta = {
                    "eps": float(eps),
                    "min_samples": int(min_samples),
                    "n_clusters": int(n_clusters),
                    "noise_ratio": float(noise_ratio),
                }

    if best_labels is None:
        raise RuntimeError("DBSCAN could not produce labels")

    n_clusters = best_meta.get("n_clusters", 0)
    if n_clusters == 0:
        km = KMeans(n_clusters=expected_clusters, random_state=42, n_init="auto")
        km_labels = km.fit_predict(data)
        best_labels = km_labels
        best_meta = {
            "fallback": "kmeans",
            "n_clusters": int(expected_clusters),
        }

    return best_labels, best_meta


def prepare_cluster_payload(
    seg_ids: List[int],
    labels: np.ndarray,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seg_id, label in zip(seg_ids, labels.tolist()):
        out.append({"seg_id": seg_id, "cluster": int(label)})
    return out


def build_visualization_html(
    target_path: Path,
    data: Dict[str, Any],
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(data, ensure_ascii=False)
    template = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <title>HMD 文単位クラスタリング可視化</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; margin: 0; padding: 0; background: #f4f4f4; }
    header { background: #1f2933; color: #fefefe; padding: 12px 20px; }
    main { display: flex; height: calc(100vh - 60px); }
    .sidebar { width: 280px; padding: 16px; background: #ffffff; box-shadow: 2px 0 6px rgba(0,0,0,0.08); overflow-y: auto; }
    .content { flex: 1; display: flex; flex-direction: column; }
    #plot { flex: 1; min-height: 360px; }
    .detail { width: 360px; padding: 16px; background: #ffffff; box-shadow: -2px 0 6px rgba(0,0,0,0.08); overflow-y: auto; }
    label { display: block; margin-top: 12px; font-weight: 600; }
    select { width: 100%; padding: 6px; margin-top: 4px; }
    .meta { margin-top: 12px; font-size: 0.9rem; color: #555; }
    .seg-text { white-space: pre-wrap; margin-top: 8px; }
    button { margin-right: 6px; padding: 6px 10px; }
    #waveform { margin-top: 12px; height: 120px; background: #f8fafc; border: 1px solid #dde3ed; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2"></script>
  <script src="https://cdn.jsdelivr.net/npm/wavesurfer.js@7.7.10"></script>
</head>
<body>
  <header>
    <h1>文単位クラスタリング可視化</h1>
    <p>特徴ごとのクラスタリング結果を比較し、文テキストと音声を確認できます。</p>
  </header>
  <main>
    <div class="sidebar">
      <label for="featureSelect">特徴タイプ</label>
      <select id="featureSelect"></select>
      <label for="modeSelect">クラスタリングモード</label>
      <select id="modeSelect"></select>
      <div class="meta" id="clusterMeta"></div>
      <div class="meta" id="shapeLegend"></div>
    </div>
    <div class="content">
      <div id="plot"></div>
      <div style="padding: 12px; background: #ffffff; border-top: 1px solid #ddd;">
        <strong>Cluster Summary:</strong>
        <span id="summaryArea">なし</span>
      </div>
    </div>
    <div class="detail">
      <h2 id="detailTitle">詳細</h2>
      <div class="meta" id="detailTiming"></div>
      <div class="seg-text" id="detailText"></div>
      <div id="waveform"></div>
      <div style="margin-top:8px;">
        <button id="playButton">再生</button>
        <button id="stopButton">停止</button>
      </div>
    </div>
  </main>
  <script>
    const DATA = __DATA_JSON__;

    const featureSelect = document.getElementById('featureSelect');
    const modeSelect = document.getElementById('modeSelect');
    const clusterMeta = document.getElementById('clusterMeta');
    const shapeLegend = document.getElementById('shapeLegend');
    const summaryArea = document.getElementById('summaryArea');
    const detailTitle = document.getElementById('detailTitle');
    const detailTiming = document.getElementById('detailTiming');
    const detailText = document.getElementById('detailText');
    const playButton = document.getElementById('playButton');
    const stopButton = document.getElementById('stopButton');
    const plotElement = document.getElementById('plot');

    let wavePlayer = null;
    let waveReady = false;
    let currentSegment = null;

    function initSelectors() {
      Object.entries(DATA.features).forEach(([key, cfg]) => {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = cfg.label;
        featureSelect.appendChild(opt);
      });

      Object.entries(DATA.clusterModes).forEach(([key, meta]) => {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = meta.label;
        modeSelect.appendChild(opt);
      });

      featureSelect.addEventListener('change', render);
      modeSelect.addEventListener('change', render);

      playButton.addEventListener('click', () => {
        if (!currentSegment) return;
        ensureWavePlayer(() => {
          wavePlayer.setTime(currentSegment.start);
          wavePlayer.play();
        });
      });

      stopButton.addEventListener('click', () => {
        if (wavePlayer) {
          wavePlayer.stop();
        }
      });

      featureSelect.value = Object.keys(DATA.features)[0];
      modeSelect.value = Object.keys(DATA.clusterModes)[0];
    }

    function ensureWavePlayer(callback) {
      if (!wavePlayer) {
        wavePlayer = WaveSurfer.create({
          container: '#waveform',
          url: DATA.audio,
          waveColor: '#8da2fb',
          progressColor: '#3d55ff',
          cursorColor: '#111',
          height: 120,
        });
        wavePlayer.on('ready', () => {
          waveReady = true;
          if (callback) callback();
        });
        wavePlayer.on('timeupdate', () => {
          if (!currentSegment) return;
          if (wavePlayer.getCurrentTime() >= currentSegment.end) {
            wavePlayer.pause();
          }
        });
      } else if (waveReady) {
        if (callback) callback();
      } else if (callback) {
        wavePlayer.once('ready', callback);
      }
    }

    function render() {
      const featureKey = featureSelect.value;
      const modeKey = modeSelect.value;
      const feature = DATA.features[featureKey];
      const clusterKey = DATA.clusterModes[modeKey].clusterKey;
      const coordsKey = DATA.clusterModes[modeKey].coordsKey;

      const points = [];
      const labelCounts = new Map();

      feature.segments.forEach(seg => {
        const coord = feature.coords[coordsKey][seg.seg_id];
        if (!coord) return;
        const clusterLabel = feature.clusters[clusterKey][seg.seg_id] ?? -1;
        const colorLabel = clusterLabel;
        const symbol = feature.shapes ? feature.shapes[seg.seg_id] : null;
        if (!labelCounts.has(clusterLabel)) labelCounts.set(clusterLabel, 0);
        labelCounts.set(clusterLabel, labelCounts.get(clusterLabel) + 1);
        points.push({
          x: coord[0],
          y: coord[1],
          cluster: clusterLabel,
          color: colorLabel,
          symbol,
          seg_id: seg.seg_id,
          start: seg.start,
          end: seg.end,
          text: seg.text,
          speaker_gt: seg.speaker_gt || null,
        });
      });

      const palette = DATA.palette;
      const trace = {
        x: points.map(p => p.x),
        y: points.map(p => p.y),
        text: points.map(p => `[${p.seg_id}] ${p.text.slice(0, 30)}${p.text.length > 30 ? '…' : ''}`),
        customdata: points.map(p => p),
        mode: 'markers',
        type: 'scatter',
        marker: {
          size: 9,
          color: points.map(p => {
            const value = p.color;
            if (typeof value !== 'number' || value < 0) return '#b0b0b0';
            return palette[value % palette.length] || '#b0b0b0';
          }),
          symbol: points.map(p => p.symbol || 'circle'),
          line: { width: 0.6, color: '#1f2933' },
        },
        hovertemplate: '<b>seg %{customdata.seg_id}</b><br>' +
          '%{customdata.start:.2f}s – %{customdata.end:.2f}s<br>' +
          '%{customdata.text}<extra></extra>',
      };

      Plotly.newPlot(plotElement, [trace], {
        margin: { t: 20, r: 10, b: 40, l: 40 },
        xaxis: { title: DATA.clusterModes[modeKey].xLabel },
        yaxis: { title: DATA.clusterModes[modeKey].yLabel },
        paper_bgcolor: '#fdfdfd',
        plot_bgcolor: '#fdfdfd',
      }, { responsive: true });

      const summary = Array.from(labelCounts.entries())
        .sort((a, b) => a[0] - b[0])
        .map(([label, count]) => `${label}: ${count}`)
        .join(', ');
      summaryArea.textContent = summary || 'クラスタなし';

      const metaInfo = feature.meta[clusterKey] || {};
      const entries = Object.entries(metaInfo)
        .map(([k, v]) => {
          if (typeof v === 'number' && Number.isFinite(v)) {
            return `${k}: ${v.toFixed(3)}`;
          }
          return `${k}: ${v}`;
        })
        .join('<br>');
      clusterMeta.innerHTML = entries;

      if (feature.shapeLegend) {
        const rows = Object.entries(feature.shapeLegend)
          .map(([label, symbol]) => `${label}: ${symbol}`)
          .join('<br>');
        shapeLegend.innerHTML = `<strong>形状(正解話者):</strong><br>${rows}`;
      } else {
        shapeLegend.innerHTML = '';
      }

      if (plotElement.removeAllListeners) {
        plotElement.removeAllListeners('plotly_click');
      }
      plotElement.on('plotly_click', (eventData) => {
        if (!eventData || !eventData.points || !eventData.points.length) {
          return;
        }
        const info = eventData.points[0].customdata;
        showDetail(info);
      });
    }

    function showDetail(info) {
      currentSegment = info;
      const speakerNote = info.speaker_gt ? ` (話者: ${info.speaker_gt})` : '';
      detailTitle.textContent = `seg ${info.seg_id}${speakerNote}`;
      detailTiming.textContent = `${info.start.toFixed(2)}s – ${info.end.toFixed(2)}s`;
      detailText.textContent = info.text;
      ensureWavePlayer(() => {
        wavePlayer.setTime(info.start);
      });
    }

    initSelectors();
    render();
  </script>
</body>
</html>
"""
    html = template.replace("__DATA_JSON__", data_json)
    target_path.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="文単位クラスタリング & 可視化生成")
    ap.add_argument("--segments", required=True, help="segments.jsonl path")
    ap.add_argument("--stage2", required=True, help="stage2_discourse.json path")
    ap.add_argument("--audio", required=True, help="dialogue wav path")
    ap.add_argument(
        "--output-root",
        required=True,
        help="Output directory (e.g. data/LD01-Dialogue-01)",
    )
    ap.add_argument("--umap-neighbors", default=15, type=int)
    ap.add_argument("--umap-min-dist", default=0.1, type=float)
    ap.add_argument("--ground-truth", default=None, help="Ground truth dialogue label file (e.g. data/LD01-Dialogue-01.txt)")
    args = ap.parse_args()

    segments_path = Path(args.segments)
    stage2_path = Path(args.stage2)
    audio_path = Path(args.audio)
    output_root = Path(args.output_root)

    if not segments_path.exists():
        raise FileNotFoundError(segments_path)
    if not stage2_path.exists():
        raise FileNotFoundError(stage2_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    segments = load_segments(segments_path)
    stage2 = load_stage2(stage2_path)
    expected_clusters = derive_expected_clusters(stage2)

    ground_truth_path: Optional[Path] = None
    if args.ground_truth:
        potential = Path(args.ground_truth)
        if potential.exists():
            ground_truth_path = potential
        else:
            print(f"[WARN] Ground truth path not found: {potential}")
    else:
        candidate = segments_path.parent.parent / f"{segments_path.parent.name}.txt"
        if candidate.exists():
            ground_truth_path = candidate

    ground_truth_map: Dict[int, str] | None = None
    if ground_truth_path and ground_truth_path.exists():
        ground_truth_map = load_ground_truth(ground_truth_path)
        if len(ground_truth_map) != len(segments):
            print(
                f"[WARN] Ground truth line count ({len(ground_truth_map)}) "
                f"differs from segments ({len(segments)})"
            )

    wav, sr = load_audio(audio_path)
    bank = FeatureBank(sample_rate=sr)

    speaker_rows: List[Dict[str, Any]] = []
    prosody_rows: List[Dict[str, Any]] = []
    acoustic_rows: List[Dict[str, Any]] = []
    spectral_rows: List[Dict[str, Any]] = []
    env_rows: List[Dict[str, Any]] = []

    spectral_matrix: List[np.ndarray] = []
    speaker_matrix: List[np.ndarray] = []

    symbol_cycle = [
        "circle",
        "square",
        "diamond",
        "triangle-up",
        "triangle-down",
        "cross",
        "x",
        "star",
        "hexagon",
        "hourglass",
    ]
    label_to_symbol: Dict[str, str] = {}
    shapes_by_seg: Dict[int, str] = {}

    if ground_truth_map:
        for idx, seg in enumerate(segments):
            label = ground_truth_map.get(idx)
            if label is None:
                continue
            if label not in label_to_symbol:
                symbol = symbol_cycle[len(label_to_symbol) % len(symbol_cycle)]
                label_to_symbol[label] = symbol
            shapes_by_seg[seg.seg_id] = label_to_symbol[label]

    shape_legend = label_to_symbol if label_to_symbol else {}

    for seg in segments:
        chunk = slice_audio(wav, sr, seg.start, seg.end)

        try:
            speaker_vec = bank.speaker_embedding(chunk)
        except Exception:
            speaker_vec = np.zeros(192, dtype=np.float32)
        speaker_rows.append({
            "seg_id": seg.seg_id,
            "start": seg.start,
            "end": seg.end,
            "embedding": speaker_vec.tolist(),
        })
        speaker_matrix.append(speaker_vec)

        prosody = bank.prosody(chunk, seg.text, seg.duration)
        prosody_rows.append({"seg_id": seg.seg_id, **prosody})

        acoustic = bank.acoustic_stats(chunk)
        acoustic_rows.append({"seg_id": seg.seg_id, **acoustic})

        spectral = bank.spectral_series(chunk)
        spectral_rows.append({"seg_id": seg.seg_id, "vec": spectral.tolist()})
        spectral_matrix.append(spectral)

        env = bank.environment(chunk)
        env_rows.append({"seg_id": seg.seg_id, **env})

    features_dir = output_root / "features"
    write_jsonl(features_dir / "speaker_embedding.jsonl", speaker_rows)
    write_jsonl(features_dir / "prosody.jsonl", prosody_rows)
    write_jsonl(features_dir / "acoustic_stats.jsonl", acoustic_rows)
    write_jsonl(features_dir / "spectral_series_raw.jsonl", spectral_rows)
    write_jsonl(features_dir / "environment.jsonl", env_rows)

    def df_from_rows(rows: List[Dict[str, Any]], key_prefix: str | None = None) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if key_prefix:
            cols = [c for c in df.columns if c != "seg_id"]
            df = df.set_index("seg_id")
            df = df.rename(columns=lambda c: f"{key_prefix}_{c}")
            df.insert(0, "seg_id", df.index)
            df = df.reset_index(drop=True)
        return df

    feature_configs = {
        "speaker_embedding": {
            "label": "Speaker Embedding",
            "matrix": np.vstack(speaker_matrix),
            "rows": speaker_rows,
        },
        "prosody": {
            "label": "Prosody",
            "matrix": df_from_rows(prosody_rows).drop(columns=["seg_id"]).to_numpy(),
            "rows": prosody_rows,
        },
        "acoustic_stats": {
            "label": "Acoustic Stats",
            "matrix": df_from_rows(acoustic_rows).drop(columns=["seg_id"]).to_numpy(),
            "rows": acoustic_rows,
        },
        "spectral_series_pca64": {
            "label": "Spectral Series",
            "matrix": None,  # Placeholder, computed below
            "rows": spectral_rows,
        },
        "environment": {
            "label": "Environment",
            "matrix": df_from_rows(env_rows).drop(columns=["seg_id"]).to_numpy(),
            "rows": env_rows,
        },
    }

    spectral_matrix_np = np.vstack(spectral_matrix)
    scaler_spec = StandardScaler()
    spectral_scaled = scaler_spec.fit_transform(spectral_matrix_np)
    max_components = max(1, min(64, spectral_scaled.shape[0], spectral_scaled.shape[1]))
    pca64 = PCA(n_components=max_components, random_state=42)
    spectral_pca = pca64.fit_transform(spectral_scaled)
    feature_configs["spectral_series_pca64"]["matrix"] = spectral_pca

    cluster_dir = output_root / "clusters"
    embed_dir = output_root / "embeddings"

    audio_dir = output_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_filename = Path(args.audio).name
    audio_out_path = audio_dir / audio_filename
    if not audio_out_path.exists():
        shutil.copy(audio_path, audio_out_path)

    seg_ids = [seg.seg_id for seg in segments]
    viz_segments: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        entry: Dict[str, Any] = {
            "seg_id": seg.seg_id,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        }
        if ground_truth_map:
            entry["speaker_gt"] = ground_truth_map.get(idx)
        viz_segments.append(entry)

    n_segments = len(seg_ids)

    def compute_neighbors(divisor: int) -> int:
        if n_segments <= 2:
            return max(1, n_segments - 1)
        cand = max(2, n_segments // divisor)
        cand = min(cand, n_segments - 1)
#        return min(cand, 15)
        return cand

    umap_neighbors_quarter = compute_neighbors(5) if n_segments else 2
    umap_neighbors_half = compute_neighbors(3) if n_segments else 2

    viz_payload: Dict[str, Any] = {
        "segments": viz_segments,
        "audio": str(Path("..") / "audio" / audio_filename),
        "features": {},
        "clusterModes": {
            "dbscan_highd": {
                "label": "DBSCAN (PCA≤50高次元)",
                "clusterKey": "dbscan_highd",
                "coordsKey": "pca2",
                "xLabel": "PCA-1",
                "yLabel": "PCA-2",
            },
            "dbscan_pca2": {
                "label": "DBSCAN (PCA2次元)",
                "clusterKey": "dbscan_pca2",
                "coordsKey": "pca2",
                "xLabel": "PCA-1",
                "yLabel": "PCA-2",
            },
            "dbscan_umap2_quarter": {
                "label": f"DBSCAN (UMAP2 近傍{umap_neighbors_quarter})",
                "clusterKey": "dbscan_umap2_quarter",
                "coordsKey": "umap2_quarter",
                "xLabel": "UMAP-1",
                "yLabel": "UMAP-2",
            },
            "dbscan_umap2_half": {
                "label": f"DBSCAN (UMAP2 近傍{umap_neighbors_half})",
                "clusterKey": "dbscan_umap2_half",
                "coordsKey": "umap2_half",
                "xLabel": "UMAP-1",
                "yLabel": "UMAP-2",
            },
        },
        "palette": [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
            "#2E91E5", "#E15F99", "#1CA71C", "#FB0D0D", "#DA16FF",
            "#222222", "#0A8754", "#A4243B",
        ],
        "stage2": stage2,
    }

    manifest: Dict[str, Any] = {
        "features": {},
        "audio": str(Path("..") / "audio" / audio_filename),
        "segments_file": str(Path(args.segments).name),
    }

    eps_candidates = [0.3, 0.5, 0.7, 0.9]
    min_samples_candidates = [3, 5, 8, max(3, round(0.01 * len(seg_ids)))]

    for feat_key, cfg in feature_configs.items():
        label = cfg["label"]
        matrix = cfg["matrix"]
        rows = cfg["rows"]

        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)

        highd, pca_model = run_pca(scaled, 50)
        pca2 = PCA(n_components=2, random_state=42).fit_transform(scaled)
        umap2_quarter = umap.UMAP(
            n_neighbors=max(2, umap_neighbors_quarter),
            min_dist=args.umap_min_dist,
            metric="euclidean",
            random_state=42,
        ).fit_transform(highd)
        umap2_half = umap.UMAP(
            n_neighbors=max(2, umap_neighbors_half),
            min_dist=args.umap_min_dist,
            metric="euclidean",
            random_state=42,
        ).fit_transform(highd)

        labels_highd, meta_highd = dbscan_grid(highd, expected_clusters, eps_candidates, min_samples_candidates)
        labels_pca2, meta_pca2 = dbscan_grid(pca2, expected_clusters, eps_candidates, min_samples_candidates)
        labels_umap2_quarter, meta_umap2_quarter = dbscan_grid(
            umap2_quarter, expected_clusters, eps_candidates, min_samples_candidates
        )
        labels_umap2_half, meta_umap2_half = dbscan_grid(
            umap2_half, expected_clusters, eps_candidates, min_samples_candidates
        )

        meta_umap2_quarter["umap_n_neighbors"] = umap_neighbors_quarter
        meta_umap2_half["umap_n_neighbors"] = umap_neighbors_half

        feature_entry = {
            "label": label,
            "segments": viz_segments,
            "shapes": shapes_by_seg,
            "shapeLegend": shape_legend or None,
            "coords": {
                "pca2": {seg_id: pca2[i].tolist() for i, seg_id in enumerate(seg_ids)},
                "umap2_quarter": {seg_id: umap2_quarter[i].tolist() for i, seg_id in enumerate(seg_ids)},
                "umap2_half": {seg_id: umap2_half[i].tolist() for i, seg_id in enumerate(seg_ids)},
            },
            "clusters": {
                "dbscan_highd": {seg_id: int(labels_highd[i]) for i, seg_id in enumerate(seg_ids)},
                "dbscan_pca2": {seg_id: int(labels_pca2[i]) for i, seg_id in enumerate(seg_ids)},
                "dbscan_umap2_quarter": {
                    seg_id: int(labels_umap2_quarter[i]) for i, seg_id in enumerate(seg_ids)
                },
                "dbscan_umap2_half": {
                    seg_id: int(labels_umap2_half[i]) for i, seg_id in enumerate(seg_ids)
                },
            },
            "meta": {
                "dbscan_highd": meta_highd,
                "dbscan_pca2": meta_pca2,
                "dbscan_umap2_quarter": meta_umap2_quarter,
                "dbscan_umap2_half": meta_umap2_half,
            },
        }
        viz_payload["features"][feat_key] = feature_entry

        feat_cluster_dir = cluster_dir / feat_key
        write_json(
            feat_cluster_dir / "dbscan_highd.json",
            prepare_cluster_payload(seg_ids, labels_highd),
        )
        write_json(
            feat_cluster_dir / "dbscan_pca2.json",
            prepare_cluster_payload(seg_ids, labels_pca2),
        )
        write_json(
            feat_cluster_dir / "dbscan_umap2_quarter.json",
            prepare_cluster_payload(seg_ids, labels_umap2_quarter),
        )
        write_json(
            feat_cluster_dir / "dbscan_umap2_half.json",
            prepare_cluster_payload(seg_ids, labels_umap2_half),
        )

        feat_embed_dir = embed_dir / feat_key
        write_jsonl(
            feat_embed_dir / "pca2.jsonl",
            (
                {"seg_id": seg_id, "x": float(pca2[i, 0]), "y": float(pca2[i, 1])}
                for i, seg_id in enumerate(seg_ids)
            ),
        )
        write_jsonl(
            feat_embed_dir / "umap2_quarter.jsonl",
            (
                {"seg_id": seg_id, "x": float(umap2_quarter[i, 0]), "y": float(umap2_quarter[i, 1])}
                for i, seg_id in enumerate(seg_ids)
            ),
        )
        write_jsonl(
            feat_embed_dir / "umap2_half.jsonl",
            (
                {"seg_id": seg_id, "x": float(umap2_half[i, 0]), "y": float(umap2_half[i, 1])}
                for i, seg_id in enumerate(seg_ids)
            ),
        )

        manifest["features"][feat_key] = {
            "label": label,
            "clusters": {
                "dbscan_highd": str(Path("..") / "clusters" / feat_key / "dbscan_highd.json"),
                "dbscan_pca2": str(Path("..") / "clusters" / feat_key / "dbscan_pca2.json"),
                "dbscan_umap2_quarter": str(Path("..") / "clusters" / feat_key / "dbscan_umap2_quarter.json"),
                "dbscan_umap2_half": str(Path("..") / "clusters" / feat_key / "dbscan_umap2_half.json"),
            },
            "embeddings": {
                "pca2": str(Path("..") / "embeddings" / feat_key / "pca2.jsonl"),
                "umap2_quarter": str(Path("..") / "embeddings" / feat_key / "umap2_quarter.jsonl"),
                "umap2_half": str(Path("..") / "embeddings" / feat_key / "umap2_half.jsonl"),
            },
        }

    viz_dir = output_root / "viz"
    write_json(viz_dir / "data_manifest.json", manifest)
    build_visualization_html(viz_dir / "index.html", viz_payload)

    print(f"[HMD] Clustering + viz complete -> {output_root}")


if __name__ == "__main__":
    main()
