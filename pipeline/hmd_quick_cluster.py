#!/usr/bin/env python3
"""
HMD Quick Cluster
-----------------
One-shot pipeline:
  1) (Optional) ASR with OpenAI Whisper (pure) → segments.jsonl / plain.txt / (approx) words.jsonl
  2) ECAPA-TDNN speaker embeddings per segment
  3) Preprocess → PCA(≤D) for evaluation + PCA(2D) for visualization
  4) K-Means clustering on PCA(2D)
     - if k_min == k_max → fixed-k
     - else choose k by max silhouette score (evaluation space selectable: 2d or 50d)
  5) Save artifacts compatible with hmd_sentence_cluster.py outputs
     - features/, embeddings/, clusters/, viz/

Priority of configuration: CLI > .env > built-in defaults
Requires: python 3.10+, numpy, pandas, torch, torchaudio, speechbrain, librosa, scikit-learn, plotly, (optional) kaleido

Usage examples:
  python HMD/src/hmd_quick_cluster.py HMD/data/LD01-Dialogue-01.wav \
    --k-min 2 --k-max 6 --preprocess zscore --pca-dims 50 --pca-eval 50d

"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ML stack
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize

# Audio stack
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# Viz
import plotly.express as px

# .env (optional)
try:
    from dotenv import dotenv_values
except Exception:  # pragma: no cover
    dotenv_values = None

# -------------------------
# Defaults
# -------------------------
DFLT = {
    "data_root": str((Path(os.getenv("CV_OUTPUT_DIR", "output/pipeline")) / "quick_artifacts").resolve()),
    # Whisper
    "whisper_model": "large-v3-turbo",
    "whisper_language": "ja",
    "whisper_device": "auto",
    "whisper_no_condition": True,
    "whisper_no_words": False,
    # Embedding/Preprocess
    "speaker_sr": 16000,
    "preprocess": "zscore",  # raw|zscore|l2|zscore_l2|whiten
    "pca_dims": 50,
    "pca_eval": "50d",  # "2d" or "50d"
    # Kmeans / scoring
    "k_min": 2,
    "k_max": 12,
    "random_state": 42,
    "kmeans_n_init": "auto",
    "silhouette_metric": "euclidean",
    "min_segments": 4,
    # Output
    "overwrite": False,
    "save_html": True,
    "save_png": True,
    "reuse_segments": True,
}

ENV_KEYS = {
    # Paths
    "HMD_DATA_ROOT": "data_root",
    # Whisper
    "WHISPER_MODEL": "whisper_model",
    "WHISPER_LANGUAGE": "whisper_language",
    "WHISPER_DEVICE": "whisper_device",
    "WHISPER_NO_CONDITION": "whisper_no_condition",
    "WHISPER_NO_WORDS": "whisper_no_words",
    # Embedding/Preprocess
    "HMD_SPEAKER_SR": "speaker_sr",
    "HMD_PREPROCESS": "preprocess",
    "HMD_PCA_DIMS": "pca_dims",
    "HMD_PCA_EVAL": "pca_eval",
    # Clustering
    "HMD_K_MIN": "k_min",
    "HMD_K_MAX": "k_max",
    "HMD_RANDOM_STATE": "random_state",
    "HMD_KMEANS_N_INIT": "kmeans_n_init",
    "HMD_SILHOUETTE_METRIC": "silhouette_metric",
    "HMD_MIN_SEGMENTS": "min_segments",
    # Output
    "HMD_OVERWRITE": "overwrite",
    "HMD_SAVE_HTML": "save_html",
    "HMD_SAVE_PNG": "save_png",
    "HMD_REUSE_SEGMENTS": "reuse_segments",
}

# -------------------------
# Utilities
# -------------------------

def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _load_env(env_path: Optional[str]) -> Dict[str, Any]:
    env: Dict[str, Any] = {}
    if env_path:
        if dotenv_values is None:
            raise RuntimeError("python-dotenv not installed. pip install python-dotenv or omit --env")
        if not os.path.exists(env_path):
            raise FileNotFoundError(env_path)
        env.update({k: v for k, v in dotenv_values(env_path).items() if v is not None})
    # process env overrides
    for k, v in os.environ.items():
        if k in ENV_KEYS:
            env[k] = v
    return env


def _merge_cfg(cli: Dict[str, Any], env: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(DFLT)
    # env → internal keys
    for ek, ik in ENV_KEYS.items():
        if ek in env:
            val = env[ek]
            if ik in ("speaker_sr", "pca_dims", "k_min", "k_max", "random_state", "min_segments"):
                try:
                    val = int(val)
                except Exception:
                    pass
            if ik in ("whisper_no_condition", "whisper_no_words", "overwrite", "save_html", "save_png", "reuse_segments"):
                val = _coerce_bool(val)
            cfg[ik] = val
    # cli (already parsed types)
    for k, v in cli.items():
        if v is not None:
            cfg[k] = v
    # sanity
    cfg["k_min"] = max(2, int(cfg["k_min"]))
    cfg["k_max"] = max(cfg["k_min"], int(cfg["k_max"]))
    cfg["pca_eval"] = str(cfg["pca_eval"]).lower()
    if cfg["pca_eval"] not in ("2d", "50d"):
        cfg["pca_eval"] = "50d"
    return cfg


@dataclass
class Segment:
    seg_id: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# -------------------------
# ASR (OpenAI Whisper, pure)
# -------------------------

def asr_whisper_pure(
    audio: Path,
    out_dir: Path,
    *,
    model_name: str,
    language: str,
    device: str,
    no_condition: bool,
    no_words: bool,
) -> List[Segment]:
    """Run OpenAI Whisper (pure) and save plain/segments/words(approx)."""
    try:
        import whisper  # openai-whisper
        import re
        from datetime import datetime
        WORDLIKE_RE = re.compile(r"[一-龥ぁ-んァ-ンー]+|[A-Za-z0-9]+|[^\s]")
    except Exception as e:  # pragma: no cover
        raise RuntimeError("openai-whisper is required. pip install -U openai-whisper") from e

    out_dir.mkdir(parents=True, exist_ok=True)

    # device resolve
    if device == "auto":
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(
        str(audio),
        language=language,
        task="transcribe",
        condition_on_previous_text=not no_condition,
        fp16=(device == "cuda"),
        verbose=False,
    )

    segments = result.get("segments", [])
    plain_txt = out_dir / "plain.txt"
    seg_jsonl = out_dir / "segments.jsonl"
    words_jsonl = out_dir / "words.jsonl"

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "audio": str(audio),
                "model": f"{model_name}",
                "language": language,
                "invoked_at": __import__("datetime").datetime.now().isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # write segments & plain
    texts: List[str] = []
    with seg_jsonl.open("w", encoding="utf-8") as fseg:
        for seg in segments:
            text = (seg.get("text") or "").strip()
            obj = {"start": float(seg.get("start", 0.0)), "end": float(seg.get("end", 0.0)), "text": text}
            fseg.write(json.dumps(obj, ensure_ascii=False) + "\n")
            texts.append(text)
    plain_txt.write_text("".join(texts), encoding="utf-8")

    # approx words
    if not no_words:
        with words_jsonl.open("w", encoding="utf-8") as fword:
            for seg in segments:
                text = (seg.get("text") or "").strip()
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                dur = max(0.0, end - start)
                tokens = WORDLIKE_RE.findall(text)
                n = len(tokens)
                if n == 0:
                    continue
                step = dur / n if n > 0 else 0.0
                for i, w in enumerate(tokens):
                    w_start = start + step * i
                    w_end = start + step * (i + 1)
                    fword.write(
                        json.dumps(
                            {
                                "start": float(w_start),
                                "end": float(w_end),
                                "word": w,
                                "segment_start": float(start),
                                "segment_end": float(end),
                                "approx": True,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    # parse back into Segment list
    segs: List[Segment] = []
    with open(seg_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            o = json.loads(line)
            segs.append(Segment(seg_id=i, start=float(o["start"]), end=float(o["end"]), text=str(o["text"])) )
    return segs


# -------------------------
# Audio helpers
# -------------------------

def load_audio_mono_16k(path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def slice_audio(wav: torch.Tensor, sr: int, start: float, end: float) -> torch.Tensor:
    s_i = max(0, int(round(start * sr)))
    e_i = min(wav.shape[-1], int(round(end * sr)))
    if e_i <= s_i:
        e_i = min(wav.shape[-1], s_i + int(0.05 * sr))
    return wav[:, s_i:e_i]


# -------------------------
# Feature extraction
# -------------------------

def ecapa_embeddings(audio_path: Path, segments: List[Segment], *, target_sr: int) -> np.ndarray:
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
    wav, sr = load_audio_mono_16k(audio_path, target_sr)
    embs: List[np.ndarray] = []
    with torch.inference_mode():
        for seg in segments:
            chunk = slice_audio(wav, sr, seg.start, seg.end)
            if chunk.numel() == 0:
                embs.append(np.zeros(192, dtype=np.float32))
                continue
            emb = classifier.encode_batch(chunk).squeeze().detach().cpu().numpy()
            embs.append(emb)
    return np.stack(embs)


# -------------------------
# Preprocess & PCA
# -------------------------

def apply_preprocess(X: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "zscore").lower()
    if mode == "raw":
        return X
    if mode == "zscore":
        return StandardScaler().fit_transform(X)
    if mode == "l2":
        return normalize(X, norm="l2")
    if mode == "zscore_l2":
        return normalize(StandardScaler().fit_transform(X), norm="l2")
    if mode == "whiten":
        return PCA(whiten=True, random_state=0).fit_transform(X)
    raise ValueError(f"unknown preprocess mode: {mode}")


# -------------------------
# KMeans grid (silhouette selection)
# -------------------------

def kmeans_silhouette_select(
    X_cluster: np.ndarray,
    X_eval: np.ndarray,
    k_min: int,
    k_max: int,
    *,
    n_init: Any = "auto",
    random_state: int = 42,
    metric: str = "euclidean",
) -> Tuple[int, np.ndarray, Dict[str, Any]]:
    best_k, best_score, best_labels = None, -999.0, None
    scores: List[Tuple[int, float, float]] = []  # (k, sil_eval, sil_2d)
    for k in range(int(k_min), int(k_max) + 1):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X_cluster)
        # guard: silhouette needs >= 2 clusters and < n_samples
        if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
            sil_eval = -1.0
            sil_2d = -1.0
        else:
            sil_eval = float(silhouette_score(X_eval, labels, metric=metric))
            sil_2d = float(silhouette_score(X_cluster, labels, metric=metric))
        scores.append((k, sil_eval, sil_2d))
        if sil_eval > best_score:
            best_k, best_score, best_labels = k, sil_eval, labels
    meta = {
        "scan": [
            {"k": int(k), "silhouette_eval": float(s1), "silhouette_2d": float(s2)}
            for (k, s1, s2) in scores
        ],
        "best": {"k": int(best_k), "silhouette_eval": float(best_score)},
    }
    return int(best_k), best_labels, meta


# -------------------------
# Writers
# -------------------------

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------
# Visualization
# -------------------------

def save_scatter_pca2(
    X2: np.ndarray,
    labels: np.ndarray,
    segments: List[Segment],
    out_html: Path,
    out_png: Optional[Path],
):
    df = pd.DataFrame(
        {
            "x": X2[:, 0],
            "y": X2[:, 1],
            "cluster": labels.astype(int),
            "seg_id": [s.seg_id for s in segments],
            "start": [s.start for s in segments],
            "end": [s.end for s in segments],
            "text": [s.text for s in segments],
        }
    )
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=df["cluster"].astype(str),
        hover_data=["seg_id", "start", "end", "text"],
        title="PCA-2D KMeans Clusters (Speaker Embeddings)",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)
    if out_png is not None:
        try:
            fig.write_image(str(out_png), scale=2)
        except Exception:
            # kaleido 未導入でも処理を続ける
            pass


# -------------------------
# Main
# -------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HMD Quick Cluster: Whisper → ECAPA → PCA2D KMeans → Save")
    ap.add_argument("audio", help="Input audio file (wav)")
    ap.add_argument("--env", default=None, help=".env path (optional)")

    # Paths
    ap.add_argument("--data-root", default=None, help="Output root (default: HMD/data)")

    # Whisper
    ap.add_argument("--whisper-model", default=None, help="Whisper model (default: large-v3-turbo)")
    ap.add_argument("--language", dest="whisper_language", default=None, help="Whisper language (default: ja)")
    ap.add_argument("--device", dest="whisper_device", default=None, help="auto|cpu|cuda")
    ap.add_argument("--no-condition", dest="whisper_no_condition", action="store_true")
    ap.add_argument("--no-words", dest="whisper_no_words", action="store_true")
    ap.add_argument("--reuse-segments", dest="reuse_segments", action="store_true")

    # Embedding/Preprocess/PCA
    ap.add_argument("--speaker-sr", type=int, default=None, help="ECAPA SR (default: 16000)")
    ap.add_argument("--preprocess", default=None, choices=["raw", "zscore", "l2", "zscore_l2", "whiten"])
    ap.add_argument("--pca-dims", type=int, default=None, help="PCA dims for eval space (≤N, default: 50)")
    ap.add_argument("--pca-eval", default=None, choices=["2d", "50d"], help="Silhouette eval space (default: 50d)")

    # KMeans / selection
    ap.add_argument("--k-min", type=int, default=None)
    ap.add_argument("--k-max", type=int, default=None)
    ap.add_argument("--random-state", type=int, default=None)
    ap.add_argument("--kmeans-n-init", default=None)
    ap.add_argument("--silhouette-metric", default=None)
    ap.add_argument("--min-segments", type=int, default=None)

    # Output
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-html", dest="save_html", action="store_false")
    ap.add_argument("--no-png", dest="save_png", action="store_false")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Merge config: CLI > .env > defaults
    env = _load_env(getattr(args, "env", None)) if getattr(args, "env", None) else {}
    cli = {k: getattr(args, k) for k in vars(args)}
    cfg = _merge_cfg(cli, env)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        maybe = Path(cfg["data_root"]) / args.audio
        if maybe.exists():
            audio_path = maybe
        else:
            raise FileNotFoundError(f"Audio not found: {args.audio}")

    basename = audio_path.stem
    out_dir = Path(cfg["data_root"]) / basename
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Reuse segments or run ASR
    seg_path = out_dir / "segments.jsonl"
    if cfg["reuse_segments"] and seg_path.exists():
        segments: List[Segment] = []
        with open(seg_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                o = json.loads(line)
                segments.append(Segment(i, float(o["start"]), float(o["end"]), str(o["text"])) )
        print(f"[asr] reuse segments: {seg_path} ({len(segments)} segs)")
    else:
        print(f"[asr] running Whisper ({cfg['whisper_model']}) ...")
        segments = asr_whisper_pure(
            audio_path,
            out_dir,
            model_name=cfg["whisper_model"],
            language=cfg["whisper_language"],
            device=cfg["whisper_device"],
            no_condition=bool(cfg["whisper_no_condition"]),
            no_words=bool(cfg["whisper_no_words"]),
        )
        print(f"[asr] segments: {len(segments)}")

    if len(segments) < 2:
        raise SystemExit("Not enough segments for clustering")

    # 2) ECAPA embeddings
    print("[embed] ECAPA embeddings ...")
    X = ecapa_embeddings(audio_path, segments, target_sr=int(cfg["speaker_sr"]))

    # 3) Preprocess & PCA spaces
    Xp = apply_preprocess(X, cfg["preprocess"])  # for PCA

    # eval space (≤pca_dims)
    p_dim = max(2, min(int(cfg["pca_dims"]), Xp.shape[0] - 1, Xp.shape[1]))
    pca_eval = PCA(n_components=p_dim, random_state=0)
    X_eval = pca_eval.fit_transform(Xp)

    # viz/cluster space (2D)
    pca2 = PCA(n_components=2, random_state=0)
    X2 = pca2.fit_transform(Xp)

    # 4) KMeans selection
    k_min, k_max = int(cfg["k_min"]), int(cfg["k_max"])
    max_allowed = max(2, len(segments) - 1)
    if max_allowed < 2:
        raise SystemExit("Not enough segments for clustering")
    if len(segments) <= int(cfg["min_segments"]):
        k_min = k_max = min(2, max_allowed)
    else:
        k_max = min(k_max, max_allowed)
        k_min = min(k_min, k_max)
        if k_min < 2:
            k_min = 2

    if k_min == k_max:
        k_star = k_min
        km = KMeans(n_clusters=k_star, n_init=cfg["kmeans_n_init"], random_state=int(cfg["random_state"]))
        labels = km.fit_predict(X2)
        meta_scan = [{"k": int(k_star), "silhouette_eval": float(silhouette_score(X_eval, labels)), "silhouette_2d": float(silhouette_score(X2, labels))}]
        meta_best = {"k": int(k_star), "silhouette_eval": meta_scan[0]["silhouette_eval"]}
        meta = {"scan": meta_scan, "best": meta_best}
    else:
        eval_space = X_eval if cfg["pca_eval"] == "50d" else X2
        k_star, labels, meta = kmeans_silhouette_select(
            X2,
            eval_space,
            k_min,
            k_max,
            n_init=cfg["kmeans_n_init"],
            random_state=int(cfg["random_state"]),
            metric=str(cfg["silhouette_metric"]),
        )

    print(f"[kmeans] k={k_star} (eval space={cfg['pca_eval']})")

    # 5) Save artifacts -------------------------------------------------
    # features
    features_dir = out_dir / "features"
    rows_feat = (
        {"seg_id": s.seg_id, "start": s.start, "end": s.end, "embedding": X[i].tolist()} 
        for i, s in enumerate(segments)
    )
    write_jsonl(features_dir / "speaker_embedding.jsonl", rows_feat)

    # embeddings
    emb_dir = out_dir / "embeddings" / "speaker_embedding"
    write_jsonl(
        emb_dir / "pca2.jsonl",
        ({"seg_id": i, "x": float(X2[i, 0]), "y": float(X2[i, 1])} for i in range(X2.shape[0])),
    )
    # store eval vectors (≤pca_dims) for reproducibility
    write_jsonl(
        emb_dir / "pca_eval.jsonl",
        ({"seg_id": i, "vec": X_eval[i].tolist()} for i in range(X_eval.shape[0])),
    )

    # clusters
    clusters_dir = out_dir / "clusters" / "speaker_embedding"
    write_json(
        clusters_dir / "kmeans_pca2.json",
        [{"seg_id": int(i), "cluster": int(c)} for i, c in enumerate(labels)],
    )
    write_json(
        clusters_dir / "kmeans_pca2_meta.json",
        {
            "k": int(k_star),
            "preprocess": str(cfg["preprocess"]),
            "pca_dims_eval": int(p_dim),
            "pca_eval_space": str(cfg["pca_eval"]),
            **meta,
        },
    )

    # viz
    viz_dir = out_dir / "viz"
    html_path = viz_dir / "quick_scatter.html"
    png_path = (viz_dir / "quick_scatter.png") if cfg["save_png"] else None
    if cfg["save_html"] or cfg["save_png"]:
        save_scatter_pca2(X2, labels, segments, html_path, png_path)
        print(f"[viz] html: {html_path}")
        if png_path and png_path.exists():
            print(f"[viz] png : {png_path}")

    print(f"[done] outputs under: {out_dir}")


if __name__ == "__main__":
    main()
