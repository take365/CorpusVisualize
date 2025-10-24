#!/usr/bin/env python3
"""
HMD Cluster Sweep (patched)

Speaker Embedding を 16kHz リサンプルの上で算出し、
StandardScaler → PCA(≤N) をクラスタリング空間として DBSCAN をグリッド探索。
- KMeans へのフォールバックは **なし**（要求どおり）
- 可視化は PCA 2D（同じスケーリング系列）
- 前処理モード（raw/zscore/l2/whiten/zscore_l2）を縦並び比較
- 正解ラベル（行頭の接頭辞＋コロン）を形状に反映
- Silhouette & Gap は参考値（KMeans による k 固定クラスタを評価）

出力: <output_root>/cluster_sweep/index.html

例:
  python HMD/src/hmd_cluster_sweep.py \
    --segments HMD/data/LD01-Dialogue-01/segments.jsonl \
    --audio    HMD/data/LD01-Dialogue-01.wav \
    --output-root HMD/data/LD01-Dialogue-01 \
    --pca-dims 50 \
    --expected-k 2
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize
from plotly import express as px
import plotly.graph_objects as go

# optional deps for on-the-fly speaker embedding
import torch
import torchaudio
from speechbrain.inference import EncoderClassifier  # SB 1.0+

# -----------------------------
# Gap Statistic（簡易実装）
# -----------------------------
def compute_gap_statistic(X: np.ndarray, k_values=range(1, 13), n_refs: int = 8, random_state: int = 0) -> List[float]:
    rng = np.random.default_rng(random_state)
    gaps: List[float] = []
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    for k in k_values:
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        km.fit(X)
        ref_inertias = []
        for _ in range(n_refs):
            ref = rng.uniform(low=X_min, high=X_max, size=X.shape)
            km_ref = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
            km_ref.fit(ref)
            ref_inertias.append(km_ref.inertia_)
        gap = float(np.log(np.mean(ref_inertias)) - np.log(km.inertia_))
        gaps.append(gap)
    return gaps

# -----------------------------
# 16kHz リサンプル + Speaker Embedding
# -----------------------------
def compute_speaker_embeddings(audio_path: Path, segments: List[dict]) -> np.ndarray:
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono 化
    target_sr = 16000
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        sr = target_sr
    embs: List[np.ndarray] = []
    with torch.inference_mode():
        for r in segments:
            s = float(r.get("start", 0.0))
            e = float(r.get("end", 0.0))
            s_i = max(0, int(sr * s))
            e_i = min(wav.shape[-1], int(sr * e))
            if e_i <= s_i:
                e_i = min(wav.shape[-1], s_i + int(0.05 * sr))  # 50ms フォールバック
            seg = wav[:, s_i:e_i]
            emb = classifier.encode_batch(seg).squeeze().detach().cpu().numpy()
            embs.append(emb)
    X = np.stack(embs)
    return X

# -----------------------------
# 正解ラベル（形状）読み込み
#   優先順: --ground-truth → segments と同じディレクトリの .txt → 親の <dirname>.txt
#   行頭『A:』『B:』『話者1:』などの 接頭辞+コロン を抽出
# -----------------------------
SYMBOLS = [
    "circle","square","triangle-up","diamond","x","star","cross",
    "triangle-down","triangle-left","triangle-right","hexagon","star-square",
]
_PREFIX_RE = re.compile(r"^\s*([^:：\s]+)\s*[:：]")

def resolve_gt_path(seg_path: Path, gt_arg: str | None) -> Path | None:
    if gt_arg:
        p = Path(gt_arg)
        return p if p.exists() else None
    cand1 = seg_path.with_suffix(".txt")
    if cand1.exists():
        return cand1
    parent = seg_path.parent
    cand2 = parent.parent / f"{parent.name}.txt"
    if cand2.exists():
        return cand2
    return None

def parse_prefix_label(line: str) -> str:
    m = _PREFIX_RE.match(line)
    return m.group(1) if m else ""

def load_ground_truth(gt_path: Path | None, n: int) -> Tuple[List[str], dict]:
    if gt_path and gt_path.exists():
        raw_lines = [ln.rstrip("\n") for ln in gt_path.read_text(encoding="utf-8").splitlines()]
    else:
        raw_lines = []
    labels = [parse_prefix_label(ln) for ln in raw_lines]
    if len(labels) != n:
        print(f"[WARN] Ground truth line count ({len(labels)}) differs from segments ({n})")
    if len(labels) < n:
        labels += [""] * (n - len(labels))
    labels = labels[:n]
    uniq = []
    for g in labels:
        if g and g not in uniq:
            uniq.append(g)
    sym_map = {g: SYMBOLS[i % len(SYMBOLS)] for i, g in enumerate(uniq)}
    return labels, sym_map

# -----------------------------
# 前処理モード
# -----------------------------
DEF_MODES = [
    ("raw", "RAW"),
    ("zscore", "Z-score"),
    ("l2", "L2 Norm"),
    ("whiten", "PCA Whitening"),
    ("zscore_l2", "Z-score + L2"),
]

def apply_preprocess(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return X
    elif mode == "zscore":
        return StandardScaler().fit_transform(X)
    elif mode == "l2":
        return normalize(X, norm="l2")
    elif mode == "zscore_l2":
        return normalize(StandardScaler().fit_transform(X), norm="l2")
    elif mode == "whiten":
        pca_w = PCA(whiten=True, random_state=0)
        return pca_w.fit_transform(X)
    else:
        raise ValueError(f"unknown preprocess mode: {mode}")

# -----------------------------
# DBSCAN グリッド探索（フォールバックなし）
#   - 入力: PCA(≤N) 空間（ユークリッド距離）
#   - スコア: (|k - expected_k|, noise_ratio) の辞書式最小
# -----------------------------

def iter_grid(eps_list: Iterable[float], ms_list: Iterable[int]):
    for eps in eps_list:
        for ms in ms_list:
            yield float(eps), int(ms)

def choose_dbscan_euclid(Xp: np.ndarray, expected_k: int,
                          eps_list=(0.3, 0.5, 0.7, 0.9),
                          ms_list=(3, 5, 8, 12)):
    best = None
    best_meta = None
    for eps, ms in iter_grid(eps_list, ms_list):
        labels = DBSCAN(eps=eps, min_samples=ms, metric="euclidean").fit_predict(Xp)
        ks = len({c for c in labels if c != -1})
        noise = float((labels == -1).mean())
        score = (abs(ks - max(0, expected_k)), noise)
        if (best is None) or (score < best[0]):
            best = (score, labels)
            best_meta = {"eps": eps, "min_samples": ms, "n_clusters": int(ks), "noise_ratio": noise}
    return best[1], best_meta

# -----------------------------
# メイン
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="HMD Cluster Sweep (patched, DBSCAN sweep, no fallback)")
    parser.add_argument("--segments", required=True, help="segments.jsonl")
    parser.add_argument("--audio", required=True, help="対応する音声ファイル (wav)")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pca-dims", type=int, default=50)
    parser.add_argument("--ground-truth", help="1行1発話のテキスト。行頭の『A:』『B:』などを正解ラベルとして使用。未指定なら自動探索")
    parser.add_argument("--expected-k", type=int, default=None, help="期待クラスタ数（未指定時はGTユニーク数>0ならそれ、なければ2）")
    parser.add_argument("--save-emb", action="store_true", help="計算した埋め込みを .npy で保存")
    parser.add_argument("--eps", type=float, nargs="*", default=[0.3, 0.5, 0.7, 0.9], help="DBSCAN eps 候補のリスト")
    parser.add_argument("--min-samples", type=int, nargs="*", default=[3, 5, 8, 12], help="DBSCAN min_samples 候補のリスト")
    args = parser.parse_args()

    out_dir = Path(args.output_root) / "cluster_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_path = Path(args.segments)
    audio_path = Path(args.audio)

    # セグメント読み込み
    records = [json.loads(l) for l in open(seg_path, encoding="utf-8") if l.strip()]
    texts = [r.get("text", "") for r in records]
    times = [f"{float(r.get('start',0)):.2f}-{float(r.get('end',0)):.2f}" for r in records]

    # ground truth (形状)
    gt_path = resolve_gt_path(seg_path, args.ground_truth)
    gt_labels, sym_map = load_ground_truth(gt_path, len(records))
    gt_symbols = [sym_map.get(g, "circle") if g else "circle" for g in gt_labels]

    # expected_k 推定（指定 > GTユニーク>0 > 既定値2）
    if args.expected_k is not None:
        expected_k = int(args.expected_k)
    else:
        uniq_nonempty = len([g for g in set(gt_labels) if g])
        expected_k = uniq_nonempty if uniq_nonempty > 0 else 2

    # 既存 embedding or 算出
    has_embed = any(isinstance(r.get("embedding"), (list, tuple)) and len(r.get("embedding")) > 0 for r in records)
    if has_embed:
        X_base = np.array([r["embedding"] for r in records], dtype=float)
        print(f"[info] use existing embeddings: shape={X_base.shape}")
    else:
        print("[info] embeddings not found; computing Speaker Embeddings (16kHz) ...")
        X_base = compute_speaker_embeddings(audio_path, records)
        print(f"[info] computed embeddings: shape={X_base.shape}")
        if args.save_emb:
            np.save(out_dir / "embeddings.npy", X_base)

    section_html: List[str] = []

    for mode_key, mode_name in DEF_MODES:
        # --- 前処理適用（PCA入力前） ---
        Xp_in = apply_preprocess(X_base, mode_key)

        # --- PCA圧縮（クラスタリング空間, ≤pca_dims） ---
        p_dim = int(min(args.pca_dims, Xp_in.shape[0] - 1, Xp_in.shape[1]))
        p_dim = max(2, p_dim)
        pca_high = PCA(n_components=p_dim, random_state=0)
        Xp = pca_high.fit_transform(Xp_in)

        # --- 可視化用 2D（同一系列の前処理から） ---
        pca2 = PCA(n_components=2, random_state=0)
        X2d = pca2.fit_transform(Xp_in)

        # --- DBSCAN（グリッド探索／ユークリッド／フォールバック無し） ---
        labels, meta = choose_dbscan_euclid(Xp, expected_k, eps_list=args.eps, ms_list=args.min_samples)
        print(f"[dbscan] mode={mode_key} meta={meta}")

        # --- KMeans(k=2)（比較用。シルエットは euclidean で評価） ---
        km2 = KMeans(n_clusters=2, n_init="auto", random_state=0).fit(Xp)
        lab2 = km2.labels_
        sil2 = float(silhouette_score(Xp, lab2, metric="euclidean"))

        # --- Silhouette (k=2..12) & Gap (k=1..12) ---
        sil_x = list(range(2, 13))
        sil_y: List[float] = []
        for k in sil_x:
            km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(Xp)
            sil = float(silhouette_score(Xp, km.labels_, metric="euclidean"))
            sil_y.append(sil)
        gaps = compute_gap_statistic(Xp, k_values=range(1, 13), n_refs=8, random_state=0)

        # --- 図: DBSCAN（色=labels, 形状=GT） ---
        df_db = pd.DataFrame(dict(
            x=X2d[:, 0], y=X2d[:, 1], label=labels, text=texts, time=times, gt=gt_labels, symbol=gt_symbols
        ))
        fig_db = px.scatter(
            df_db, x="x", y="y", color=df_db["label"].astype(str), symbol="symbol",
            hover_data=["text", "time", "gt"], title=f"[{mode_name}] DBSCAN(best) eps={meta['eps']}, min_samples={meta['min_samples']}, k={meta['n_clusters']}, noise={meta['noise_ratio']:.2f}"
        )
        fig_db.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=60, b=20))
        db_html = fig_db.to_html(include_plotlyjs="cdn", full_html=False)

        # --- 図: Silhouette & Gap ---
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Scatter(x=sil_x, y=sil_y, mode="lines+markers", name="Silhouette (k=2..12)"))
        fig_metrics.add_trace(go.Scatter(x=list(range(1, 13)), y=gaps, mode="lines+markers", name="Gap (k=1..12)", yaxis="y2"))
        fig_metrics.update_layout(
            title=f"[{mode_name}] Silhouette & Gap",
            xaxis_title="k",
            yaxis=dict(title="Silhouette"),
            yaxis2=dict(title="Gap", overlaying="y", side="right"),
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        metrics_html = fig_metrics.to_html(include_plotlyjs="cdn", full_html=False)

        # --- 図: KMeans(k=2)（参考） ---
        df2 = pd.DataFrame(dict(x=X2d[:, 0], y=X2d[:, 1], label=lab2, text=texts, time=times, gt=gt_labels, symbol=gt_symbols))
        fig2 = px.scatter(
            df2, x="x", y="y", color=df2["label"].astype(str), symbol="symbol",
            hover_data=["text", "time", "gt"], title=f"[{mode_name}] KMeans (k=2), silhouette={sil2:.3f}"
        )
        fig2.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
        k2_html = fig2.to_html(include_plotlyjs="cdn", full_html=False)

        # --- 形状（正解ラベル）対応表 ---
        legend_html = ""
        if sym_map:
            legend_html = "<p><b>形状（正解話者）:</b><br>" + "<br>".join(f"{g or '-'}: {sym}" for g, sym in sym_map.items()) + "</p>"

        # --- セクション合成 ---
        section_html = [
            f"<h2>{mode_name}</h2>",
            f"<p><b>DBSCAN meta:</b> {json.dumps(meta)}</p>",
            legend_html,
            db_html,
            metrics_html,
            k2_html,
            "<hr>"
        ]

        # 追記
        (out_dir / f"section_{mode_key}.html").write_text("\n".join(section_html), encoding="utf-8")

    # --- index.html 出力 ---
    index_parts = []
    for mode_key, _ in DEF_MODES:
        p = out_dir / f"section_{mode_key}.html"
        if p.exists():
            index_parts.append(p.read_text(encoding="utf-8"))
    (out_dir / "index.html").write_text("\n".join(index_parts), encoding="utf-8")
    print(f"[done] index: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
