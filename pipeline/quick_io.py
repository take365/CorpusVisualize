from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = os.getenv("HMD_DATA_ROOT", "HMD/data")


class QuickArtifactsError(RuntimeError):
    """Raised when Quick diarization/ASR artifacts are missing or inconsistent."""


@dataclass
class QuickSegment:
    index: int
    start: float
    end: float
    text: str


@dataclass
class QuickArtifacts:
    conversation_id: str
    root: Path
    segments_path: Path
    clusters_path: Path
    segments: List[QuickSegment]
    cluster_map: Dict[int, int]
    cluster_k: int
    words_path: Optional[Path]
    words_by_segment: Dict[int, List[Dict[str, float]]]


def _normalize_root(root: Optional[os.PathLike[str] | str]) -> Path:
    base = Path(root) if root else Path(_DEFAULT_ROOT)
    return base.expanduser().resolve()


def load_quick_artifacts(conversation_id: str, *, data_root: Optional[os.PathLike[str] | str] = None) -> QuickArtifacts:
    """Load Quick diarization artifacts for a given conversation."""

    if not conversation_id:
        raise QuickArtifactsError("conversation_id is required to locate Quick artifacts")

    root = _normalize_root(data_root)
    try:
        return _load_quick_artifacts_cached(str(root), conversation_id)
    except FileNotFoundError as exc:  # pragma: no cover - dependent on runtime files
        raise QuickArtifactsError(str(exc)) from exc


@lru_cache(maxsize=32)
def _load_quick_artifacts_cached(root: str, conversation_id: str) -> QuickArtifacts:
    root_path = Path(root)
    conv_dir = root_path / conversation_id

    segments_path = conv_dir / "segments.jsonl"
    if not segments_path.exists():
        raise FileNotFoundError(f"Quick segments not found: {segments_path}")

    segments: List[QuickSegment] = []
    with segments_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            start = float(payload.get("start", 0.0))
            end = float(payload.get("end", start))
            segments.append(
                QuickSegment(
                    index=idx,
                    start=start,
                    end=end,
                    text=str(payload.get("text", "")),
                )
            )

    clusters_path = conv_dir / "clusters" / "speaker_embedding" / "kmeans_pca2.json"
    if not clusters_path.exists():
        raise FileNotFoundError(f"Quick cluster assignments not found: {clusters_path}")

    with clusters_path.open("r", encoding="utf-8") as f:
        cluster_payload = json.load(f)

    cluster_map: Dict[int, int] = {}
    if isinstance(cluster_payload, list):
        for entry in cluster_payload:
            if not isinstance(entry, dict):
                continue
            seg_id = int(entry.get("seg_id"))
            cluster = int(entry.get("cluster"))
            cluster_map[seg_id] = cluster
    elif isinstance(cluster_payload, dict):
        for seg_id_str, cluster in cluster_payload.items():
            try:
                seg_id = int(seg_id_str)
            except (TypeError, ValueError):
                continue
            cluster_map[seg_id] = int(cluster)
    else:
        raise QuickArtifactsError(f"Unsupported cluster payload format: {type(cluster_payload).__name__}")

    expected_seg_ids = {seg.index for seg in segments}
    missing = sorted(expected_seg_ids.difference(cluster_map.keys()))
    if missing:
        raise QuickArtifactsError(
            "Cluster assignments missing for seg_id(s): " + ", ".join(map(str, missing))
        )

    meta_path = clusters_path.with_name("kmeans_pca2_meta.json")
    cluster_k = None
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if isinstance(meta, dict):
            if isinstance(meta.get("best"), dict) and "k" in meta["best"]:
                try:
                    cluster_k = int(meta["best"]["k"])
                except (TypeError, ValueError):
                    cluster_k = None
            elif "k" in meta:
                try:
                    cluster_k = int(meta["k"])
                except (TypeError, ValueError):
                    cluster_k = None

    if cluster_k is None:
        cluster_k = len({c for c in cluster_map.values()})

    words_path = conv_dir / "words.jsonl"
    words_by_segment = _load_words(words_path, segments)

    return QuickArtifacts(
        conversation_id=conversation_id,
        root=root_path,
        segments_path=segments_path,
        clusters_path=clusters_path,
        segments=segments,
        cluster_map=cluster_map,
        cluster_k=cluster_k,
        words_path=words_path if words_path.exists() else None,
        words_by_segment=words_by_segment,
    )


def _load_words(words_path: Path, segments: List[QuickSegment]) -> Dict[int, List[Dict[str, float]]]:
    if not words_path.exists():
        return {}

    seg_ranges: List[Tuple[int, float, float]] = [
        (seg.index, seg.start, seg.end) for seg in segments
    ]
    words_by_segment: Dict[int, List[Dict[str, float]]] = {}
    tolerance = 1e-3

    with words_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            word_text = payload.get("word") or payload.get("text")
            if not word_text:
                continue
            word_start = float(payload.get("start", 0.0))
            word_end = float(payload.get("end", word_start))
            seg_start = float(payload.get("segment_start", payload.get("start", 0.0)))
            seg_end = float(payload.get("segment_end", payload.get("end", 0.0)))

            seg_index = _match_segment(seg_ranges, seg_start, seg_end, tolerance)
            if seg_index is None:
                logger.debug(
                    "Unable to match Quick word '%s' (%.3f-%.3f) to any segment", word_text, seg_start, seg_end
                )
                continue
            words_by_segment.setdefault(seg_index, []).append(
                {
                    "text": str(word_text),
                    "start": word_start,
                    "end": word_end,
                }
            )

    for word_list in words_by_segment.values():
        word_list.sort(key=lambda w: w.get("start", 0.0))

    return words_by_segment


def _match_segment(
    seg_ranges: Iterable[Tuple[int, float, float]],
    start: float,
    end: float,
    tolerance: float,
) -> Optional[int]:
    for idx, seg_start, seg_end in seg_ranges:
        if abs(seg_start - start) <= tolerance and abs(seg_end - end) <= tolerance:
            return idx
    # fallback: choose the segment with maximum overlap
    best_idx: Optional[int] = None
    best_overlap = 0.0
    for idx, seg_start, seg_end in seg_ranges:
        overlap = min(seg_end, end) - max(seg_start, start)
        if overlap > best_overlap and overlap > 0:
            best_idx = idx
            best_overlap = overlap
    return best_idx


__all__ = [
    "QuickArtifacts",
    "QuickArtifactsError",
    "QuickSegment",
    "load_quick_artifacts",
]
