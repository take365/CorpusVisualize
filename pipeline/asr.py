from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional

import numpy as np
from rich.progress import track

from .diarization import DiarizationSegment

try:  # pragma: no cover - optional dependency
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - lazy import handling
    WhisperModel = None  # type: ignore[misc]

try:  # pragma: no cover - optional dependency
    import ctranslate2
except Exception:  # pragma: no cover - lazy import handling
    ctranslate2 = None  # type: ignore[assignment]

_SAMPLE_SENTENCES = [
    "なるほど、それで進めましょう。",
    "一度整理してから共有します。",
    "次の案も検討しておきます。",
    "その視点は面白いですね。",
    "確認が終わったら連絡します。",
]

_MODEL_CACHE: Dict[str, WhisperModel] = {}


def _sentence_for_segment(segment: DiarizationSegment, conversation_id: str) -> str:
    key = f"{conversation_id}-{segment.start:.2f}-{segment.end:.2f}-{segment.speaker}"
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
    return _SAMPLE_SENTENCES[h % len(_SAMPLE_SENTENCES)]


def _select_compute_type() -> str:
    if ctranslate2 is not None:
        try:
            if ctranslate2.get_device_count("cuda") > 0:
                return "float16"
        except Exception:  # pragma: no cover - device query failure
            pass
    return "int8"


def _resolve_model_size(method: str) -> str:
    mapping = {
        "whisper": "medium",
        "whisperx": "medium",
        "whisper-medium": "medium",
        "whisper_medium": "medium",
        "whisper-small": "small",
        "whisper_small": "small",
    }
    if method in mapping:
        return mapping[method]
    if method.startswith("whisper-"):
        return method.split("-", 1)[1]
    return method


def _load_whisper_model(size: str) -> WhisperModel:
    if WhisperModel is None:  # pragma: no cover - dependency not installed
        raise RuntimeError("faster-whisper is not installed. Please install it to use whisper ASR.")
    if size not in _MODEL_CACHE:
        compute_type = _select_compute_type()
        device = "cuda" if compute_type == "float16" else "cpu"
        try:
            _MODEL_CACHE[size] = WhisperModel(
                model_size_or_path=size,
                device=device,
                compute_type=compute_type,
            )
        except Exception:
            if device == "cuda":
                _MODEL_CACHE[size] = WhisperModel(
                    model_size_or_path=size,
                    device="cpu",
                    compute_type="int8",
                )
            else:
                raise
    return _MODEL_CACHE[size]


def _align_transcripts(
    diar_segments: List[DiarizationSegment],
    asr_results: List[Dict[str, float]],
) -> List[str]:
    transcripts: List[str] = []
    for segment in diar_segments:
        collected: List[str] = []
        best_match: Optional[str] = None
        best_overlap = 0.0
        for res in asr_results:
            overlap = min(segment.end, res["end"]) - max(segment.start, res["start"])
            if overlap <= 0:
                continue
            collected.append(res["text"].strip())
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = res["text"].strip()
        if collected:
            transcripts.append(" ".join(filter(None, collected)).strip())
        else:
            transcripts.append(best_match or "")
    return transcripts


def _transcribe_with_whisper(
    audio: np.ndarray,
    sample_rate: int,
    language: str,
    method: str,
) -> List[Dict[str, float]]:
    size = _resolve_model_size(method)
    model = _load_whisper_model(size)
    float_audio = np.asarray(audio, dtype=np.float32)
    segments, _info = model.transcribe(
        float_audio,
        language=language,
        beam_size=5,
        vad_filter=True,
        temperature=0.0,
    )
    results: List[Dict[str, float]] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        results.append({"start": float(seg.start), "end": float(seg.end), "text": text})
    return results


def transcribe(
    segments: Iterable[DiarizationSegment],
    conversation_id: str,
    method: str = "dummy",
    *,
    audio: Optional[np.ndarray] = None,
    sample_rate: Optional[int] = None,
    language: str = "ja",
) -> List[str]:
    """Return transcripts per diarized segment."""
    segment_list = list(segments)
    if not segment_list:
        return []

    method = (method or "dummy").lower()

    if method == "dummy":
        return [
            _sentence_for_segment(segment, conversation_id)
            for segment in track(segment_list, description="Transcribing", transient=True)
        ]

    if method.startswith("whisper"):
        if audio is None or sample_rate is None:
            raise ValueError("Audio data and sample rate are required for whisper ASR")
        asr_results = _transcribe_with_whisper(audio, sample_rate, language, method)
        if not asr_results:
            return [""] * len(segment_list)
        return _align_transcripts(segment_list, asr_results)

    raise ValueError(f"Unsupported ASR method: {method}")
