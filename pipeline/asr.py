from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from typing import Dict, Iterable, List, Optional

import numpy as np
from rich.progress import track

from .diarization import DiarizationSegment
from .quick_io import QuickArtifactsError, load_quick_artifacts

try:  # pragma: no cover - optional dependency
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - lazy import handling
    WhisperModel = None  # type: ignore[misc]

try:  # pragma: no cover - optional dependency
    import ctranslate2
except Exception:  # pragma: no cover - lazy import handling
    ctranslate2 = None  # type: ignore[assignment]


@dataclass
class WordTiming:
    text: str
    start: float
    end: float


@dataclass
class TranscribedSegment:
    text: str
    words: List[WordTiming]
    raw_text: str


_SAMPLE_SENTENCES = [
    "なるほど、それで進めましょう。",
    "一度整理してから共有します。",
    "次の案も検討しておきます。",
    "その視点は面白いですね。",
    "確認が終わったら連絡します。",
]

_MODEL_CACHE: Dict[str, WhisperModel] = {}

logger = logging.getLogger(__name__)


def _sentence_for_segment(segment: DiarizationSegment, conversation_id: str) -> str:
    key = f"{conversation_id}-{segment.start:.2f}-{segment.end:.2f}-{segment.speaker}"
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
    return _SAMPLE_SENTENCES[h % len(_SAMPLE_SENTENCES)]


def _select_compute_type() -> str:
    if ctranslate2 is not None:
        try:
            if hasattr(ctranslate2, "get_cuda_device_count"):
                if ctranslate2.get_cuda_device_count() > 0:
                    return "float16"
            elif hasattr(ctranslate2, "get_device_count"):
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
    asr_results: List[Dict[str, object]],
) -> List[TranscribedSegment]:
    outputs: List[TranscribedSegment] = []
    for segment in diar_segments:
        candidate_words: List[WordTiming] = []
        raw_candidates: List[str] = []
        for res in asr_results:
            overlap = min(segment.end, float(res["end"])) - max(segment.start, float(res["start"]))
            if overlap <= 0:
                continue
            raw_text = str(res.get("text", "")).strip()
            if raw_text:
                raw_candidates.append(raw_text)
            for word in res.get("words", []):
                w_start = float(word.get("start", res["start"]))
                w_end = float(word.get("end", res["end"]))
                if w_end <= segment.start or w_start >= segment.end:
                    continue
                trimmed_text = str(word.get("text", "")).strip()
                if not trimmed_text:
                    continue
                candidate_words.append(
                    WordTiming(
                        text=trimmed_text,
                        start=max(segment.start, w_start),
                        end=min(segment.end, w_end),
                    )
                )
        candidate_words.sort(key=lambda w: w.start)
        deduped_raw = []
        for raw in raw_candidates:
            if raw and raw not in deduped_raw:
                deduped_raw.append(raw)
        candidate_raw_text = " ".join(deduped_raw).strip()

        if candidate_words:
            text_value = " ".join(w.text for w in candidate_words).strip()
            outputs.append(
                TranscribedSegment(
                    text=text_value,
                    words=candidate_words,
                    raw_text=(candidate_raw_text or text_value),
                )
            )
            continue

        best_text: Optional[str] = None
        best_overlap = 0.0
        for res in asr_results:
            overlap = min(segment.end, float(res["end"])) - max(segment.start, float(res["start"]))
            if overlap <= 0:
                continue
            if overlap > best_overlap:
                best_overlap = overlap
                best_text = str(res.get("text", "")).strip()
        final_text = best_text or ""
        outputs.append(
            TranscribedSegment(
                text=final_text,
                words=[],
                raw_text=(candidate_raw_text or final_text),
            )
        )
    return outputs


def _transcribe_with_whisper(
    audio: np.ndarray,
    sample_rate: int,
    language: str,
    method: str,
) -> List[Dict[str, object]]:
    size = _resolve_model_size(method)
    model = _load_whisper_model(size)
    float_audio = np.asarray(audio, dtype=np.float32)
    segments, _info = model.transcribe(
        float_audio,
        language=language,
        beam_size=5,
        vad_filter=True,
        temperature=0.0,
        word_timestamps=True,
    )
    results: List[Dict[str, object]] = []
    for seg in segments:
        text_content = seg.text.strip()
        if not text_content:
            continue
        words_payload: List[Dict[str, object]] = []
        if getattr(seg, "words", None):
            for word in seg.words:
                w_text = getattr(word, "word", "").strip()
                if not w_text:
                    continue
                w_start = float(getattr(word, "start", seg.start))
                w_end = float(getattr(word, "end", seg.end))
                if w_end <= w_start:
                    continue
                words_payload.append({"text": w_text, "start": w_start, "end": w_end})
        results.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text_content,
                "words": words_payload,
            }
        )
    return results


def _build_quick_asr_results(conversation_id: str) -> List[Dict[str, object]]:
    artifacts = load_quick_artifacts(conversation_id)
    results: List[Dict[str, object]] = []
    total_words = 0

    for seg in artifacts.segments:
        words_payload: List[Dict[str, object]] = []
        for word in artifacts.words_by_segment.get(seg.index, []):
            text = str(word.get("text", "")).strip()
            if not text:
                continue
            start = float(word.get("start", seg.start))
            end = float(word.get("end", start))
            if end <= start:
                continue
            words_payload.append({"text": text, "start": start, "end": end})
        words_payload.sort(key=lambda w: w.get("start", 0.0))
        total_words += len(words_payload)
        results.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": str(seg.text or "").strip(),
                "words": words_payload,
            }
        )

    logger.info(
        "Quick ASR reuse from %s (segments=%d, words=%d)",
        artifacts.segments_path,
        len(results),
        total_words,
    )
    return results


def _dummy_words(text: str, start: float, end: float) -> List[WordTiming]:
    tokens = [tok for tok in text.replace("、", " ").replace("。", " ").split() if tok]
    if not tokens:
        return []
    duration = max(end - start, 0.5)
    step = duration / len(tokens)
    words: List[WordTiming] = []
    cursor = start
    for token in tokens:
        words.append(WordTiming(text=token, start=cursor, end=min(end, cursor + step)))
        cursor += step
    return words


def transcribe(
    segments: Iterable[DiarizationSegment],
    conversation_id: str,
    method: str = "dummy",
    *,
    audio: Optional[np.ndarray] = None,
    sample_rate: Optional[int] = None,
    language: str = "ja",
) -> List[TranscribedSegment]:
    """Return transcripts per diarized segment with optional word timings."""
    segment_list = list(segments)
    if not segment_list:
        return []

    method = (method or "dummy").lower()

    if method == "dummy":
        results: List[TranscribedSegment] = []
        for segment in track(segment_list, description="Transcribing", transient=True):
            sentence = _sentence_for_segment(segment, conversation_id)
            words = _dummy_words(sentence, segment.start, segment.end)
            results.append(TranscribedSegment(text=sentence, words=words, raw_text=sentence))
        return results

    if method == "quick":
        try:
            asr_results = _build_quick_asr_results(conversation_id)
        except QuickArtifactsError as exc:
            raise RuntimeError(
                f"Quick ASR artifacts unavailable for '{conversation_id}': {exc}"
            ) from exc
        if not asr_results:
            return [TranscribedSegment(text="", words=[], raw_text="") for _ in segment_list]
        return _align_transcripts(segment_list, asr_results)

    if method.startswith("whisper"):
        if audio is None or sample_rate is None:
            raise ValueError("Audio data and sample rate are required for whisper ASR")
        asr_results = _transcribe_with_whisper(audio, sample_rate, language, method)
        if not asr_results:
            return [TranscribedSegment(text="", words=[], raw_text="") for _ in segment_list]
        return _align_transcripts(segment_list, asr_results)

    raise ValueError(f"Unsupported ASR method: {method}")
