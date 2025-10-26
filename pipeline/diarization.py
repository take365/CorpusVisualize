from __future__ import annotations

import os
import string
import warnings
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


from .quick_io import QuickArtifactsError, load_quick_artifacts


logger = logging.getLogger(__name__)

@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str


class EnergyBasedDiarizer:
    """Simple energy-based diarizer that alternates speaker labels."""

    def __init__(self, min_seg: float, max_seg: float, energy_threshold: float = 0.02):
        self.min_seg = min_seg
        self.max_seg = max_seg
        self.energy_threshold = energy_threshold

    def __call__(self, audio: np.ndarray, sample_rate: int) -> List[DiarizationSegment]:
        frame_length = int(sample_rate * self.min_seg)
        hop_length = max(1, frame_length // 2)
        frames = []
        for start in range(0, len(audio), hop_length):
            end = min(len(audio), start + frame_length)
            if end - start < frame_length // 2:
                break
            chunk = audio[start:end]
            energy = float(np.sqrt(np.mean(chunk ** 2)))
            if energy < self.energy_threshold:
                continue
            frames.append((start / sample_rate, end / sample_rate))

        if not frames:
            total = len(audio) / sample_rate
            frames = [(0.0, min(total, self.max_seg))]

        segments: List[DiarizationSegment] = []
        speaker_toggle = 0
        last_end = 0.0
        for start, end in frames:
            start = max(start, last_end)
            if end - start < self.min_seg:
                end = start + self.min_seg
            if end - start > self.max_seg:
                end = start + self.max_seg
            segments.append(
                DiarizationSegment(
                    start=float(start),
                    end=float(end),
                    speaker="A" if speaker_toggle % 2 == 0 else "B",
                )
            )
            speaker_toggle += 1
            last_end = end
        return segments


class PyannoteDiarizer:
    """pyannote.audio backed diarizer with graceful fallback handling."""

    def __init__(
        self,
        min_seg: float,
        max_seg: float,
        model: str = "pyannote/speaker-diarization-3.1",
        auth_token: Optional[str] = None,
        overrides: Optional[Dict[str, object]] = None,
    ) -> None:
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pyannote.audio がインストールされていません。requirements に追加して下さい。"
            ) from exc

        token = auth_token or os.getenv("PYANNOTE_AUDIO_AUTH_TOKEN")
        token = token or os.getenv("HUGGINGFACE_TOKEN")
        token = token or os.getenv("HF_TOKEN")
        if token is None:
            raise RuntimeError(
                "pyannote.audio の認証トークンが未設定です。"
                " `PYANNOTE_AUDIO_AUTH_TOKEN` もしくは HuggingFace token を環境変数に設定してください。"
            )

        self.min_seg = min_seg
        self.max_seg = max_seg
        self._Pipeline = Pipeline
        base_pipeline = Pipeline.from_pretrained(model, use_auth_token=token)
        if base_pipeline is None:
            raise RuntimeError(
                "pyannote.audio パイプラインの取得に失敗しました。モデルページで利用規約に同意し、"
                "アクセス権のあるトークンを使用しているか確認してください。"
            )
        available_params = getattr(base_pipeline, "parameters", lambda: {})()
        filtered: Dict[str, object] = {}
        invalid: List[str] = []
        for key, value in (overrides or {}).items():
            if isinstance(value, dict):
                section = available_params.get(key)
                if not isinstance(section, dict):
                    invalid.append(key)
                    continue
                valid_inner: Dict[str, object] = {}
                for inner_key, inner_value in value.items():
                    if inner_key in section:
                        valid_inner[inner_key] = inner_value
                    else:
                        invalid.append(f"{key}.{inner_key}")
                if valid_inner:
                    filtered[key] = valid_inner
            else:
                invalid.append(key)
        if invalid:
            warnings.warn(
                "pyannote.audio で未サポートのパラメータが指定されたため無視しました: "
                + ", ".join(sorted(set(invalid)))
            )
        self._overrides = filtered
        try:
            self._pipeline = base_pipeline.instantiate(self._overrides)
        except Exception as exc:
            raise RuntimeError(f"pyannote.audio パイプラインの初期化に失敗しました: {exc}") from exc
        self._speaker_alias: Dict[str, str] = {}
        self._alphabet_iter = iter(string.ascii_uppercase)

    def _alias(self, speaker: str) -> str:
        if speaker not in self._speaker_alias:
            try:
                alias = next(self._alphabet_iter)
            except StopIteration:
                alias = f"S{len(self._speaker_alias)}"
            self._speaker_alias[speaker] = alias
        return self._speaker_alias[speaker]

    def __call__(self, audio: np.ndarray, sample_rate: int) -> List[DiarizationSegment]:
        import torch

        if audio.ndim != 1:
            audio = np.asarray(audio).reshape(-1)
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        if waveform.abs().max() > 1:  # 正規化で pyannote 安定化
            waveform = waveform / waveform.abs().max()

        diarization = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})

        segments: List[DiarizationSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = float(turn.start or 0.0)
            end = float(turn.end or start)
            if end <= start:
                continue
            label = self._alias(str(speaker or "UNK"))

            seg_start = start
            while seg_start < end:
                seg_end = min(end, seg_start + self.max_seg)
                if seg_end - seg_start < self.min_seg and seg_end != end:
                    seg_end = min(end, seg_start + self.min_seg)
                if seg_end <= seg_start:
                    break
                segments.append(
                    DiarizationSegment(
                        start=float(seg_start),
                        end=float(seg_end),
                        speaker=label,
                    )
                )
                seg_start = seg_end
        segments.sort(key=lambda seg: seg.start)
        return segments



class QuickClusterDiarizer:
    """Reuse Quick diarization clusters produced ahead of the pipeline."""

    def __init__(self, data_root: Optional[str] = None) -> None:
        self._data_root = data_root
        self._conversation_id: Optional[str] = None

    def set_context(self, audio_path, conversation_id: Optional[str] = None) -> None:
        base = conversation_id
        if not base:
            stem = getattr(audio_path, "stem", None)
            if stem is None:
                stem = os.path.splitext(os.path.basename(str(audio_path)))[0]
            base = stem
        self._conversation_id = str(base)

    def __call__(self, audio: np.ndarray, sample_rate: int) -> List[DiarizationSegment]:
        if self._conversation_id is None:
            raise RuntimeError(
                "QuickClusterDiarizer requires conversation context. "
                "Call set_context() with the audio path before diarization."
            )

        try:
            artifacts = load_quick_artifacts(self._conversation_id, data_root=self._data_root)
        except QuickArtifactsError as exc:
            raise RuntimeError(
                f"Quick artifacts unavailable for '{self._conversation_id}': {exc}"
            ) from exc

        speaker_map: Dict[int, str] = {}
        segments: List[DiarizationSegment] = []
        for seg in artifacts.segments:
            cluster = artifacts.cluster_map.get(seg.index)
            if cluster is None:
                raise RuntimeError(f"Cluster label missing for seg_id={seg.index}")
            label = speaker_map.setdefault(cluster, _cluster_label(cluster))
            segments.append(
                DiarizationSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    speaker=label,
                )
            )

        logger.info(
            "QuickClusterDiarizer using %s (clusters=%d, segments=%d)",
            artifacts.segments_path,
            artifacts.cluster_k,
            len(segments),
        )
        return segments


def _cluster_label(cluster_index: int) -> str:
    alphabet = string.ascii_uppercase
    if 0 <= cluster_index < len(alphabet):
        return alphabet[cluster_index]
    return f"S{cluster_index}"


def get_diarizer(
    name: str,
    min_seg: float,
    max_seg: float,
    overrides: Optional[Dict[str, object]] = None,
    quick_data_root: Optional[str] = None,
) -> Callable[[np.ndarray, int], List[DiarizationSegment]]:
    name = (name or "").lower()
    energy_aliases = {"energy_basic", "energy", "energy_split", "dummy"}
    if name in energy_aliases:
        return EnergyBasedDiarizer(min_seg=min_seg, max_seg=max_seg)

    if name in {"pyannote", "pyannote.audio", "pyannote_audio"}:
        try:
            return PyannoteDiarizer(min_seg=min_seg, max_seg=max_seg, overrides=overrides)
        except RuntimeError as exc:
            warnings.warn(f"pyannote.audio の初期化に失敗したため EnergyBased にフォールバックします: {exc}")
            return EnergyBasedDiarizer(min_seg=min_seg, max_seg=max_seg)

    if name in {"quick_cluster", "quick"}:
        return QuickClusterDiarizer(data_root=quick_data_root)

    raise ValueError(f"Unsupported diarization method: {name}")
