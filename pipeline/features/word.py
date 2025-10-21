from __future__ import annotations

from typing import Iterable, List

import numpy as np

from ..types import WordSchema
from ..text import ProsodyInfo, analyze_prosody

try:  # pragma: no cover - optional heavy dependency
    import pyworld
except Exception:  # pragma: no cover
    pyworld = None  # type: ignore


class WordFeatureExtractor:
    def __init__(self, prosody_backend: str = "unidic", pitch_backend: str = "pyworld") -> None:
        self.prosody_backend = prosody_backend
        self.pitch_backend = pitch_backend

    def __call__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        words: Iterable[object],
        fallback_valence: float,
        fallback_arousal: float,
    ) -> List[WordSchema]:
        results: List[WordSchema] = []
        for word in words:
            text = getattr(word, "text", "")
            start = float(getattr(word, "start", 0.0))
            end = float(getattr(word, "end", start))
            if end <= start:
                end = start + 0.2

            start_idx = max(int(start * sample_rate), 0)
            end_idx = min(int(end * sample_rate), audio.shape[-1])
            clip = audio[start_idx:end_idx]

            if clip.size == 0:
                margin = sample_rate // 40
                clip = audio[max(0, start_idx - margin) : min(audio.shape[-1], end_idx + margin)]

            pitch_curve: List[float] = []
            pitch_mean = None
            if clip.size:
                if self.pitch_backend == "pyworld" and pyworld is not None:
                    pitch_curve = self._pitch_pyworld(clip, sample_rate)
                else:
                    pitch_curve = self._pitch_librosa(clip, sample_rate)
                if pitch_curve:
                    pitch_mean = float(np.mean(pitch_curve))

            loudness = float(np.sqrt(np.mean(clip ** 2))) if clip.size else 0.0
            duration = max(end - start, 1e-3)
            tempo = len(str(text).replace(" ", "")) / duration

            if self.prosody_backend == "unidic":
                prosody = analyze_prosody(text)
            else:
                prosody = ProsodyInfo(kana=None, accent=None)

            results.append(
                WordSchema(
                    text=str(text),
                    kana=prosody.kana,
                    accent=prosody.accent,
                    start=start,
                    end=end,
                    pitch_mean=pitch_mean,
                    pitch_curve=pitch_curve,
                    loudness=float(loudness),
                    tempo=float(tempo),
                    valence=fallback_valence,
                    arousal=fallback_arousal,
                )
            )
        return results

    @staticmethod
    def _pitch_pyworld(audio: np.ndarray, sample_rate: int) -> List[float]:  # pragma: no cover - heavy path
        if pyworld is None:
            return []
        f0, t = pyworld.dio(audio.astype(np.float64), sample_rate)
        refined = pyworld.stonemask(audio.astype(np.float64), f0, t, sample_rate)
        return [float(v) for v in refined if np.isfinite(v) and v > 0]

    @staticmethod
    def _pitch_librosa(audio: np.ndarray, sample_rate: int) -> List[float]:  # pragma: no cover - fallback
        try:
            import librosa

            f0 = librosa.yin(audio, fmin=65, fmax=440, sr=sample_rate)
            f0 = np.array(f0)
            f0 = f0[np.isfinite(f0) & (f0 > 0)]
            return f0.astype(float).tolist()
        except Exception:
            return []
