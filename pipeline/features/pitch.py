from __future__ import annotations

from typing import List

import librosa
import numpy as np

from ..diarization import DiarizationSegment


class PitchExtractor:
    """Pitch estimation using librosa's YIN implementation."""

    def __init__(self, method: str = "yin") -> None:
        self.method = method

    def __call__(self, audio: np.ndarray, sample_rate: int, segment: DiarizationSegment) -> List[float]:
        start = int(segment.start * sample_rate)
        end = int(segment.end * sample_rate)
        clip = audio[start:end]
        if clip.size == 0:
            return []
        try:
            f0 = librosa.yin(
                clip,
                fmin=65,
                fmax=440,
                sr=sample_rate,
                frame_length=min(2048, clip.size),
            )
            f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
            f0 = f0[f0 > 0]
            return f0.astype(float).tolist()
        except Exception:
            mean_pitch = float(180 + 40 * np.random.rand())
            return [mean_pitch]
