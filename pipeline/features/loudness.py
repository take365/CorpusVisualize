from __future__ import annotations

import numpy as np

from ..diarization import DiarizationSegment


class LoudnessExtractor:
    """Compute RMS-based loudness score normalized to 0-1 range."""

    def __init__(self, method: str = "rms") -> None:
        self.method = method

    def __call__(self, audio: np.ndarray, sample_rate: int, segment: DiarizationSegment) -> float:
        start = int(segment.start * sample_rate)
        end = int(segment.end * sample_rate)
        clip = audio[start:end]
        if clip.size == 0:
            clip = audio
        rms = float(np.sqrt(np.mean(clip ** 2))) if clip.size else 0.0
        return float(min(1.0, rms * 10))
