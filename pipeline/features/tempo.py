from __future__ import annotations

import numpy as np

from ..diarization import DiarizationSegment


class TempoExtractor:
    """Estimate speaking tempo by combining text length with duration."""

    def __init__(self, method: str = "chars_per_sec") -> None:
        self.method = method

    def __call__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segment: DiarizationSegment,
        transcript: str,
    ) -> float:
        duration = max(1e-3, segment.end - segment.start)
        if self.method == "chars_per_sec":
            speed = len(transcript) / duration
        elif self.method == "vad_ratio":
            clip = audio[int(segment.start * sample_rate) : int(segment.end * sample_rate)]
            if clip.size:
                voiced = float(np.mean(np.abs(clip) > 0.02))
            else:
                voiced = 0.0
            speed = voiced
        else:
            speed = len(transcript.split()) / duration
        return float(speed)
