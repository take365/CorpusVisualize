from __future__ import annotations

import numpy as np

from ..diarization import DiarizationSegment


class EmotionExtractor:
    """Estimate coarse emotion distribution from simple acoustic cues."""

    def __init__(self, method: str = "energy_based") -> None:
        self.method = method

    def __call__(self, audio: np.ndarray, sample_rate: int, segment: DiarizationSegment) -> dict:
        start = int(segment.start * sample_rate)
        end = int(segment.end * sample_rate)
        clip = audio[start:end]
        if clip.size == 0:
            clip = audio
        energy = float(np.sqrt(np.mean(clip ** 2))) if clip.size else 0.0
        pitch_var = float(np.var(clip)) if clip.size else 0.0
        base = np.array([max(energy, 1e-3), pitch_var + 0.1, 0.2, 0.3])
        probs = base / base.sum()
        return {
            "neutral": float(probs[0]),
            "joy": float(probs[1]),
            "anger": float(probs[2]),
            "sad": float(probs[3]),
        }
