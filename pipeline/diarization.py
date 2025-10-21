from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


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


def get_diarizer(name: str, min_seg: float, max_seg: float) -> EnergyBasedDiarizer:
    name = name.lower()
    if name in {"energy_split", "energy", "dummy"}:
        return EnergyBasedDiarizer(min_seg=min_seg, max_seg=max_seg)
    raise ValueError(f"Unsupported diarization method: {name}")
