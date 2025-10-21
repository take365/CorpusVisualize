from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List

import pandas as pd

from .types import SegmentSchema, SpeakerAggregate


def aggregate_speakers(segments: Iterable[SegmentSchema]) -> List[SpeakerAggregate]:
    totals = defaultdict(lambda: {
        "duration": 0.0,
        "loudness": [],
        "tempo": [],
        "emotions": defaultdict(float),
        "count": 0,
    })

    for seg in segments:
        bucket = totals[seg.speaker]
        duration = seg.end - seg.start
        bucket["duration"] += duration
        bucket["count"] += 1
        bucket["loudness"].append(seg.loudness)
        bucket["tempo"].append(seg.tempo)
        for name, score in seg.emotion.items():
            bucket["emotions"][name] += score

    aggregates: List[SpeakerAggregate] = []
    for speaker, data in totals.items():
        loudness_avg = sum(data["loudness"]) / max(1, len(data["loudness"]))
        tempo_avg = sum(data["tempo"]) / max(1, len(data["tempo"]))
        dominant_emotion = max(data["emotions"], key=data["emotions"].get)
        aggregates.append(
            SpeakerAggregate(
                speaker=speaker,
                segment_count=data["count"],
                total_duration=data["duration"],
                avg_loudness=loudness_avg,
                avg_tempo=tempo_avg,
                dominant_emotion=dominant_emotion,
            )
        )
    return aggregates


def aggregates_to_dataframe(aggregates: Iterable[SpeakerAggregate]) -> pd.DataFrame:
    return pd.DataFrame([agg.dict() for agg in aggregates])
