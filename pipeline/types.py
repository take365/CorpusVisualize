from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class HighlightSchema(BaseModel):
    startChar: int
    endChar: int
    tag: str

    @validator("endChar")
    def _validate_span(cls, value: int, values) -> int:
        start = values.get("startChar", 0)
        if value < start:
            raise ValueError("endChar must be greater than or equal to startChar")
        return value


class SegmentFeatures(BaseModel):
    emotion: Dict[str, float]
    pitch: List[float]
    loudness: float
    tempo: float
    dialect: Dict[str, float]
    highlights: List[HighlightSchema] = Field(default_factory=list)

    @validator("pitch", pre=True, each_item=True)
    def _ensure_finite_pitch(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("pitch values must be finite")
        return float(value)

    @validator("emotion")
    def _validate_emotion(cls, value: Dict[str, float]) -> Dict[str, float]:
        if not value:
            raise ValueError("emotion map cannot be empty")
        return {k: float(v) for k, v in value.items()}

    @validator("dialect")
    def _validate_dialect(cls, value: Dict[str, float]) -> Dict[str, float]:
        if not value:
            raise ValueError("dialect map cannot be empty")
        return {k: float(v) for k, v in value.items()}


class SegmentSchema(BaseModel):
    id: str
    conversation_id: str
    source_file: str
    start: float
    end: float
    speaker: str
    text: str
    emotion: Dict[str, float]
    pitch: List[float]
    loudness: float
    tempo: float
    dialect: Dict[str, float]
    highlights: List[HighlightSchema] = Field(default_factory=list)
    created_at: datetime
    analyzer: Dict[str, str]

    @validator("end")
    def _validate_duration(cls, value: float, values) -> float:
        start = values.get("start", 0.0)
        if value <= start:
            raise ValueError("end must be greater than start")
        return float(value)

    @validator("speaker")
    def _validate_speaker(cls, value: str) -> str:
        if not value:
            raise ValueError("speaker id is required")
        return value

    @validator("created_at", pre=True, always=True)
    def _default_timestamp(cls, value: Optional[datetime]) -> datetime:
        return value or datetime.utcnow()


class PipelineSettings(BaseModel):
    diarization: str = "energy_split"
    asr: str = "dummy"
    emotion: str = "energy_based"
    pitch: str = "yin"
    loudness: str = "rms"
    tempo: str = "chars_per_sec"
    dialect: str = "lexicon_tfidf"
    lexicon: str = "pos_dict"
    language: str = "ja"
    min_seg_sec: float = 2.0
    max_seg_sec: float = 30.0
    sample_rate: int = 16000
    output_format: str = "jsonl"


class SpeakerAggregate(BaseModel):
    speaker: str
    segment_count: int
    total_duration: float
    avg_loudness: float
    avg_tempo: float
    dominant_emotion: str

