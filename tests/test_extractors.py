import numpy as np

from pipeline.diarization import DiarizationSegment
from pipeline.features import (
    DialectScorer,
    EmotionExtractor,
    LexiconHighlighter,
    LoudnessExtractor,
    PitchExtractor,
    TempoExtractor,
)


def generate_sine(duration: float = 2.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return 0.2 * np.sin(2 * np.pi * 220 * t)


def test_feature_extractors():
    audio = generate_sine()
    sr = 16000
    segment = DiarizationSegment(start=0.0, end=2.0, speaker="A")
    transcript = "なるほど、それで進めましょう。"

    emotion = EmotionExtractor()(audio, sr, segment)
    pitch = PitchExtractor()(audio, sr, segment)
    loudness = LoudnessExtractor()(audio, sr, segment)
    tempo = TempoExtractor()(audio, sr, segment, transcript)
    dialect = DialectScorer()(audio, sr, segment, transcript)
    highlights = LexiconHighlighter()(transcript, segment)

    assert abs(sum(emotion.values()) - 1.0) < 1e-6
    assert isinstance(pitch, list)
    assert loudness > 0
    assert tempo > 0
    assert set(dialect.keys()) == {"kansai", "kanto", "tohoku", "kyushu", "hokkaido"}
    assert isinstance(highlights, list)
