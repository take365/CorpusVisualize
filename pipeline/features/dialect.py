from __future__ import annotations

import math
from typing import Any, Dict

from ..diarization import DiarizationSegment

_DIALECT_KEYWORDS = {
    "kansai": ["やね", "せやな", "ほんま", "なんでやねん"],
    "kanto": ["ですね", "かな", "でしょう"],
    "tohoku": ["だべ", "だっぺ"],
    "kyushu": ["やけん", "ばい"],
    "hokkaido": ["だべさ", "したっけ"],
}


class DialectScorer:
    """Score dialect tendencies based on keyword hits."""

    def __init__(self, method: str = "lexicon_tfidf") -> None:
        self.method = method

    def __call__(
        self,
        audio: Any,
        sample_rate: int,
        segment: DiarizationSegment,
        transcript: str,
    ) -> Dict[str, float]:
        text = transcript.lower()
        scores: Dict[str, float] = {}
        total = 0.0
        for key, words in _DIALECT_KEYWORDS.items():
            count = sum(text.count(word) for word in words)
            score = 0.1 + math.log1p(count)
            scores[key] = score
            total += score
        if total == 0:
            uniform = 1.0 / len(_DIALECT_KEYWORDS)
            return {k: uniform for k in _DIALECT_KEYWORDS}
        return {k: v / total for k, v in scores.items()}
