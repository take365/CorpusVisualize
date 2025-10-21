from __future__ import annotations

from typing import Dict, List

from ..diarization import DiarizationSegment

_HIGHLIGHT_TAGS: Dict[str, List[str]] = {
    "dialect": ["やね", "ばい", "だべ", "でしょう"],
    "polite": ["ありがとうございます", "お願いします", "恐縮です"],
}


class LexiconHighlighter:
    """Highlight specific lexical cues in transcripts."""

    def __init__(self, method: str = "pos_dict") -> None:
        self.method = method

    def __call__(self, transcript: str, segment: DiarizationSegment) -> List[Dict[str, int]]:
        highlights: List[Dict[str, int]] = []
        for tag, lexemes in _HIGHLIGHT_TAGS.items():
            for word in lexemes:
                start = transcript.find(word)
                if start != -1:
                    end = start + len(word)
                    highlights.append(
                        {
                            "startChar": start,
                            "endChar": end,
                            "tag": tag,
                        }
                    )
        return highlights
