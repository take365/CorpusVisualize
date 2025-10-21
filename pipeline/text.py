from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from fugashi import Tagger
import unidic_lite


@lru_cache(maxsize=1)
def _get_tagger() -> Tagger:
    return Tagger(f"-d {unidic_lite.DICDIR}")


def katakana_to_hiragana(text: str) -> str:
    chars = []
    for ch in text:
        code = ord(ch)
        if 0x30A0 <= code <= 0x30FF:
            chars.append(chr(code - 0x60))
        else:
            chars.append(ch)
    return "".join(chars)


ACCENT_MAP = {
    "0": "Heiban",
    "1": "Atamadaka",
    "2": "Nakadaka",
    "3": "Odaka",
}


@dataclass
class ProsodyInfo:
    kana: Optional[str]
    accent: Optional[str]


def analyze_prosody(text: str) -> ProsodyInfo:
    tagger = _get_tagger()
    nodes = list(tagger(text))
    if not nodes:
        return ProsodyInfo(kana=None, accent=None)

    kana_parts: List[str] = []
    accent_types: List[str] = []
    accent_connections: List[str] = []

    for node in nodes:
        feature = node.feature
        kana = getattr(feature, "kanaBase", None) or getattr(feature, "kana", None)
        if kana and kana != "*":
            kana_parts.append(kana)
        a_type = getattr(feature, "aType", None)
        if a_type and a_type != "*":
            accent_types.append(str(a_type))
        a_con = getattr(feature, "aConType", None)
        if a_con and a_con != "*":
            accent_connections.append(str(a_con))

    kana_text = katakana_to_hiragana("".join(kana_parts)) if kana_parts else None

    accent_label: Optional[str]
    if accent_types:
        primary = accent_types[0]
        base = ACCENT_MAP.get(primary, f"aType={primary}")
        if accent_connections:
            accent_label = f"{base} (aCon={accent_connections[0]})"
        else:
            accent_label = base
    else:
        accent_label = None

    return ProsodyInfo(kana=kana_text, accent=accent_label)
