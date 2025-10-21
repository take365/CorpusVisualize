from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .types import SegmentSchema


def write_segments_jsonl(segments: Iterable[SegmentSchema], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg.dict(), default=str, ensure_ascii=False))
            f.write("\n")


def write_speakers_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
