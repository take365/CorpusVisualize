from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np


def load_audio(path: Path, sample_rate: int) -> Tuple[np.ndarray, int]:
    """Load mono audio with the desired sample rate."""
    data, sr = librosa.load(path, sr=sample_rate, mono=True)
    return data, sr
