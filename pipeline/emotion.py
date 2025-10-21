from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


_SER_LABELS = ["anger", "happiness", "sadness", "neutral"]
_LABEL_TO_VA = {
    "anger": (-0.4, 0.7),
    "happiness": (0.7, 0.5),
    "sadness": (-0.7, -0.5),
    "neutral": (0.0, 0.0),
}


@dataclass
class EmotionResult:
    label: str
    scores: Dict[str, float]
    valence: float
    arousal: float


class EmotionAnalyzer:
    def __init__(self, backend: str = "dummy", device: Optional[str] = None) -> None:
        self.backend = (backend or "dummy").lower()
        self.device = device or "cpu"
        self._classifier = None

        if self.backend == "speechbrain":  # pragma: no cover - optional heavy dep
            try:
                from speechbrain.pretrained import EncoderClassifier

                self._classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                    run_opts={"device": self.device},
                )
            except Exception as exc:
                logger.warning("Failed to initialise SpeechBrain emotion model: %s", exc)
                self.backend = "dummy"
                self._classifier = None

    def analyse(self, audio: np.ndarray, sample_rate: int) -> EmotionResult:
        if self.backend == "speechbrain" and self._classifier is not None:
            try:
                return self._analyse_speechbrain(audio, sample_rate)
            except Exception as exc:  # pragma: no cover - inference failure
                logger.warning("SpeechBrain inference failed: %s", exc)
        return self._analyse_dummy(audio)

    @staticmethod
    def _analyse_dummy(audio: np.ndarray) -> EmotionResult:
        energy = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        valence = float(np.clip(0.5 - energy, -1.0, 1.0))
        arousal = float(np.clip(energy * 2.0, -1.0, 1.0))
        scores = {label: (1.0 / len(_SER_LABELS)) for label in _SER_LABELS}
        return EmotionResult(label="neutral", scores=scores, valence=valence, arousal=arousal)

    def _analyse_speechbrain(self, audio: np.ndarray, sample_rate: int) -> EmotionResult:
        assert self._classifier is not None
        import torch

        tensor = torch.from_numpy(audio).float().unsqueeze(0)
        if tensor.abs().max() > 1:
            tensor = tensor / tensor.abs().max()
        tensor = tensor.to(self.device)
        self._classifier.eval()
        with torch.no_grad():
            logits = self._classifier.classify_batch(tensor, sample_rate)[0]
        probs = logits.detach().cpu().softmax(dim=-1).numpy()
        scores = {label: float(prob) for label, prob in zip(_SER_LABELS, probs)}
        label = max(scores, key=scores.get)
        valence, arousal = _LABEL_TO_VA.get(label, (0.0, 0.0))
        return EmotionResult(label=label, scores=scores, valence=valence, arousal=arousal)
