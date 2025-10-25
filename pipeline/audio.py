from pathlib import Path
import numpy as np
import torchaudio
from rich.console import Console
console = Console()
def _cache_path(wav_path: Path, sample_rate: int) -> Path:
    out = Path("output/cache_audio")
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{wav_path.stem}_sr{sample_rate}_mono.npy"

def load_audio(path: str | Path, sample_rate: int = 16000):
    path = Path(path)
    cpath = _cache_path(path, sample_rate)
    if cpath.exists():
        console.print("audio cache load")
        audio = np.load(cpath, mmap_mode="r")  # ほぼ即時
        return audio, sample_rate

    console.print("audio load",path)
    waveform, sr = torchaudio.load(str(path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        sr = sample_rate
    audio = waveform.squeeze(0).numpy().astype(np.float32)

    np.save(cpath, audio)  # 次回からはmmapで即読
    audio = np.load(cpath, mmap_mode="r")
    return audio, sr
