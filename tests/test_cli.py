import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from pipeline.run_pipeline import app


def create_audio(path):
    sr = 16000
    t = np.linspace(0, 4.0, int(sr * 4.0), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 220 * t)
    sf.write(path, audio, sr)


def test_cli_pipeline(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    audio_path = input_dir / "sample.wav"
    create_audio(audio_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.stdout
    assert (output_dir / "segments.jsonl").exists()
    content = (output_dir / "segments.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert content, "segments.jsonl should contain data"
