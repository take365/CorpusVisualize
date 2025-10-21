from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import os
import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .aggregate import aggregate_speakers, aggregates_to_dataframe
from .asr import TranscribedSegment, transcribe
from .audio import load_audio
from .diarization import DiarizationSegment, get_diarizer
from .export import write_segments_jsonl, write_speakers_parquet
from .features import (
    DialectScorer,
    EmotionExtractor,
    LexiconHighlighter,
    LoudnessExtractor,
    PitchExtractor,
    TempoExtractor,
    WordFeatureExtractor,
)
from .emotion import EmotionAnalyzer
from .asr import TranscribedSegment
from .types import PipelineSettings, SegmentSchema

load_dotenv()
app = typer.Typer(help="CorpusVisualize audio-processing pipeline")
console = Console()


ENV_MAPPING: Dict[str, str] = {
    "CV_ASR": "asr",
    "CV_DIARIZATION": "diarization",
    "CV_EMOTION": "emotion",
    "CV_PITCH": "pitch",
    "CV_LOUDNESS": "loudness",
    "CV_TEMPO": "tempo",
    "CV_DIALECT": "dialect",
    "CV_LEXICON": "lexicon",
    "CV_LANGUAGE": "language",
    "CV_SER_BACKEND": "ser_backend",
    "CV_PROSODY_BACKEND": "prosody_backend",
    "CV_WORD_PITCH_BACKEND": "word_pitch_backend",
    "CV_PYANNOTE_THRESHOLD": "pyannote_threshold",
    "CV_PYANNOTE_MIN_CLUSTER_SIZE": "pyannote_min_cluster_size",
    "CV_PYANNOTE_MIN_DURATION_OFF": "pyannote_min_duration_off",
    "CV_PYANNOTE_NUM_SPEAKERS": "pyannote_num_speakers",
}


def _settings_from_env() -> Dict[str, str]:
    values: Dict[str, str] = {}
    for env_key, field in ENV_MAPPING.items():
        value = os.getenv(env_key)
        if value is not None:
            values[field] = value
    float_mappings = [
        ("CV_MIN_SEG_SEC", "min_seg_sec"),
        ("CV_MAX_SEG_SEC", "max_seg_sec"),
        ("CV_PYANNOTE_THRESHOLD", "pyannote_threshold"),
        ("CV_PYANNOTE_MIN_DURATION_OFF", "pyannote_min_duration_off"),
    ]
    for float_key, field in float_mappings:
        value = os.getenv(float_key)
        if value is not None:
            try:
                values[field] = float(value)
            except ValueError:
                pass
    int_mappings = [
        ("CV_SAMPLE_RATE", "sample_rate"),
        ("CV_PYANNOTE_MIN_CLUSTER_SIZE", "pyannote_min_cluster_size"),
        ("CV_PYANNOTE_NUM_SPEAKERS", "pyannote_num_speakers"),
    ]
    for int_key, field in int_mappings:
        value = os.getenv(int_key)
        if value is not None:
            try:
                values[field] = int(value)
            except ValueError:
                pass
    return values


def _load_settings(config_path: Optional[Path]) -> PipelineSettings:
    data: Dict[str, object] = _settings_from_env()
    if config_path is not None:
        with Path(config_path).open("r", encoding="utf-8") as f:
            file_data = yaml.safe_load(f) or {}
            if not isinstance(file_data, dict):
                raise ValueError("Config file must contain a mapping")
            data.update(file_data)
    return PipelineSettings(**data)


def _apply_cli_overrides(settings: PipelineSettings, **overrides) -> PipelineSettings:
    data = settings.dict()
    data.update({k: v for k, v in overrides.items() if v is not None})
    return PipelineSettings(**data)


def _list_audio_files(input_dir: Path) -> List[Path]:
    files = [p for p in input_dir.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".mp3"}]
    return sorted(files)


@app.command()
def main(
    input_dir: Path = typer.Option(..., help='Directory containing audio files'),
    output_dir: Path = typer.Option(..., help='Directory for processed outputs'),
    emotion: Optional[str] = typer.Option(None, help="Emotion extractor method"),
    pitch: Optional[str] = typer.Option(None, help="Pitch extraction method"),
    loudness: Optional[str] = typer.Option(None, help="Loudness method"),
    tempo: Optional[str] = typer.Option(None, help="Tempo method"),
    dialect: Optional[str] = typer.Option(None, help="Dialect scoring method"),
    lexicon: Optional[str] = typer.Option(None, help="Lexicon highlight method"),
    asr: Optional[str] = typer.Option(None, help="ASR method"),
    diar: Optional[str] = typer.Option(None, help="Diarization method"),
    language: Optional[str] = typer.Option(None, help="ASR language hint (e.g. ja)"),
    ser_backend: Optional[str] = typer.Option(None, help="Speech emotion backend (dummy|speechbrain)"),
    prosody_backend: Optional[str] = typer.Option(None, help="Prosody backend for kana/accent"),
    word_pitch_backend: Optional[str] = typer.Option(None, help="Pitch backend for word analysis"),
    min_seg_sec: Optional[float] = typer.Option(None, help="Minimum segment length in seconds"),
    max_seg_sec: Optional[float] = typer.Option(None, help="Maximum segment length in seconds"),
    sample_rate: Optional[int] = typer.Option(None, help="Target sample rate"),
    pyannote_threshold: Optional[float] = typer.Option(None, help="Override pyannote clustering threshold"),
    pyannote_min_cluster_size: Optional[int] = typer.Option(None, help="Minimum cluster size for pyannote clustering"),
    pyannote_min_duration_off: Optional[float] = typer.Option(None, help="Minimum silence duration for pyannote segmentation"),
    pyannote_num_speakers: Optional[int] = typer.Option(None, help="Estimated number of speakers for pyannote (optional)"),
    config: Optional[Path] = typer.Option(None, help="Optional YAML config"),
    limit: Optional[int] = typer.Option(None, help="Optional limit for number of files to process"),
) -> None:
    base_settings = _load_settings(config)
    settings = _apply_cli_overrides(
        base_settings,
        diarization=diar,
        asr=asr,
        emotion=emotion,
        pitch=pitch,
        loudness=loudness,
        tempo=tempo,
        dialect=dialect,
        lexicon=lexicon,
        language=language,
        ser_backend=ser_backend,
        prosody_backend=prosody_backend,
        word_pitch_backend=word_pitch_backend,
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
        sample_rate=sample_rate,
        pyannote_threshold=pyannote_threshold,
        pyannote_min_cluster_size=pyannote_min_cluster_size,
        pyannote_min_duration_off=pyannote_min_duration_off,
        pyannote_num_speakers=pyannote_num_speakers,
    )

    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pyannote_overrides: Dict[str, object] = {}
    if settings.pyannote_threshold is not None:
        pyannote_overrides.setdefault("clustering", {})["threshold"] = settings.pyannote_threshold
    if settings.pyannote_min_cluster_size is not None:
        pyannote_overrides.setdefault("clustering", {})["min_cluster_size"] = settings.pyannote_min_cluster_size
    if settings.pyannote_min_duration_off is not None:
        pyannote_overrides.setdefault("segmentation", {})["min_duration_off"] = settings.pyannote_min_duration_off
    if settings.pyannote_num_speakers is not None:
        pyannote_overrides["num_speakers"] = settings.pyannote_num_speakers
    diarizer = get_diarizer(
        settings.diarization,
        settings.min_seg_sec,
        settings.max_seg_sec,
        overrides=pyannote_overrides or None,
    )
    emotion_extractor = EmotionExtractor(settings.emotion)
    pitch_extractor = PitchExtractor(settings.pitch)
    loudness_extractor = LoudnessExtractor(settings.loudness)
    tempo_extractor = TempoExtractor(settings.tempo)
    dialect_scorer = DialectScorer(settings.dialect)
    lexicon_highlighter = LexiconHighlighter(settings.lexicon)
    emotion_analyzer = EmotionAnalyzer(settings.ser_backend, device="cuda" if settings.ser_backend == "speechbrain" else "cpu")
    word_extractor = WordFeatureExtractor(settings.prosody_backend, settings.word_pitch_backend)

    audio_files = _list_audio_files(input_dir)
    if limit is not None:
        audio_files = audio_files[:limit]

    if not audio_files:
        console.print("[yellow]No audio files found.[/yellow]")
        raise typer.Exit(code=1)

    all_segments: List[SegmentSchema] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Processing audio", total=len(audio_files))
        for audio_path in audio_files:
            progress.update(task, description=f"Loading {audio_path.name}")
            audio, sr = load_audio(audio_path, settings.sample_rate)
            conversation_id = audio_path.stem
            conversation_dir = output_dir / conversation_id
            conversation_dir.mkdir(parents=True, exist_ok=True)
            raw_text_rel = str(Path(conversation_id) / f"{conversation_id}.raw.txt")
            raw_text_path = conversation_dir / f"{conversation_id}.raw.txt"

            diarized: List[DiarizationSegment] = diarizer(audio, sr)
            transcripts = transcribe(
                diarized,
                conversation_id,
                method=settings.asr,
                audio=audio,
                sample_rate=sr,
                language=settings.language,
            )

            raw_lines: List[str] = []
            file_segments: List[SegmentSchema] = []

            for idx, (segment, transcription) in enumerate(zip(diarized, transcripts)):
                progress.update(task, description=f"Analyzing {audio_path.name} segment {idx+1}")

                seg_start = max(int(segment.start * sr), 0)
                seg_end = min(int(segment.end * sr), audio.shape[-1])
                segment_audio = audio[seg_start:seg_end] if seg_end > seg_start else audio

                emotion_result = emotion_analyzer.analyse(segment_audio, sr)
                if settings.ser_backend == "dummy":
                    emotion_map = emotion_extractor(audio, sr, segment)
                else:
                    emotion_map = emotion_result.scores or emotion_extractor(audio, sr, segment)

                if "happiness" in emotion_map and "joy" not in emotion_map:
                    emotion_map["joy"] = emotion_map.pop("happiness")
                if "sadness" in emotion_map and "sad" not in emotion_map:
                    emotion_map["sad"] = emotion_map.pop("sadness")
                emotion_map.setdefault("neutral", 0.0)

                total = sum(emotion_map.values()) or 1.0
                emotion_map = {k: float(v) / total for k, v in emotion_map.items()}

                words = word_extractor(
                    audio,
                    sr,
                    transcription.words,
                    emotion_result.valence,
                    emotion_result.arousal,
                )

                text_value = transcription.text
                raw_lines.append(transcription.raw_text)

                seg_schema = SegmentSchema(
                    id=f"{conversation_id}_s{idx:03d}",
                    conversation_id=conversation_id,
                    source_file=str(audio_path.relative_to(input_dir)),
                    start=segment.start,
                    end=segment.end,
                    speaker=segment.speaker,
                    text=text_value,
                    emotion=emotion_map,
                    pitch=pitch_extractor(audio, sr, segment),
                    loudness=loudness_extractor(audio, sr, segment),
                    tempo=tempo_extractor(audio, sr, segment, text_value),
                    dialect=dialect_scorer(audio, sr, segment, text_value),
                    highlights=lexicon_highlighter(text_value, segment),
                    words=words,
                    created_at=datetime.utcnow(),
                    analyzer={
                        "asr": settings.asr,
                        "diarization": settings.diarization,
                        "emotion": settings.emotion,
                        "ser_backend": settings.ser_backend,
                        "raw_transcript_file": raw_text_rel,
                        "pitch": settings.pitch,
                        "loudness": settings.loudness,
                        "tempo": settings.tempo,
                        "dialect": settings.dialect,
                        "lexicon": settings.lexicon,
                        "language": settings.language,
                        "prosody_backend": settings.prosody_backend,
                        "word_pitch_backend": settings.word_pitch_backend,
                        "valence": f"{emotion_result.valence:.3f}",
                        "arousal": f"{emotion_result.arousal:.3f}",
                    },
                )
                all_segments.append(seg_schema)
                file_segments.append(seg_schema)
            progress.advance(task)

            raw_payload = "\n".join(line for line in raw_lines if line)
            raw_text_path.write_text(raw_payload, encoding="utf-8")

            segments_path = conversation_dir / "segments.jsonl"
            speakers_path = conversation_dir / "speakers.parquet"

            write_segments_jsonl(file_segments, segments_path)

            aggregates = aggregate_speakers(file_segments)
            df = aggregates_to_dataframe(aggregates)
            write_speakers_parquet(df, speakers_path)

    if not all_segments:
        console.print("[yellow]No segments produced. Check input data or parameters.[/yellow]")
        raise typer.Exit(code=1)

    console.print(
        "[green]Processing complete[/green]",
        f"segments={len(all_segments)}",
        f"files={len(audio_files)}",
        f"output_dir={output_dir}",
    )


if __name__ == "__main__":
    app()
