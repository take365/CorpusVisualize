from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .aggregate import aggregate_speakers, aggregates_to_dataframe
from .asr import transcribe
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
)
from .types import PipelineSettings, SegmentSchema

app = typer.Typer(help="CorpusVisualize audio-processing pipeline")
console = Console()


def _load_settings(config_path: Optional[Path]) -> PipelineSettings:
    if config_path is None:
        return PipelineSettings()
    with Path(config_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
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
    min_seg_sec: Optional[float] = typer.Option(None, help="Minimum segment length in seconds"),
    max_seg_sec: Optional[float] = typer.Option(None, help="Maximum segment length in seconds"),
    sample_rate: Optional[int] = typer.Option(None, help="Target sample rate"),
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
        min_seg_sec=min_seg_sec,
        max_seg_sec=max_seg_sec,
        sample_rate=sample_rate,
    )

    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    diarizer = get_diarizer(settings.diarization, settings.min_seg_sec, settings.max_seg_sec)
    emotion_extractor = EmotionExtractor(settings.emotion)
    pitch_extractor = PitchExtractor(settings.pitch)
    loudness_extractor = LoudnessExtractor(settings.loudness)
    tempo_extractor = TempoExtractor(settings.tempo)
    dialect_scorer = DialectScorer(settings.dialect)
    lexicon_highlighter = LexiconHighlighter(settings.lexicon)

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

            diarized: List[DiarizationSegment] = diarizer(audio, sr)
            transcripts = transcribe(
                diarized,
                conversation_id,
                method=settings.asr,
                audio=audio,
                sample_rate=sr,
                language=settings.language,
            )

            for idx, (segment, transcript) in enumerate(zip(diarized, transcripts)):
                progress.update(task, description=f"Analyzing {audio_path.name} segment {idx+1}")
                seg_schema = SegmentSchema(
                    id=f"{conversation_id}_s{idx:03d}",
                    conversation_id=conversation_id,
                    source_file=str(audio_path.relative_to(input_dir)),
                    start=segment.start,
                    end=segment.end,
                    speaker=segment.speaker,
                    text=transcript,
                    emotion=emotion_extractor(audio, sr, segment),
                    pitch=pitch_extractor(audio, sr, segment),
                    loudness=loudness_extractor(audio, sr, segment),
                    tempo=tempo_extractor(audio, sr, segment, transcript),
                    dialect=dialect_scorer(audio, sr, segment, transcript),
                    highlights=lexicon_highlighter(transcript, segment),
                    created_at=datetime.utcnow(),
                    analyzer={
                        "asr": settings.asr,
                        "diarization": settings.diarization,
                        "emotion": settings.emotion,
                        "pitch": settings.pitch,
                        "loudness": settings.loudness,
                        "tempo": settings.tempo,
                        "dialect": settings.dialect,
                        "lexicon": settings.lexicon,
                        "language": settings.language,
                    },
                )
                all_segments.append(seg_schema)
            progress.advance(task)

    if not all_segments:
        console.print("[yellow]No segments produced. Check input data or parameters.[/yellow]")
        raise typer.Exit(code=1)

    segments_path = output_dir / "segments.jsonl"
    speakers_path = output_dir / "speakers.parquet"

    write_segments_jsonl(all_segments, segments_path)

    aggregates = aggregate_speakers(all_segments)
    df = aggregates_to_dataframe(aggregates)
    write_speakers_parquet(df, speakers_path)

    console.print(
        "[green]Processing complete[/green]",
        f"segments={len(all_segments)}",
        f"files={len(audio_files)}",
        f"output={segments_path}",
    )


if __name__ == "__main__":
    app()
