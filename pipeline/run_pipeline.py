from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import math
import os
import subprocess
import sys
import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from openai import OpenAI

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
from .llm import SegmentDraft, WordBoundary, refine_segments_with_llm
from .report_html import generate_comparison_html
from .quick_io import QuickArtifactsError, load_quick_artifacts
from .types import PipelineSettings, SegmentSchema

load_dotenv()
app = typer.Typer(help="CorpusVisualize audio-processing pipeline")
console = Console()

DEFAULT_INPUT_DIR = Path(os.getenv("CV_INPUT_DIR", "data"))
DEFAULT_OUTPUT_DIR = Path(os.getenv("CV_OUTPUT_DIR", "output"))


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
    "CV_LLM_ENABLED": "llm_enabled",
    "CV_LLM_BASE_URL": "llm_base_url",
    "CV_LLM_MODEL": "llm_model",
    "CV_LLM_API_KEY": "llm_api_key",
    "CV_LLM_MAX_TOKENS": "llm_max_tokens",
    "CV_LLM_TEMPERATURE": "llm_temperature",
    "CV_REF_DIR": "ref_dir",
    "CV_COMPARISON_THEME": "comparison_theme",
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
        ("CV_LLM_TEMPERATURE", "llm_temperature"),
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
        ("CV_LLM_MAX_TOKENS", "llm_max_tokens"),
    ]
    for int_key, field in int_mappings:
        value = os.getenv(int_key)
        if value is not None:
            try:
                values[field] = int(value)
            except ValueError:
                pass
    bool_mappings = [
        ("CV_LLM_ENABLED", "llm_enabled"),
        ("CV_COMPARISON_HTML", "comparison_html"),
    ]
    for bool_key, field in bool_mappings:
        value = os.getenv(bool_key)
        if value is not None:
            values[field] = value.strip().lower() in {"1", "true", "yes", "on"}
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


def _gather_audio_inputs(inputs: Iterable[Path], default_dir: Path) -> List[Path]:
    collected: List[Path] = []
    if inputs:
        for item in inputs:
            resolved = item.expanduser()
            if resolved.is_dir():
                collected.extend(_list_audio_files(resolved.resolve()))
            elif resolved.is_file():
                collected.append(resolved.resolve())
            else:
                raise typer.BadParameter(f"Input not found: {item}")
    else:
        collected = _list_audio_files(default_dir)
    deduped: List[Path] = []
    seen = set()
    for path in collected:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


REFERENCE_EXTS = (".jsonl", ".json", ".txt")


def _find_reference(conversation_id: str, ref_root: Path) -> Optional[Path]:
    for ext in REFERENCE_EXTS:
        candidate = ref_root / f"{conversation_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _needs_quick_artifacts(settings: PipelineSettings) -> bool:
    diar = (settings.diarization or "").lower()
    asr_method = (settings.asr or "").lower()
    quick_aliases = {"quick", "quick_cluster"}
    return diar in quick_aliases or asr_method in quick_aliases


def _resolve_quick_data_root(output_root: Path) -> Path:
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HMD_DATA_ROOT"] = str(root)
    return root


def _ensure_quick_artifacts(
    audio_path: Path,
    settings: PipelineSettings,
    console: Console,
    data_root: Path,
) -> None:
    if not _needs_quick_artifacts(settings):
        return

    conversation_id = audio_path.stem
    try:
        load_quick_artifacts(conversation_id, data_root=data_root)
        return
    except QuickArtifactsError:
        pass

    console.print(
        f"[cyan]Running Quick clustering helper for {audio_path.name}[/cyan]"
    )
    cmd = [
        sys.executable,
        "-m",
        "pipeline.hmd_quick_cluster",
        str(audio_path),
    ]
    cmd.extend(["--data-root", str(data_root)])
    project_env = Path(".env")
    if project_env.exists():
        cmd.extend(["--env", str(project_env.resolve())])
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Quick clustering helper failed for {audio_path.name} (exit {exc.returncode})"
        ) from exc

    try:
        load_quick_artifacts(conversation_id, data_root=data_root)
    except QuickArtifactsError as exc:
        raise RuntimeError(
            f"Quick artifacts are still unavailable for {conversation_id}: {exc}"
        ) from exc


@app.command()
def main(
    inputs: List[Path] = typer.Argument(
        None,
        help="Audio files or directories to process. Overrides --input-dir when provided.",
    ),
    input_dir: Path = typer.Option(
        DEFAULT_INPUT_DIR,
        help='Directory containing audio files',
        show_default=True,
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        help='Directory for processed outputs',
        show_default=True,
    ),
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
    llm: Optional[bool] = typer.Option(None, "--llm/--no-llm", help="Enable LLM-based sentence refinement"),
    llm_base_url: Optional[str] = typer.Option(None, help="Base URL for LLM API (e.g. http://host:port/v1)"),
    llm_model: Optional[str] = typer.Option(None, help="LLM model identifier"),
    llm_api_key: Optional[str] = typer.Option(None, help="API key for LLM endpoint"),
    llm_max_tokens: Optional[int] = typer.Option(None, help="Maximum tokens for LLM response"),
    llm_temperature: Optional[float] = typer.Option(None, help="Sampling temperature for LLM response"),
    ref_dir: Optional[Path] = typer.Option(None, help="Directory containing reference transcripts for comparison"),
    comparison_html: Optional[bool] = typer.Option(
        None,
        "--comparison-html/--no-comparison-html",
        help="Generate HTML comparison report when reference is available",
    ),
    comparison_theme: Optional[str] = typer.Option(None, help="Comparison HTML theme (light|dark)"),
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
        llm_enabled=llm,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_max_tokens=llm_max_tokens,
        llm_temperature=llm_temperature,
        ref_dir=str(ref_dir) if ref_dir is not None else None,
        comparison_html=comparison_html,
        comparison_theme=comparison_theme,
    )

    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_dir_path: Optional[Path] = None
    if settings.ref_dir:
        candidate = Path(settings.ref_dir).expanduser()
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate
        if resolved.exists():
            ref_dir_path = resolved
        else:
            console.print(
                f"[yellow]Reference directory not found; comparison HTML disabled ({resolved}).[/yellow]"
            )
    comparison_theme = (settings.comparison_theme or "light").lower()
    if comparison_theme not in {"light", "dark"}:
        console.print(
            f"[yellow]Unknown comparison theme '{comparison_theme}'; falling back to 'light'.[/yellow]"
        )
        comparison_theme = "light"

    quick_data_root = _resolve_quick_data_root(output_dir)

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
        quick_data_root=str(quick_data_root),
    )
    emotion_extractor = EmotionExtractor(settings.emotion)
    pitch_extractor = PitchExtractor(settings.pitch)
    loudness_extractor = LoudnessExtractor(settings.loudness)
    tempo_extractor = TempoExtractor(settings.tempo)
    dialect_scorer = DialectScorer(settings.dialect)
    lexicon_highlighter = LexiconHighlighter(settings.lexicon)
    emotion_analyzer = EmotionAnalyzer(settings.ser_backend, device="cuda" if settings.ser_backend == "speechbrain" else "cpu")
    word_extractor = WordFeatureExtractor(settings.prosody_backend, settings.word_pitch_backend)

    llm_client: Optional[OpenAI] = None
    if settings.llm_enabled:
        base_url = (settings.llm_base_url or "").strip()
        if base_url:
            base = base_url.rstrip("/")
            if not base.endswith("/v1"):
                base = f"{base}/v1"
            try:
                llm_client = OpenAI(api_key=settings.llm_api_key or "", base_url=base)
            except Exception as exc:
                console.print(
                    f"[yellow]Failed to initialise LLM client ({exc}); continuing without LLM refinement.[/yellow]"
                )
                llm_client = None
        else:
            console.print("[yellow]LLM base URL is empty; disabling LLM refinement.[/yellow]")

    audio_files = _gather_audio_inputs(inputs, input_dir)
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
            _ensure_quick_artifacts(audio_path, settings, console, quick_data_root)
            audio, sr = load_audio(audio_path, settings.sample_rate)
            conversation_id = audio_path.stem
            conversation_dir = output_dir / conversation_id
            conversation_dir.mkdir(parents=True, exist_ok=True)
            raw_text_rel = str(Path(conversation_id) / f"{conversation_id}.raw.txt")
            raw_text_path = conversation_dir / f"{conversation_id}.raw.txt"

            set_context = getattr(diarizer, "set_context", None)
            if callable(set_context):
                set_context(audio_path, conversation_id)

            diarized: List[DiarizationSegment] = diarizer(audio, sr)
            transcripts = transcribe(
                diarized,
                conversation_id,
                method=settings.asr,
                audio=audio,
                sample_rate=sr,
                language=settings.language,
                quick_data_root=quick_data_root,
            )

            raw_drafts: List[SegmentDraft] = []
            for segment, transcription in zip(diarized, transcripts):
                word_boundaries: List[WordBoundary] = []
                for word in transcription.words:
                    start = float(getattr(word, "start", segment.start))
                    end = float(getattr(word, "end", start))
                    if not math.isfinite(start) or not math.isfinite(end):
                        continue
                    if end <= start:
                        end = start + 1e-3
                    word_boundaries.append(
                        WordBoundary(
                            text=str(getattr(word, "text", "")),
                            start=start,
                            end=end,
                        )
                    )
                raw_drafts.append(
                    SegmentDraft(
                        speaker=segment.speaker,
                        start=segment.start,
                        end=segment.end,
                        text=str(transcription.text).strip(),
                        words=word_boundaries,
                    )
                )

            refined_drafts = refine_segments_with_llm(
                raw_drafts,
                llm_client,
                settings.llm_model,
                settings.llm_max_tokens,
                settings.llm_temperature,
            )

            raw_lines: List[str] = []
            file_segments: List[SegmentSchema] = []

            for idx, seg in enumerate(refined_drafts):
                progress.update(task, description=f"Analyzing {audio_path.name} segment {idx+1}")
                synthetic = DiarizationSegment(seg.start, seg.end, seg.speaker)
                seg_start = max(int(seg.start * sr), 0)
                seg_end = min(int(seg.end * sr), audio.shape[-1])
                segment_audio = audio[seg_start:seg_end] if seg_end > seg_start else audio

                emotion_result = emotion_analyzer.analyse(segment_audio, sr)
                if settings.ser_backend == "dummy":
                    emotion_map = emotion_extractor(audio, sr, synthetic)
                else:
                    emotion_map = emotion_result.scores or emotion_extractor(audio, sr, synthetic)

                if "happiness" in emotion_map and "joy" not in emotion_map:
                    emotion_map["joy"] = emotion_map.pop("happiness")
                if "sadness" in emotion_map and "sad" not in emotion_map:
                    emotion_map["sad"] = emotion_map.pop("sadness")
                emotion_map.setdefault("neutral", 0.0)

                total = sum(emotion_map.values()) or 1.0
                emotion_map = {k: float(v) / total for k, v in emotion_map.items()}

                words_schema = word_extractor(
                    audio,
                    sr,
                    seg.words,
                    emotion_result.valence,
                    emotion_result.arousal,
                )

                if seg.text:
                    text_value = seg.text.strip()
                elif seg.words:
                    text_value = "".join(word.text for word in seg.words)
                else:
                    text_value = ""

                raw_lines.append(text_value)

                try:
                    source_rel = str(audio_path.relative_to(input_dir))
                except ValueError:
                    source_rel = audio_path.name

                seg_schema = SegmentSchema(
                    id=f"{conversation_id}_s{idx:03d}",
                    conversation_id=conversation_id,
                    source_file=source_rel,
                    start=seg.start,
                    end=seg.end,
                    speaker=seg.speaker,
                    text=text_value,
                    emotion=emotion_map,
                    pitch=pitch_extractor(audio, sr, synthetic),
                    loudness=loudness_extractor(audio, sr, synthetic),
                    tempo=tempo_extractor(audio, sr, synthetic, text_value),
                    dialect=dialect_scorer(audio, sr, synthetic, text_value),
                    highlights=lexicon_highlighter(text_value, synthetic),
                    words=words_schema,
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
                        "llm_model": settings.llm_model if llm_client is not None and settings.llm_enabled else "disabled",
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

            if settings.comparison_html:
                search_dirs: List[Path] = []
                if ref_dir_path is not None:
                    search_dirs.append(ref_dir_path)
                parent_dir = audio_path.parent.resolve()
                if parent_dir not in search_dirs:
                    search_dirs.append(parent_dir)

                ref_path: Optional[Path] = None
                for directory in search_dirs:
                    candidate = _find_reference(conversation_id, directory)
                    if candidate is not None:
                        ref_path = candidate
                        break

                if ref_path is None:
                    console.print(
                        f"[dim]Reference not found for {conversation_id}; skipping comparison HTML.[/dim]"
                    )
                else:
                    try:
                        report_root = conversation_dir / "viz"
                        result = generate_comparison_html(
                            segments_path,
                            ref_path,
                            report_root,
                            title=f"{conversation_id} comparison",
                            theme=comparison_theme,
                            timestamped=False,
                        )
                        console.print(
                            f"[green]Generated comparison HTML[/green] {result.html_path.relative_to(output_dir)}"
                        )
                    except Exception as exc:
                        console.print(
                            f"[yellow]Failed to generate comparison HTML for {conversation_id}: {exc}[/yellow]"
                        )

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
