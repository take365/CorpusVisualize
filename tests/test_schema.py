from datetime import datetime

from pipeline.types import SegmentSchema


def test_segment_schema_validation():
    seg = SegmentSchema(
        id="conv_s000",
        conversation_id="conv",
        source_file="conv.wav",
        start=0.0,
        end=2.5,
        speaker="A",
        text="なるほど、それで進めましょう。",
        emotion={"neutral": 0.6, "joy": 0.2, "anger": 0.1, "sad": 0.1},
        pitch=[210.0, 212.0],
        loudness=0.5,
        tempo=3.2,
        dialect={"kansai": 0.2, "kanto": 0.2, "tohoku": 0.2, "kyushu": 0.2, "hokkaido": 0.2},
        highlights=[{"startChar": 0, "endChar": 5, "tag": "dialect"}],
        created_at=datetime.utcnow(),
        analyzer={
            "asr": "dummy",
            "diarization": "energy_split",
            "emotion": "energy_based",
            "pitch": "yin",
            "loudness": "rms",
            "tempo": "chars_per_sec",
            "dialect": "lexicon_tfidf",
            "lexicon": "pos_dict",
        },
    )

    assert seg.end > seg.start
    assert seg.emotion["neutral"] > 0
