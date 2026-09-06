from pathlib import Path

import pytest

from app.config import TranscriptionSection
from app.exceptions import TranscriptionError
from app.services import transcription_service
from app.services.transcription_service import (
    TranscriptionService,
    find_cached_whisper_model,
)


def _snapshot(cache: Path, model: str) -> Path:
    snapshot = (
        cache
        / f"models--Systran--faster-whisper-{model}"
        / "snapshots"
        / "revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    return snapshot


def test_partial_model_cache_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(transcription_service, "MINIMUM_MODEL_SIZE", 1)
    _snapshot(tmp_path, "large-v3")

    assert find_cached_whisper_model("large-v3", tmp_path) is None
    with pytest.raises(TranscriptionError, match="не установлена полностью"):
        TranscriptionService(
            TranscriptionSection(model="large-v3")
        ).require_local_model(tmp_path)


def test_complete_model_cache_is_resolved(tmp_path, monkeypatch):
    monkeypatch.setattr(transcription_service, "MINIMUM_MODEL_SIZE", 1)
    snapshot = _snapshot(tmp_path, "medium")
    (snapshot / "model.bin").write_bytes(b"model")

    assert find_cached_whisper_model("medium", tmp_path) == snapshot.resolve()
