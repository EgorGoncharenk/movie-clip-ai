import pytest
from pydantic import ValidationError

from app.config import Settings
from app.cli import _apply_transcription_overrides


def test_default_config_is_vertical():
    settings = Settings()
    assert settings.encoding.width == 1080
    assert settings.encoding.height == 1920
    assert settings.transcription.model == "medium"


def test_horizontal_output_is_rejected():
    with pytest.raises(ValidationError):
        Settings.model_validate(
            {"encoding": {"width": 1920, "height": 1080}}
        )


def test_unknown_whisper_model_is_rejected():
    with pytest.raises(ValidationError):
        Settings.model_validate(
            {"transcription": {"model": "imaginary"}}
        )


def test_cli_overrides_are_validated():
    with pytest.raises(ValidationError):
        _apply_transcription_overrides(
            Settings(),
            language=None,
            model="imaginary",
            device=None,
        )
