"""Application-specific exceptions with user-facing messages."""


class ClipAppError(Exception):
    """Base exception for expected application failures."""


class EnvironmentCheckError(ClipAppError):
    """Raised when a required external component is unavailable."""


class MediaProbeError(ClipAppError):
    """Raised when FFprobe cannot read an input file."""


class TranscriptionError(ClipAppError):
    """Raised when speech recognition cannot be completed."""


class ExportError(ClipAppError):
    """Raised when a clip cannot be rendered."""


class InvalidClipRangeError(ClipAppError):
    """Raised when a manually supplied clip range is invalid."""


class OperationCancelled(ClipAppError):
    """Raised when the user requests cooperative cancellation."""
