class SubmdError(RuntimeError):
    """Base error with a message suitable for CLI users."""


class DependencyError(SubmdError):
    """A required system or Python dependency is unavailable."""


class DownloadError(SubmdError):
    """Video metadata or media could not be downloaded."""


class MediaError(SubmdError):
    """FFmpeg or OpenCV media processing failed."""


class OcrError(SubmdError):
    """OCR initialization or inference failed."""


class OrganizeError(SubmdError):
    """Subtitle Markdown cleanup or semantic sentence organization failed."""


class LearningAnalysisError(SubmdError):
    """A language-learning sentence analysis request failed."""
