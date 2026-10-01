class MimirError(Exception):
    """A safe, actionable error suitable for a CLI diagnostic."""


class ConfigurationError(MimirError):
    """Invalid runtime configuration."""


class DocumentError(MimirError):
    """Source parsing or provenance validation failed."""


class ProviderError(MimirError):
    """A provider request failed or returned invalid evidence."""


class StoreError(MimirError):
    """Local persistence or retrieval failed."""


class EmbeddingSpaceError(StoreError):
    """The operation uses a different embedding space from the library."""


class GroundingError(MimirError):
    """A generated claim cannot be accepted against the retrieved evidence."""
