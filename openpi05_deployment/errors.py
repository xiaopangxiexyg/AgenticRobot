"""Error types for OpenPI deployment integration."""


class OpenPIDeploymentError(RuntimeError):
    """Base error for OpenPI deployment failures."""


class ServerMetadataError(OpenPIDeploymentError):
    """Raised when the OpenPI server metadata is missing or incompatible."""


class ObservationAdapterError(OpenPIDeploymentError):
    """Raised when a deployment observation cannot be adapted safely."""


class InferenceResultError(OpenPIDeploymentError):
    """Raised when a server inference result violates the runtime contract."""


class DatasetTrajectoryError(OpenPIDeploymentError):
    """Raised when a dataset-backed deployment trajectory cannot be used safely."""
