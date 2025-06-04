# [Patch v5.5.14] Custom exception for pipeline failures
class PipelineError(Exception):
    """Raised when a pipeline stage fails."""
    pass
