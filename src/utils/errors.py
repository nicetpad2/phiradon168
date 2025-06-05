# [Patch v5.5.14] Custom exception for pipeline failures
class PipelineError(Exception):
    """Raised when a pipeline stage fails."""
    pass


# [Patch v5.7.3] Helper to log stack trace then re-raise exceptions
def log_and_raise(exc: Exception, log_file: str) -> None:
    """Write stack trace to ``log_file`` and re-raise ``exc``."""
    import traceback
    with open(log_file, "a", encoding="utf-8") as f:
        traceback.print_exc(file=f)
    raise exc
