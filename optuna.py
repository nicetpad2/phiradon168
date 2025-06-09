"""Stub module to satisfy `import optuna` in test environment."""

class Trial:
    """Dummy Trial class placeholder."""
    pass


def create_study(*args, **kwargs):
    """Stub for Optuna's create_study; raises if used without real Optuna."""
    raise ImportError("Optuna is not installed in the test environment.")
