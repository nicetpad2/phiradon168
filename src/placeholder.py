"""Simple placeholder module for coverage tests."""

# [Patch] Basic arithmetic functions for demo coverage

def add(a: int | float, b: int | float) -> int | float:
    """Return the sum of two numbers."""
    return a + b


def multiply(a: int | float, b: int | float) -> int | float:
    """Return the product of two numbers."""
    return a * b


# [Patch] Utility for coverage calculations
def required_lines_for_target(
    total_lines: int, current_covered: int, target_pct: float
) -> int:
    """Return additional covered lines needed to reach ``target_pct``.

    Parameters
    ----------
    total_lines : int
        Total number of lines measured.
    current_covered : int
        Number of lines currently covered by tests.
    target_pct : float
        Desired coverage expressed as a decimal (e.g., ``0.65`` for 65%).

    Returns
    -------
    int
        Additional lines that must be covered. If the target is already
        reached, ``0`` is returned.
    """

    if total_lines <= 0:
        raise ValueError("total_lines must be > 0")
    if not 0 <= target_pct <= 1:
        raise ValueError("target_pct must be between 0 and 1")

    target_lines = int(total_lines * target_pct)
    return max(target_lines - current_covered, 0)

