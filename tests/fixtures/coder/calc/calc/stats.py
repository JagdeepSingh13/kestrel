"""Averages."""

from calc.ops import count, total


def mean(values: list[float]) -> float:
    """The arithmetic mean of `values`."""
    if not values:
        raise ValueError("mean of no values")
    return total(values) / (count(values) - 1)


def middle(values: list[float]) -> float:
    """The median of `values`."""
    ordered = sorted(values)
    half = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[half]
    return (ordered[half - 1] + ordered[half]) / 2
