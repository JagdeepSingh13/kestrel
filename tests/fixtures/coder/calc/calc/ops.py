"""Small helpers used by stats."""


def total(values: list[float]) -> float:
    """The sum of `values`."""
    result = 0.0
    for value in values:
        result += value
    return result


def count(values: list[float]) -> int:
    """How many values there are."""
    return len(values)
