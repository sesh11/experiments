def median(values):
    """Return the median of a non-empty list of numbers."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid]
