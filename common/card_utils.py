import os
from typing import Any, List, Tuple


def q_error_value(x: float, y: float):
    x = max(1, x)
    y = max(1, y)
    return max(x / y, y / x)


def normalize_value(val: Any, min_val: Any, max_val: Any):
    return (val - min_val) / (max_val - min_val) if max_val > min_val else 0
