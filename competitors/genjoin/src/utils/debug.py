# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Added for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Addition: a timing decorator for the local GenJoin reproduction code.

"""Debug utils — timing decorator, etc."""

import time
from functools import wraps


def timeit_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMING] {func.__name__}: {(end - start) * 1000:.2f}ms")
        return result
    return wrapper
