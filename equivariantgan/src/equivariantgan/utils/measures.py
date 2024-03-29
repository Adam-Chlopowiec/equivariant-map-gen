from contextlib import contextmanager
from time import perf_counter


@contextmanager
def measure_time() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
