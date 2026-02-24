from __future__ import annotations

import time
from typing import Callable


def run_loop(interval_seconds: int, fn: Callable[[], None], logger=None) -> None:
    while True:
        start = time.time()
        try:
            fn()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if logger:
                logger.exception(f"loop error: {e}")
        elapsed = time.time() - start
        time.sleep(max(1, interval_seconds - int(elapsed)))
