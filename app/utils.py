from __future__ import annotations

import time
import uuid


def now_ms() -> str:
    return str(int(time.time() * 1000))


def gen_client_id(prefix: str = "bot") -> str:
    return f"{prefix}-{uuid.uuid4()}"
