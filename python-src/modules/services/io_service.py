from __future__ import annotations

import io
import sys


def ensure_utf8_stdio() -> None:
    """Ensure stdout/stderr can emit UTF-8 even when Finder starts the app."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if not stream:
            continue
        encoding = getattr(stream, "encoding", None)
        if encoding and encoding.lower().startswith("utf-8"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            buffer = getattr(stream, "buffer", None)
            if buffer is None:
                continue
            try:
                new_stream = io.TextIOWrapper(
                    buffer, encoding="utf-8", errors="replace", line_buffering=True
                )
            except Exception:
                continue
            setattr(sys, name, new_stream)
        except Exception:
            pass
