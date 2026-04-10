from __future__ import annotations

import os

_SUPPORTED = {"tauri", "legacy"}


def get_platform() -> str:
    value = os.environ.get("MTGA_PLATFORM", "").strip().lower()
    if value in _SUPPORTED:
        return value
    raise RuntimeError(
        "MTGA_PLATFORM 未设置或非法（仅支持 tauri/legacy），请在入口处注入。"
    )


__all__ = ["get_platform"]
