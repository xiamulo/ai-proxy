from __future__ import annotations

from .privileges import (
    check_is_admin,
    is_admin,
    is_windows_admin,
    is_windows_elevated,
    run_as_admin,
)
from .system import is_linux, is_macos, is_posix, is_windows, platform_tag

__all__ = [
    "check_is_admin",
    "is_admin",
    "is_linux",
    "is_macos",
    "is_posix",
    "is_windows",
    "is_windows_admin",
    "is_windows_elevated",
    "platform_tag",
    "run_as_admin",
]
