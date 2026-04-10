from __future__ import annotations

import os
import sys


def is_windows() -> bool:
    return os.name == "nt"


def is_posix() -> bool:
    return os.name == "posix"


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def platform_tag() -> str:
    if is_macos():
        return "macos"
    if is_windows():
        return "windows"
    if is_linux():
        return "linux"
    return sys.platform or os.name


__all__ = ["is_windows", "is_posix", "is_macos", "is_linux", "platform_tag"]
