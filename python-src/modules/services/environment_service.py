from __future__ import annotations

from collections.abc import Callable


def check_environment(*, check_resources: Callable[[], list[str]]) -> tuple[bool, str]:
    """检查运行环境所需资源是否齐全。"""
    missing_resources = check_resources()

    if missing_resources:
        error_msg = "环境检查失败，缺少以下资源:\n" + "\n".join(missing_resources)
        return False, error_msg

    return True, "环境检查通过"
