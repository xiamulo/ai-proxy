from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProxyState:
    instance: Any | None = None


_STATE = ProxyState()


def get_proxy_instance() -> Any | None:
    """读取当前代理实例"""
    return _STATE.instance


def set_proxy_instance(instance: Any | None) -> None:
    """更新当前代理实例"""
    _STATE.instance = instance
