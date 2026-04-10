from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DebugModeToggleHandler:
    proxy_ui: Any | None = None
    runtime_options: Any | None = None

    def bind(self, *, proxy_ui: Any, runtime_options: Any) -> None:
        self.proxy_ui = proxy_ui
        self.runtime_options = runtime_options

    def __call__(self) -> None:
        if self.proxy_ui is None or self.runtime_options is None:
            return
        enabled = bool(self.runtime_options.debug_mode_var.get())
        self.proxy_ui.set_network_env_precheck_enabled(enabled)
