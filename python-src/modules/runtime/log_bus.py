from __future__ import annotations

from modules.platform.platform_context import get_platform

platform = get_platform()
if platform == "tauri":
    from modules.runtime import log_bus_tauri as _impl
elif platform == "legacy":
    from modules.runtime import log_bus_legacy as _impl
else:
    raise RuntimeError(f"Unsupported MTGA_PLATFORM: {platform}")

push_log = _impl.push_log
pull_logs = _impl.pull_logs

__all__ = ["push_log", "pull_logs"]
