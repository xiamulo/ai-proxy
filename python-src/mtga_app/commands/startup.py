from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from pytauri import Commands

from modules.runtime.operation_result import OperationResult

from .common import build_result_payload, collect_logs

_logger = logging.getLogger(__name__)
_STARTUP_CONTEXT_TIMEOUT_S = 10

if TYPE_CHECKING:
    from modules.runtime.resource_manager import ResourceManager
    from modules.services.startup_context import StartupContext


@lru_cache(maxsize=1)
def _get_hosts_state_module() -> Any:
    return import_module("modules.hosts.hosts_state")


@lru_cache(maxsize=1)
def _get_environment_service_module() -> Any:
    return import_module("modules.services.environment_service")


@lru_cache(maxsize=1)
def _get_resource_manager_module() -> Any:
    return import_module("modules.runtime.resource_manager")


@lru_cache(maxsize=1)
def _get_startup_context_module() -> Any:
    return import_module("modules.services.startup_context")


@lru_cache(maxsize=1)
def _get_resource_manager() -> ResourceManager:
    resource_manager_cls = _get_resource_manager_module().ResourceManager
    return resource_manager_cls()


@lru_cache(maxsize=1)
def _get_startup_context() -> StartupContext | None:
    build_startup_context = _get_startup_context_module().build_startup_context
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(build_startup_context)
            return future.result(timeout=_STARTUP_CONTEXT_TIMEOUT_S)
    except FuturesTimeoutError:
        _logger.warning("build_startup_context timed out after %ss", _STARTUP_CONTEXT_TIMEOUT_S)
        return None
    except Exception as exc:  # noqa: BLE001
        _logger.warning("build_startup_context failed: %s", exc)
        return None


def _check_environment() -> tuple[bool, str]:
    resource_manager = _get_resource_manager()
    check_environment = cast(
        Callable[..., tuple[bool, str]],
        _get_environment_service_module().check_environment,
    )
    return check_environment(
        check_resources=resource_manager.check_resources,
    )


def register_startup_commands(commands: Commands) -> None:
    @commands.command()
    async def startup_status() -> dict[str, Any]:
      try:
        hosts_state_module = _get_hosts_state_module()
        resource_manager_module = _get_resource_manager_module()
        logs, _log_func = collect_logs()
        context = _get_startup_context()
        env_ok, env_message = _check_environment()
        get_packaging_runtime = cast(
            Callable[[], str],
            resource_manager_module.get_packaging_runtime,
        )
        get_legacy_user_data_dir = cast(
            Callable[[], str],
            resource_manager_module.get_legacy_user_data_dir,
        )
        has_legacy_user_data_dir = cast(
            Callable[[], bool],
            resource_manager_module.has_legacy_user_data_dir,
        )
        get_hosts_modify_block_state = hosts_state_module.get_hosts_modify_block_state
        allow_unsafe_hosts_flag = cast(
            str,
            hosts_state_module.ALLOW_UNSAFE_HOSTS_FLAG,
        )
        runtime = get_packaging_runtime()
        block_state = get_hosts_modify_block_state()
        hosts_preflight_report = context.hosts_preflight_report if context else None
        hosts_preflight_ok = None
        hosts_preflight_status = None
        if hosts_preflight_report is not None:
            hosts_preflight_ok = bool(getattr(hosts_preflight_report, "ok", False))
            status = getattr(hosts_preflight_report, "status", None)
            if status is not None:
                hosts_preflight_status = getattr(status, "value", str(status))

        block_status = None
        block_report = block_state.report
        if block_report is not None:
            status = getattr(block_report, "status", None)
            if status is not None:
                block_status = getattr(status, "value", str(status))

        explicit_proxy_detected = False
        network_env_report = context.network_env_report if context else None
        if network_env_report is not None:
            explicit_proxy_detected = bool(
                getattr(network_env_report, "explicit_proxy_detected", False)
            )

        result = OperationResult.success(
            env_ok=env_ok,
            env_message=env_message,
            runtime=runtime,
            legacy_user_data_dir_detected=has_legacy_user_data_dir(),
            legacy_user_data_dir=get_legacy_user_data_dir(),
            allow_unsafe_hosts_flag=allow_unsafe_hosts_flag,
            hosts_modify_blocked=block_state.blocked,
            hosts_modify_block_status=block_status,
            hosts_preflight_ok=hosts_preflight_ok,
            hosts_preflight_status=hosts_preflight_status,
            explicit_proxy_detected=explicit_proxy_detected,
        )
        return build_result_payload(result, logs, "")
      except Exception as exc:  # noqa: BLE001
        _logger.warning("startup_status failed: %s", exc)
        fallback_logs: list[str] = [f"启动状态获取异常: {exc}"]
        fallback_result = OperationResult.failure(f"启动状态获取失败: {exc}")
        return build_result_payload(fallback_result, fallback_logs, "启动状态")

    _ = startup_status
