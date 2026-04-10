from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import requests

from modules.proxy.proxy_config import (
    ANTHROPIC_NATIVE_MODEL_DISCOVERY,
    ANTHROPIC_PROVIDER,
    GEMINI_NATIVE_BEARER_MODEL_DISCOVERY,
    GEMINI_NATIVE_X_GOOG_API_KEY_MODEL_DISCOVERY,
    GEMINI_PROVIDER,
    ProxyConfig,
    normalize_middle_route,
    normalize_provider,
    provider_supports_model_discovery,
)
from modules.proxy.upstream_adapter import LiteLLMUpstreamAdapter, normalize_upstream_error

HTTP_OK = 200
CONTENT_PREVIEW_LEN = 120
GENERATION_TEST_PROMPT = "你是谁"


@dataclass(frozen=True)
class ModelDiscoveryResult:
    model_ids: list[str]
    ok: bool
    strategy_id: str | None = None


class ModelDiscoveryStrategy:
    def __init__(
        self,
        *,
        id: str,
        label: str,
        path: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.id = id
        self.label = label
        self.path = path
        self.headers = headers or {}


class UpstreamErrorInfoLike(Protocol):
    status_code: int
    detail_text: str


def _read_config_str(config_group: dict[str, Any], key: str) -> str:
    value = config_group.get(key)
    return value if isinstance(value, str) else ""


def _coerce_object_dict(payload: object) -> dict[str, object] | None:
    if isinstance(payload, dict):
        return cast(dict[str, object], payload)
    return None


def _extract_model_items(payload: object) -> list[dict[str, object]]:
    payload_dict = _coerce_object_dict(payload)
    if payload_dict is None:
        return []
    data_obj = payload_dict.get("data")
    if not isinstance(data_obj, list):
        return []
    items: list[dict[str, object]] = []
    for item in cast(list[object], data_obj):
        item_dict = _coerce_object_dict(item)
        if item_dict is not None:
            items.append(item_dict)
    return items


def _extract_model_id(item: dict[str, object]) -> str | None:
    model_id = item.get("id")
    if isinstance(model_id, str) and model_id.strip():
        return model_id.strip()
    return None


def _coerce_payload_dict(payload: object) -> dict[str, object] | None:
    payload_dict = _coerce_object_dict(payload)
    if payload_dict is not None:
        return payload_dict

    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=False)
        dumped_dict = _coerce_object_dict(dumped)
        if dumped_dict is not None:
            return dumped_dict

    return None


def _consume_stream_response(
    response_payload: Any,
) -> tuple[str, int | str]:
    """消费流式响应，返回 (拼接内容, total_tokens)。"""
    content_parts: list[str] = []
    total_tokens: int | str = "未知"
    for chunk in response_payload:
        chunk_dict = _coerce_payload_dict(chunk)
        if chunk_dict is None:
            continue
        choices_obj = chunk_dict.get("choices")
        if isinstance(choices_obj, list):
            for choice in cast(list[object], choices_obj):
                choice_dict = _coerce_object_dict(choice)
                if choice_dict is None:
                    continue
                delta = _coerce_object_dict(choice_dict.get("delta"))
                if delta is not None:
                    c = delta.get("content")
                    if isinstance(c, str):
                        content_parts.append(c)
        usage_obj = _coerce_object_dict(chunk_dict.get("usage"))
        if usage_obj is not None:
            tok = usage_obj.get("total_tokens")
            if isinstance(tok, (int, str)):
                total_tokens = tok
    return "".join(content_parts), total_tokens


def _build_model_discovery_strategies(
    config_group: dict[str, Any],
) -> list[ModelDiscoveryStrategy]:
    provider = normalize_provider(_read_config_str(config_group, "provider"))
    api_key = _read_config_str(config_group, "api_key").strip()
    strategies: list[ModelDiscoveryStrategy] = []

    if provider == ANTHROPIC_PROVIDER:
        if api_key:
            strategies.append(
                ModelDiscoveryStrategy(
                    id=ANTHROPIC_NATIVE_MODEL_DISCOVERY,
                    label="Anthropic /v1/models",
                    path="/v1/models",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                    },
                )
            )
        return strategies

    if provider == GEMINI_PROVIDER:
        if api_key:
            strategies.append(
                ModelDiscoveryStrategy(
                    id=GEMINI_NATIVE_X_GOOG_API_KEY_MODEL_DISCOVERY,
                    label="Gemini /v1beta/models?key=***",
                    path=f"/v1beta/models?key={api_key}",
                )
            )
            strategies.append(
                ModelDiscoveryStrategy(
                    id=GEMINI_NATIVE_BEARER_MODEL_DISCOVERY,
                    label="Gemini /v1beta/models + Bearer",
                    path="/v1beta/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            )
        return strategies

    if api_key:
        strategies.append(
            ModelDiscoveryStrategy(
                id="openai_compatible_bearer",
                label="OpenAI Compatible /v1/models",
                path="/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        )
    return strategies


def _log_response_error(response: requests.Response, log_func: Callable[[str], None]) -> None:
    log_func(f"❌ 模型列表获取失败: HTTP {response.status_code}")
    detail = response.text.strip()
    if detail:
        log_func(f"   错误信息: {detail[:200]}")


def _should_continue_model_discovery(status_code: int) -> bool:
    return status_code in {400, 401, 403, 404, 405, 415, 429, 500, 502, 503, 504}


def _build_generation_test_proxy_config(config_group: dict[str, Any]) -> ProxyConfig:
    provider = normalize_provider(_read_config_str(config_group, "provider"))
    model_discovery_strategy = config_group.get("model_discovery_strategy")
    if not isinstance(model_discovery_strategy, str):
        model_discovery_strategy = None
    middle_route = normalize_middle_route(
        _read_config_str(config_group, "middle_route") or None,
        provider=provider,
    )
    return ProxyConfig(
        target_api_base_url=_read_config_str(config_group, "api_url"),
        custom_model_id=_read_config_str(config_group, "model_id"),
        target_model_id=_read_config_str(config_group, "model_id"),
        provider=provider,
        middle_route=middle_route,
        stream_mode=(
            config_group.get("stream_mode")
            if isinstance(config_group.get("stream_mode"), str)
            else None
        ),
        debug_mode=bool(config_group.get("debug_mode", False)),
        disable_ssl_strict_mode=bool(config_group.get("disable_ssl_strict_mode", False)),
        prompt_cache_enabled=bool(config_group.get("prompt_cache_enabled", True)),
        request_params_enabled=bool(config_group.get("request_params_enabled", True)),
        prompt_cache_bucket_id="",
        api_key=_read_config_str(config_group, "api_key"),
        mtga_auth_key="",
        model_discovery_strategy=model_discovery_strategy,
    )


def _run_generation_test_with_litellm(
    config_group: dict[str, Any],
    log_func: Callable[[str], None],
) -> None:
    provider = normalize_provider(_read_config_str(config_group, "provider"))
    model_id = _read_config_str(config_group, "model_id").strip()
    api_url = _read_config_str(config_group, "api_url").rstrip("/")
    if not api_url or not model_id:
        log_func("测活失败: API URL或模型ID为空")
        return

    proxy_config = _build_generation_test_proxy_config(config_group)
    adapter = LiteLLMUpstreamAdapter(
        disable_ssl_strict_mode=proxy_config.disable_ssl_strict_mode,
        log_func=log_func,
    )
    try:
        route = adapter.build_route(proxy_config)
        test_data: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": GENERATION_TEST_PROMPT}],
            "max_tokens": 1,
            "stream": True,
        }
        if route.request_params_enabled:
            test_data["temperature"] = 0
        log_func(
            "正在测活模型: "
            f"{model_id} (provider={provider}, request_api={route.request_api}, 会消耗少量tokens)"
        )

        response_payload = adapter.create_chat_completion(
            route=route,
            request_data=test_data,
        )

        # 处理流式响应：逐块消费，收集内容片段
        try:
            content, total_tokens = _consume_stream_response(response_payload)
        except Exception:  # noqa: BLE001
            content, total_tokens = "", "未知"

        log_func(f"✅ 模型测活成功: {model_id}")
        if content:
            preview = content[:CONTENT_PREVIEW_LEN]
            suffix = "..." if len(content) > CONTENT_PREVIEW_LEN else ""
            log_func(f"   响应内容: {preview}{suffix}")
        if total_tokens != "未知":
            log_func(f"   消耗tokens: {total_tokens}")
    except Exception as exc:  # noqa: BLE001
        error_info = cast(UpstreamErrorInfoLike, normalize_upstream_error(exc))
        log_func(f"❌ 模型测活失败: HTTP {error_info.status_code}")
        detail = error_info.detail_text.strip() or None
        if detail is not None:
            log_func(f"   错误信息: {detail[:200]}")
    finally:
        adapter.close()


def _try_fetch_model_payload(
    *,
    api_url: str,
    strategy: ModelDiscoveryStrategy,
    log_func: Callable[[str], None],
    model_id: str | None = None,
) -> tuple[Any | None, bool]:
    test_url = f"{api_url}{strategy.path}"
    suffix = f": {model_id}" if model_id else ""
    log_func(f"正在获取模型列表 ({strategy.label}): {test_url}")

    try:
        response = requests.get(test_url, headers=strategy.headers, timeout=10)
    except requests.exceptions.Timeout:
        log_func(f"❌ 模型列表获取超时{suffix}")
        payload = None
        should_continue = True
    except requests.exceptions.RequestException as exc:
        log_func(f"❌ 模型列表获取网络错误{suffix}: {str(exc)}")
        payload = None
        should_continue = True
    except Exception as exc:
        log_func(f"❌ 模型列表获取意外错误{suffix}: {str(exc)}")
        payload = None
        should_continue = False
    else:
        if response.status_code != HTTP_OK:
            _log_response_error(response, log_func)
            payload = None
            should_continue = _should_continue_model_discovery(response.status_code)
        else:
            log_func("✅ 模型列表获取成功")
            payload = response.json()
            if payload is None:
                should_continue = True
            elif not _extract_model_items(payload):
                log_func("❌ 响应中未发现模型列表")
                payload = None
                should_continue = True
            else:
                should_continue = False

    return payload, should_continue


def _fetch_model_payload(
    config_group: dict[str, Any],
    log_func: Callable[[str], None],
    *,
    model_id: str | None = None,
) -> tuple[Any | None, str | None]:
    provider = normalize_provider(_read_config_str(config_group, "provider"))
    api_url = _read_config_str(config_group, "api_url").rstrip("/")
    if not api_url:
        log_func("检查失败: API URL为空")
        return None, None
    if not provider_supports_model_discovery(provider):
        log_func("当前提供商不支持通过 /models 自动发现模型，请直接手填实际模型ID")
        return None, None

    strategies = _build_model_discovery_strategies(config_group)
    if not strategies:
        log_func("当前提供商没有可用的模型发现策略")
        return None, None

    for strategy in strategies:
        payload, should_continue = _try_fetch_model_payload(
            api_url=api_url,
            strategy=strategy,
            log_func=log_func,
            model_id=model_id,
        )
        if payload is not None:
            return payload, strategy.id
        if not should_continue:
            return None, None
        log_func("尝试降级到下一种模型发现策略")

    return None, None


def fetch_models(
    config_group: dict[str, Any],
    log_func: Callable[[str], None],
) -> tuple[list[str], str | None]:
    payload, strategy_id = _fetch_model_payload(config_group, log_func)
    items = _extract_model_items(payload)
    model_ids = [
        model_id
        for item in items
        if (model_id := _extract_model_id(item)) is not None
    ]
    return model_ids, strategy_id


def fetch_model_list_result(
    config_group: dict[str, Any],
    log_func: Callable[[str], None],
) -> ModelDiscoveryResult:
    model_ids, strategy_id = fetch_models(config_group, log_func)
    return ModelDiscoveryResult(
        model_ids=sorted(model_ids),
        ok=bool(model_ids),
        strategy_id=strategy_id,
    )


def run_model_test(config_group: dict[str, Any], log_func: Callable[[str], None]) -> None:
    _run_generation_test_with_litellm(config_group, log_func)


def test_chat_completion(
    config_group: dict[str, Any],
    log_func: Callable[[str], None],
    *,
    thread_manager: Any | None = None,
) -> None:
    _ = thread_manager
    run_model_test(config_group, log_func)
