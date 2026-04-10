from __future__ import annotations

import os
import ssl
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import litellm

from modules.proxy.proxy_config import (
    OPENAI_CHAT_COMPLETION_PROVIDER,
    OPENAI_PROVIDER_IDS,
    OPENAI_RESPONSE_PROVIDER,
    ProxyConfig,
    normalize_middle_route,
    normalize_provider,
)
from modules.proxy.upstream_param_self_heal import (
    UpstreamParamSelfHealController,
)

type LogFunc = Callable[[str], None]
type RequestApi = Literal["chat_completions", "responses"]

CHAT_COMPLETIONS_REQUEST_API: RequestApi = "chat_completions"
RESPONSES_REQUEST_API: RequestApi = "responses"
LITELLM_CONNECT_RETRY_COUNT = 2
UPSTREAM_PARAM_SELF_HEAL_MAX_ATTEMPTS = 3
OPENAI_CHAT_COMPLETION_STANDARD_PARAMS: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "top_p",
        "n",
        "stream",
        "stream_options",
        "stop",
        "max_tokens",
        "max_completion_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "response_format",
        "seed",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "logprobs",
        "top_logprobs",
        "parallel_tool_calls",
        "service_tier",
        "reasoning_effort",
        "prediction",
        "modalities",
        "audio",
        "metadata",
        "store",
        "extra_body",
        "allowed_openai_params",
    }
)
NON_OPENAI_REQUIRED_CHAT_PARAMS: frozenset[str] = frozenset({"messages", "model"})
OPENAI_COMPATIBLE_META_PARAMS: frozenset[str] = frozenset(
    {"messages", "model", "extra_body", "allowed_openai_params"}
)
_litellm_compat_patch_state = {"applied": False}
_litellm_module = cast(Any, litellm)
_get_supported_openai_params = cast(
    Callable[..., list[object] | None],
    _litellm_module.get_supported_openai_params,
)
_create_litellm_completion = cast(Callable[..., Any], _litellm_module.completion)


@dataclass(frozen=True)
class UpstreamRoute:
    provider: str
    request_api: RequestApi
    litellm_model: str
    base_url: str
    api_key: str
    prompt_cache_enabled: bool
    request_params_enabled: bool
    middle_route_applied: bool
    middle_route_ignored: bool
    litellm_base_url: str = ""
    model_discovery_strategy: str | None = None
    prompt_cache_key: str = ""


@dataclass(frozen=True)
class UpstreamErrorInfo:
    status_code: int
    detail_text: str
    log_message: str
    response_body: dict[str, Any]


def _build_litellm_base_url(
    *,
    provider: str,
    chat_base_url: str,
    target_api_base_url: str,
    middle_route: str,
) -> str:
    if provider in OPENAI_PROVIDER_IDS:
        return chat_base_url
    return target_api_base_url.rstrip("/") + (middle_route or "")


def _build_prompt_cache_key(bucket_id: str) -> str:
    return bucket_id.strip().lower()


def _is_litellm_known_model(model_id: str) -> bool:
    """Check if litellm can resolve this model name without an explicit provider prefix."""
    try:
        litellm.get_llm_provider(model_id)
        return True
    except Exception:  # noqa: BLE001
        return False


def _ensure_openai_prefix(model_id: str) -> str:
    """Auto-add ``openai/`` prefix for models that litellm cannot resolve on its own."""
    if "/" in model_id:
        return model_id
    if _is_litellm_known_model(model_id):
        return model_id
    return f"openai/{model_id}"


def build_upstream_route(
    proxy_config: ProxyConfig,
    *,
    fallback_api_key: str = "",
) -> UpstreamRoute:
    effective_provider = normalize_provider(proxy_config.provider)
    request_api: RequestApi = (
        RESPONSES_REQUEST_API
        if effective_provider == OPENAI_RESPONSE_PROVIDER
        else CHAT_COMPLETIONS_REQUEST_API
    )
    middle_route = normalize_middle_route(
        proxy_config.middle_route,
        provider=effective_provider,
    )
    base_url = proxy_config.target_api_base_url.rstrip("/") + middle_route
    if effective_provider == OPENAI_RESPONSE_PROVIDER:
        litellm_model = proxy_config.target_model_id
    elif effective_provider == OPENAI_CHAT_COMPLETION_PROVIDER:
        litellm_model = _ensure_openai_prefix(proxy_config.target_model_id)
    else:
        litellm_model = f"{effective_provider}/{proxy_config.target_model_id}"
    litellm_base_url = _build_litellm_base_url(
        provider=effective_provider,
        chat_base_url=base_url,
        target_api_base_url=proxy_config.target_api_base_url,
        middle_route=middle_route,
    )
    return UpstreamRoute(
        provider=effective_provider,
        request_api=request_api,
        litellm_model=litellm_model,
        base_url=base_url,
        api_key=(proxy_config.api_key or fallback_api_key).strip(),
        prompt_cache_enabled=proxy_config.prompt_cache_enabled,
        request_params_enabled=proxy_config.request_params_enabled,
        middle_route_applied=True,
        middle_route_ignored=False,
        litellm_base_url=litellm_base_url,
        model_discovery_strategy=proxy_config.model_discovery_strategy,
        prompt_cache_key=_build_prompt_cache_key(proxy_config.prompt_cache_bucket_id),
    )


class LiteLLMUpstreamAdapter:
    def __init__(
        self,
        *,
        disable_ssl_strict_mode: bool,
        log_func: LogFunc,
    ) -> None:
        self._disable_ssl_strict_mode = disable_ssl_strict_mode
        self._log = log_func
        self._param_self_heal = UpstreamParamSelfHealController()

    def build_route(
        self,
        proxy_config: ProxyConfig,
        *,
        fallback_api_key: str = "",
    ) -> UpstreamRoute:
        return build_upstream_route(proxy_config, fallback_api_key=fallback_api_key)

    @staticmethod
    def _resolve_chat_completion_model(route: UpstreamRoute) -> str:
        if route.request_api != RESPONSES_REQUEST_API:
            return route.litellm_model
        if route.litellm_model.startswith("responses/"):
            return route.litellm_model
        return f"responses/{route.litellm_model}"

    def _get_supported_openai_params(
        self,
        route: UpstreamRoute,
        *,
        custom_llm_provider: str | None = None,
    ) -> list[str] | None:
        try:
            supported_params = _get_supported_openai_params(
                model=route.litellm_model,
                custom_llm_provider=custom_llm_provider,
            )
            if supported_params is None:
                return None
            return [param for param in supported_params if isinstance(param, str)]
        except Exception:
            return None

    @staticmethod
    def _strip_optional_request_params(call_kwargs: dict[str, Any]) -> None:
        for key in (
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
            "seed",
            "service_tier",
            "reasoning_effort",
            "prediction",
            "modalities",
            "audio",
            "metadata",
            "store",
            "extra_body",
            "allowed_openai_params",
        ):
            call_kwargs.pop(key, None)

    def _drop_unsupported_standard_params(
        self,
        *,
        route: UpstreamRoute,
        call_kwargs: dict[str, Any],
        allowed_params: set[str] | frozenset[str],
    ) -> None:
        dropped_params = [
            key
            for key in list(call_kwargs)
            if key in OPENAI_CHAT_COMPLETION_STANDARD_PARAMS
            and key not in allowed_params
        ]
        for key in dropped_params:
            call_kwargs.pop(key, None)
        if dropped_params:
            dropped_list = ", ".join(sorted(dropped_params))
            self._log(
                f"provider={route.provider} "
                f"request_api={route.request_api} "
                f"model={route.litellm_model} "
                f"已忽略不兼容参数: {dropped_list}"
            )

    def _normalize_openai_compatible_request(
        self,
        *,
        route: UpstreamRoute,
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        call_kwargs: dict[str, Any] = dict(request_data)
        if not route.request_params_enabled:
            self._strip_optional_request_params(call_kwargs)
            return call_kwargs
        supported_params = self._get_supported_openai_params(
            route,
            custom_llm_provider="openai",
        )
        top_level_params = set(OPENAI_COMPATIBLE_META_PARAMS)
        if supported_params is not None:
            top_level_params.update(supported_params)
            self._drop_unsupported_standard_params(
                route=route,
                call_kwargs=call_kwargs,
                allowed_params=top_level_params,
            )
        return call_kwargs

    def _normalize_provider_chat_request(
        self,
        *,
        route: UpstreamRoute,
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        if route.provider in OPENAI_PROVIDER_IDS:
            return self._normalize_openai_compatible_request(
                route=route,
                request_data=request_data,
            )
        call_kwargs = dict(request_data)
        if not route.request_params_enabled:
            self._strip_optional_request_params(call_kwargs)
            return call_kwargs
        supported_params = self._get_supported_openai_params(route)
        if supported_params is None:
            return call_kwargs
        self._drop_unsupported_standard_params(
            route=route,
            call_kwargs=call_kwargs,
            allowed_params=NON_OPENAI_REQUIRED_CHAT_PARAMS | set(supported_params),
        )
        return call_kwargs

    @staticmethod
    def _build_request_ssl_verify(
        disable_ssl_strict_mode: bool,
    ) -> bool | ssl.SSLContext:
        if not disable_ssl_strict_mode:
            return True
        cafile = os.getenv("SSL_CERT_FILE") or None
        ssl_context = ssl.create_default_context(cafile=cafile)
        strict_flag = getattr(ssl, "VERIFY_X509_STRICT", 0)
        if strict_flag and hasattr(ssl_context, "verify_flags"):
            ssl_context.verify_flags &= ~strict_flag
        return ssl_context

    def create_chat_completion(
        self,
        *,
        route: UpstreamRoute,
        request_data: dict[str, Any],
    ) -> Any:
        call_kwargs = self._normalize_provider_chat_request(
            route=route,
            request_data=request_data,
        )
        call_kwargs["model"] = self._resolve_chat_completion_model(route)
        call_kwargs["base_url"] = route.base_url
        call_kwargs["api_key"] = route.api_key
        call_kwargs["ssl_verify"] = self._build_request_ssl_verify(
            self._disable_ssl_strict_mode
        )
        return _create_litellm_completion(**call_kwargs)

    def close(self) -> None:
        return None


def normalize_upstream_error(exc: Exception) -> UpstreamErrorInfo:
    detail = str(exc)
    status_code = getattr(exc, "status_code", None) or 503
    normalized_status_code = int(status_code)
    return UpstreamErrorInfo(
        status_code=normalized_status_code,
        detail_text=detail,
        log_message=f"上游请求失败: HTTP {normalized_status_code} - {detail}",
        response_body={
            "error": {
                "message": detail,
                "type": "upstream_error",
            }
        },
    )
