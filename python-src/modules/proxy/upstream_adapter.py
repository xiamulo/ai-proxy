from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import json
import os
import re
import ssl
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast
from urllib.parse import urlparse

import litellm
from openai import OpenAI

if TYPE_CHECKING:
    from openai.types.websocket_connection_options import (
        WebSocketConnectionOptions as OpenAIWebSocketConnectionOptions,
    )
else:
    OpenAIWebSocketConnectionOptions = dict[str, Any]

from modules.proxy.proxy_config import (
    OPENAI_CHAT_COMPLETION_PROVIDER,
    OPENAI_PROVIDER_IDS,
    OPENAI_RESPONSE_PROVIDER,
    OPENAI_RESPONSES_WEBSOCKET_MODEL_ID,
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
HTTP_FORBIDDEN = 403
LITELLM_CONNECT_RETRY_COUNT = 2
UPSTREAM_PARAM_SELF_HEAL_MAX_ATTEMPTS = 3
WEBSOCKET_SESSION_IDLE_TIMEOUT_SECONDS = 15 * 60
WEBSOCKET_SESSION_MAX_COUNT = 64
OPENAI_STRICT_SCHEMA_SELF_HEAL_MAX_ATTEMPTS = 4
OPENAI_UNSUPPORTED_PARAM_SELF_HEAL_MAX_ATTEMPTS = 4
OPENAI_RESPONSES_WEBSOCKET_CONNECTION_OPTIONS: dict[str, Any] = {
    "ping_interval": None,
    "max_size": None,
}
OPENAI_UNSUPPORTED_STRICT_SCHEMA_KEYS: frozenset[str] = frozenset(
    {
        "allOf",
        "oneOf",
        "not",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
        "format",
        "propertyNames",
        "uniqueItems",
        "contains",
        "minContains",
        "maxContains",
        "prefixItems",
        "unevaluatedItems",
        "unevaluatedProperties",
        "minProperties",
        "maxProperties",
    }
)
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
        "prompt_cache_retention",
        "truncation",
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
    websocket_mode_enabled: bool
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


@dataclass
class ResponsesWebSocketFunctionCallState:
    index: int
    call_id: str
    name: str
    arguments_streamed: bool = False


@dataclass
class ResponsesWebSocketSessionState:
    session_id: str
    explicit_session_key: bool
    lock: threading.RLock = field(default_factory=threading.RLock)
    client: OpenAI | None = None
    connection_context: Any = None
    connection: Any = None
    connection_started_at: float = 0.0
    last_used_at: float = 0.0
    last_response_id: str | None = None
    last_response_connection_started_at: float = 0.0
    last_tools_fingerprint: str | None = None
    last_request_signature: str | None = None
    prompt_cache_scope: str | None = None
    prewarmed_messages_hash: str | None = None
    websocket_extra_query: dict[str, Any] | None = None
    conversation_messages: list[dict[str, Any]] = field(
        default_factory=lambda: cast(list[dict[str, Any]], [])
    )


class ResponsesWebSocketSessionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


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


def _coerce_model_dump(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return cast(dict[str, Any], dumped)
    return None


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
        websocket_mode_enabled=proxy_config.websocket_mode_enabled,
        middle_route_applied=True,
        middle_route_ignored=False,
        litellm_base_url=litellm_base_url,
        model_discovery_strategy=proxy_config.model_discovery_strategy,
        prompt_cache_key=_build_prompt_cache_key(proxy_config.prompt_cache_bucket_id),
    )


class LiteLLMUpstreamAdapter:
    _PROMPT_CACHE_ANCHOR_KEYS: ClassVar[tuple[str, ...]] = (
        "mtga_ws_session_id",
        "mtga_session_id",
        "session_id",
        "conversation_id",
        "chat_id",
        "thread_id",
    )

    def __init__(
        self,
        *,
        disable_ssl_strict_mode: bool,
        log_func: LogFunc,
    ) -> None:
        self._disable_ssl_strict_mode = disable_ssl_strict_mode
        self._log = log_func
        self._param_self_heal = UpstreamParamSelfHealController()
        self._websocket_sessions: dict[str, ResponsesWebSocketSessionState] = {}
        self._websocket_sessions_lock = threading.RLock()
        self._websocket_session_history_index: dict[
            str, dict[str, list[dict[str, Any]]]
        ] = {}
        self._websocket_session_history_hashes: dict[str, set[str]] = {}
        self._websocket_fallback_routes: set[str] = set()

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

    @staticmethod
    def _build_websocket_route_key(route: UpstreamRoute) -> str:
        return "|".join(
            [
                route.provider,
                route.request_api,
                route.base_url,
                route.litellm_model,
            ]
        )

    def _mark_websocket_route_fallback(self, route: UpstreamRoute, reason: str) -> None:
        route_key = self._build_websocket_route_key(route)
        if route_key in self._websocket_fallback_routes:
            return
        self._websocket_fallback_routes.add(route_key)
        self._log(
            "OpenAI Responses WebSocket 路由已锁定回退 HTTP: "
            f"route={route.litellm_model} reason={reason}"
        )

    def _is_websocket_route_fallback(self, route: UpstreamRoute) -> bool:
        return self._build_websocket_route_key(route) in self._websocket_fallback_routes

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
    def _build_request_prompt_cache_key(
        *,
        route: UpstreamRoute,
        request_data: dict[str, Any],
        session: ResponsesWebSocketSessionState | None = None,
    ) -> str | None:
        if not route.prompt_cache_enabled or not route.prompt_cache_key:
            return None
        cache_scope = LiteLLMUpstreamAdapter._resolve_request_prompt_cache_scope(
            request_data=request_data,
            session=session,
        )
        if cache_scope is None:
            return route.prompt_cache_key
        return f"{route.prompt_cache_key}:{cache_scope}"

    @classmethod
    def _extract_request_conversation_anchor(
        cls,
        request_data: dict[str, Any],
    ) -> str | None:
        metadata_obj = request_data.get("metadata")
        if isinstance(metadata_obj, dict):
            metadata = cast(dict[str, Any], metadata_obj)
            for key in cls._PROMPT_CACHE_ANCHOR_KEYS:
                value_obj = metadata.get(key)
                if isinstance(value_obj, str) and value_obj.strip():
                    return value_obj.strip()
        for key in cls._PROMPT_CACHE_ANCHOR_KEYS:
            value_obj = request_data.get(key)
            if isinstance(value_obj, str) and value_obj.strip():
                return value_obj.strip()
        return None

    @classmethod
    def _resolve_request_prompt_cache_scope(
        cls,
        *,
        request_data: dict[str, Any],
        session: ResponsesWebSocketSessionState | None = None,
    ) -> str | None:
        if session is not None and session.prompt_cache_scope:
            return session.prompt_cache_scope

        anchor = cls._extract_request_conversation_anchor(request_data)
        if anchor is not None:
            cache_scope = f"meta:{anchor}"
        elif session is not None and session.explicit_session_key and session.session_id.strip():
            cache_scope = f"meta:{session.session_id.strip()}"
        else:
            cache_scope = None

        messages_obj = request_data.get("messages")
        if cache_scope is None and isinstance(messages_obj, list):
            message_items: list[dict[str, Any]] = []
            for item in cast(list[Any], messages_obj):
                if isinstance(item, dict):
                    message_items.append(cast(dict[str, Any], item))
            if message_items:
                anchor_hash = _build_prompt_cache_key(
                    cls._build_messages_history_hash(message_items[:2])
                )[:16]
                cache_scope = f"anon:{anchor_hash}"

        if session is not None and cache_scope is not None:
            session.prompt_cache_scope = cache_scope
        return cache_scope

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
            "prompt_cache_retention",
            "truncation",
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
            if key in OPENAI_CHAT_COMPLETION_STANDARD_PARAMS and key not in allowed_params
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
        prompt_cache_key = self._build_request_prompt_cache_key(
            route=route,
            request_data=request_data,
        )
        if prompt_cache_key is not None:
            call_kwargs["prompt_cache_key"] = prompt_cache_key
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
    def _supports_openai_responses_websocket(
        route: UpstreamRoute, request_data: dict[str, Any]
    ) -> tuple[bool, str | None]:
        if not route.websocket_mode_enabled:
            return False, "配置组未启用 WebSocket 模式"
        if route.provider not in OPENAI_PROVIDER_IDS:
            return False, "仅 OpenAI 配置支持 WebSocket 模式"
        if route.litellm_model.strip().lower() != OPENAI_RESPONSES_WEBSOCKET_MODEL_ID:
            return False, "当前模型不是 gpt-5.4"
        if not bool(request_data.get("stream", False)):
            return False, "仅流式请求使用 WebSocket 模式"
        return True, None

    @staticmethod
    def _convert_message_content_part(part: Any) -> dict[str, Any] | None:  # noqa: PLR0911
        if isinstance(part, str):
            return {"type": "input_text", "text": part}
        if not isinstance(part, dict):
            return None
        part_dict = cast(dict[str, Any], part)
        part_type_obj = part_dict.get("type")
        part_type = part_type_obj if isinstance(part_type_obj, str) else ""
        if part_type in {"text", "input_text"}:
            text_obj = part_dict.get("text")
            if isinstance(text_obj, str):
                return {"type": "input_text", "text": text_obj}
            return None
        if part_type in {"image_url", "input_image"}:
            image_url_obj = part_dict.get("image_url")
            if isinstance(image_url_obj, str):
                return {"type": "input_image", "image_url": image_url_obj}
            if isinstance(image_url_obj, dict):
                image_url_dict = cast(dict[str, Any], image_url_obj)
                url_obj = image_url_dict.get("url")
                if isinstance(url_obj, str):
                    return {"type": "input_image", "image_url": url_obj}
            return None
        return None

    @classmethod
    def _extract_system_instruction(cls, content: Any) -> str | None:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return None
        parts: list[str] = []
        for item_obj in cast(list[Any], content):
            normalized_part = cls._convert_message_content_part(item_obj)
            if normalized_part is None or normalized_part.get("type") != "input_text":
                return None
            text_obj = normalized_part.get("text")
            if not isinstance(text_obj, str):
                return None
            parts.append(text_obj)
        return "\n".join(part for part in parts if part)

    @classmethod
    def _convert_assistant_content_part_to_output(cls, part: Any) -> dict[str, Any] | None:
        if isinstance(part, str):
            return {"type": "output_text", "text": part, "annotations": []}
        if not isinstance(part, dict):
            return None
        part_dict = cast(dict[str, Any], part)
        part_type_obj = part_dict.get("type")
        part_type = part_type_obj if isinstance(part_type_obj, str) else ""
        if part_type in {"text", "output_text", "input_text"}:
            text_obj = part_dict.get("text")
            if isinstance(text_obj, str):
                return {"type": "output_text", "text": text_obj, "annotations": []}
            return None
        if part_type == "refusal":
            refusal_obj = part_dict.get("refusal")
            if isinstance(refusal_obj, str):
                return {"type": "refusal", "refusal": refusal_obj}
        return None

    @classmethod
    def _convert_assistant_content_to_output(cls, content: Any) -> list[dict[str, Any]] | None:
        if content in (None, ""):
            return []
        if isinstance(content, str):
            return [{"type": "output_text", "text": content, "annotations": []}]
        if not isinstance(content, list):
            return None
        output_parts: list[dict[str, Any]] = []
        for part_obj in cast(list[Any], content):
            normalized_part = cls._convert_assistant_content_part_to_output(part_obj)
            if normalized_part is None:
                return None
            output_parts.append(normalized_part)
        return output_parts

    @classmethod
    def _convert_tool_output_content(cls, content: Any) -> str | list[dict[str, Any]] | None:
        if isinstance(content, str):
            return cls._canonicalize_json_string(content)
        if not isinstance(content, list):
            return None
        output_parts: list[dict[str, Any]] = []
        for part_obj in cast(list[Any], content):
            normalized_part = cls._convert_message_content_part(part_obj)
            if normalized_part is None:
                return None
            output_parts.append(normalized_part)
        return output_parts

    @staticmethod
    def _normalize_function_arguments(arguments: Any) -> str | None:
        if isinstance(arguments, str):
            return LiteLLMUpstreamAdapter._canonicalize_json_string(arguments)
        if arguments is None:
            return ""
        try:
            return json.dumps(
                LiteLLMUpstreamAdapter._canonicalize_json_data(arguments),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except TypeError:
            return None

    @staticmethod
    def _canonicalize_json_data(value: Any) -> Any:
        if isinstance(value, dict):
            mapping = cast(dict[str, Any], value)
            return {
                key: LiteLLMUpstreamAdapter._canonicalize_json_data(mapping[key])
                for key in sorted(mapping.keys())
            }
        if isinstance(value, list):
            items = cast(list[Any], value)
            return [LiteLLMUpstreamAdapter._canonicalize_json_data(item) for item in items]
        return value

    @staticmethod
    def _canonicalize_json_string(value: str) -> str:
        stripped = value.strip()
        if not stripped:
            return value
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return value
        try:
            return json.dumps(
                LiteLLMUpstreamAdapter._canonicalize_json_data(parsed),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except TypeError:
            return value

    @staticmethod
    def _new_legacy_function_call_id(index: int) -> str:
        return f"call_legacy_{index}"

    @staticmethod
    def _new_websocket_session_id() -> str:
        return f"ws_{os.urandom(6).hex()}"

    @staticmethod
    def _clone_message_list(messages: list[Any]) -> list[dict[str, Any]]:
        cloned = copy.deepcopy(messages)
        return [cast(dict[str, Any], item) for item in cloned if isinstance(item, dict)]

    @staticmethod
    def _build_messages_history_hash(messages: list[dict[str, Any]]) -> str:
        normalized = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_tools_cache_fingerprint(tools: list[dict[str, Any]] | None) -> str | None:
        if not tools:
            return None
        normalized = json.dumps(
            tools,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _messages_extend_prefix(
        previous_messages: list[dict[str, Any]],
        current_messages: list[dict[str, Any]],
    ) -> bool:
        previous_count = len(previous_messages)
        if previous_count == 0 or len(current_messages) <= previous_count:
            return False
        return current_messages[:previous_count] == previous_messages

    @staticmethod
    def _requires_full_history_websocket_turn(delta_messages: list[dict[str, Any]]) -> bool:
        for message in delta_messages:
            role_obj = message.get("role")
            role = role_obj if isinstance(role_obj, str) else ""
            if role in {"tool", "function"}:
                return True
            if role != "assistant":
                continue
            if isinstance(message.get("tool_calls"), list) or isinstance(
                message.get("function_call"), dict
            ):
                return True
        return False

    @staticmethod
    def _extract_websocket_session_key(request_data: dict[str, Any]) -> tuple[str | None, bool]:
        anchor = LiteLLMUpstreamAdapter._extract_request_conversation_anchor(request_data)
        if anchor is not None:
            return anchor, True
        return None, False

    @staticmethod
    def _sanitize_websocket_metadata_for_upstream(
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if metadata is None:
            return None
        sanitized = {
            key: value
            for key, value in metadata.items()
            if key not in LiteLLMUpstreamAdapter._PROMPT_CACHE_ANCHOR_KEYS
        }
        return sanitized or None

    def _collect_stale_websocket_sessions_locked(
        self, *, now: float
    ) -> list[ResponsesWebSocketSessionState]:
        stale_keys: list[str] = []
        for key, session in self._websocket_sessions.items():
            last_used_at = session.last_used_at or session.connection_started_at
            if last_used_at <= 0:
                stale_keys.append(key)
                continue
            if now - last_used_at >= WEBSOCKET_SESSION_IDLE_TIMEOUT_SECONDS:
                stale_keys.append(key)
        stale_sessions: list[ResponsesWebSocketSessionState] = []
        for key in stale_keys:
            session = self._websocket_sessions.pop(key, None)
            if session is not None:
                stale_sessions.append(session)
        return stale_sessions

    def _register_websocket_session_history(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        if not messages:
            return
        history_hash = self._build_messages_history_hash(messages)
        with self._websocket_sessions_lock:
            session_hashes = self._websocket_session_history_hashes.setdefault(session_id, set())
            if history_hash in session_hashes:
                return
            session_hashes.add(history_hash)
            self._websocket_session_history_index.setdefault(history_hash, {})[
                session_id
            ] = self._clone_message_list(messages)

    def _unregister_websocket_session_history(self, *, session_id: str) -> None:
        with self._websocket_sessions_lock:
            history_hashes = self._websocket_session_history_hashes.pop(session_id, set())
            for history_hash in history_hashes:
                session_snapshots = self._websocket_session_history_index.get(history_hash)
                if session_snapshots is None:
                    continue
                session_snapshots.pop(session_id, None)
                if not session_snapshots:
                    self._websocket_session_history_index.pop(history_hash, None)

    def _prune_websocket_sessions(self) -> None:
        now = time.time()
        with self._websocket_sessions_lock:
            stale_sessions = self._collect_stale_websocket_sessions_locked(now=now)
        for session in stale_sessions:
            self._unregister_websocket_session_history(session_id=session.session_id)
            with session.lock:
                self._close_websocket_session(session)
                self._reset_websocket_session_chain(session)

    def _enforce_websocket_session_capacity(self) -> None:
        sessions_to_close: list[ResponsesWebSocketSessionState] = []
        with self._websocket_sessions_lock:
            overflow = len(self._websocket_sessions) - WEBSOCKET_SESSION_MAX_COUNT
            if overflow <= 0:
                return
            sorted_sessions = sorted(
                self._websocket_sessions.values(),
                key=lambda session: session.last_used_at or session.connection_started_at,
            )
            for session in sorted_sessions[:overflow]:
                removed = self._websocket_sessions.pop(session.session_id, None)
                if removed is not None:
                    sessions_to_close.append(removed)
        for session in sessions_to_close:
            self._unregister_websocket_session_history(session_id=session.session_id)
            with session.lock:
                self._close_websocket_session(session)
                self._reset_websocket_session_chain(session)

    def _close_websocket_session(self, session: ResponsesWebSocketSessionState) -> None:
        connection_context = session.connection_context
        session.connection = None
        session.connection_context = None
        session.client = None
        session.connection_started_at = 0.0
        if connection_context is not None:
            with contextlib.suppress(Exception):
                connection_context.__exit__(None, None, None)

    def _reset_websocket_session_chain(self, session: ResponsesWebSocketSessionState) -> None:
        session.last_response_id = None
        session.last_response_connection_started_at = 0.0
        session.last_request_signature = None
        session.prewarmed_messages_hash = None
        session.conversation_messages = []

    @staticmethod
    def _build_responses_request_signature(payload: dict[str, Any]) -> str:
        normalized = {
            key: value
            for key, value in payload.items()
            if key not in {"input", "previous_response_id", "generate"}
        }
        return hashlib.sha256(
            json.dumps(
                normalized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _open_websocket_session_connection(
        self,
        *,
        session: ResponsesWebSocketSessionState,
        route: UpstreamRoute,
        reconnect_reason: str | None = None,
    ) -> None:
        if reconnect_reason:
            self._log(
                "OpenAI Responses WebSocket 会话重连: "
                f"session={session.session_id} reason={reconnect_reason}"
            )
        self._close_websocket_session(session)
        client = OpenAI(api_key=route.api_key, base_url=route.base_url)
        connect_kwargs: dict[str, Any] = {
            "websocket_connection_options": cast(
                OpenAIWebSocketConnectionOptions,
                OPENAI_RESPONSES_WEBSOCKET_CONNECTION_OPTIONS,
            )
        }
        if session.websocket_extra_query:
            connect_kwargs["extra_query"] = session.websocket_extra_query
        try:
            connection_context = client.responses.connect(**connect_kwargs)
            connection = connection_context.__enter__()
        except Exception as exc:
            status_code = self._extract_websocket_status_code(exc)
            self._log(
                "OpenAI Responses WebSocket 建连失败: "
                f"session={session.session_id} "
                f"url={route.base_url}/responses "
                f"status={status_code if status_code is not None else 'unknown'} "
                f"error={exc}"
            )
            can_retry_with_model_query = (
                session.websocket_extra_query is None
                and self._should_retry_websocket_with_model_query(route=route, exc=exc)
            )
            if not can_retry_with_model_query:
                raise
            session.websocket_extra_query = {"model": route.litellm_model}
            self._log(
                "OpenAI Responses WebSocket 代理兼容重试: "
                f"session={session.session_id} extra_query_model={route.litellm_model}"
            )
            connection_context = client.responses.connect(
                extra_query=session.websocket_extra_query,
                websocket_connection_options=cast(
                    OpenAIWebSocketConnectionOptions,
                    OPENAI_RESPONSES_WEBSOCKET_CONNECTION_OPTIONS,
                ),
            )
            connection = connection_context.__enter__()
        now = time.time()
        session.client = client
        session.connection_context = connection_context
        session.connection = connection
        session.connection_started_at = now
        session.last_used_at = now

    def _ensure_websocket_session_connection(
        self,
        *,
        session: ResponsesWebSocketSessionState,
        route: UpstreamRoute,
    ) -> None:
        connection_age = time.time() - session.connection_started_at
        if session.connection is not None and connection_age < 55 * 60:
            session.last_used_at = time.time()
            return
        reconnect_reason = (
            "connection_limit_guard"
            if session.connection is not None
            else "initial_connect"
        )
        self._open_websocket_session_connection(
            session=session,
            route=route,
            reconnect_reason=reconnect_reason,
        )

    def _find_matching_websocket_session(
        self,
        *,
        current_messages: list[dict[str, Any]],
    ) -> tuple[
        ResponsesWebSocketSessionState | None,
        int,
        int,
        str | None,
        list[dict[str, Any]] | None,
    ]:
        with self._websocket_sessions_lock:
            for prefix_len in range(len(current_messages), 0, -1):
                history_hash = self._build_messages_history_hash(current_messages[:prefix_len])
                candidate_snapshots = self._websocket_session_history_index.get(history_hash, {})
                active_candidates: list[
                    tuple[ResponsesWebSocketSessionState, list[dict[str, Any]]]
                ] = []
                for session_id, snapshot in candidate_snapshots.items():
                    session = self._websocket_sessions.get(session_id)
                    if session is None:
                        continue
                    active_candidates.append((session, self._clone_message_list(snapshot)))
                if len(active_candidates) == 1:
                    session, snapshot = active_candidates[0]
                    return session, prefix_len, 1, "history_index", snapshot
                if len(active_candidates) > 1:
                    return None, prefix_len, len(active_candidates), "history_index", None

        best_session: ResponsesWebSocketSessionState | None = None
        best_prefix_len = -1
        best_match_count = 0
        for session in self._websocket_sessions.values():
            if session.explicit_session_key or not session.conversation_messages:
                continue
            if not self._messages_extend_prefix(session.conversation_messages, current_messages):
                continue
            prefix_len = len(session.conversation_messages)
            if prefix_len > best_prefix_len:
                best_session = session
                best_prefix_len = prefix_len
                best_match_count = 1
                continue
            if prefix_len == best_prefix_len:
                best_match_count += 1
        if best_match_count == 1:
            return best_session, best_prefix_len, best_match_count, "conversation_state", None
        return None, best_prefix_len, best_match_count, "conversation_state", None

    def _get_or_create_websocket_session(
        self,
        *,
        request_data: dict[str, Any],
        current_messages: list[dict[str, Any]],
    ) -> ResponsesWebSocketSessionState:
        self._prune_websocket_sessions()
        explicit_key, explicit = self._extract_websocket_session_key(request_data)
        new_session_created = False
        selection_reason = "unknown"
        matched_prefix_len = 0
        matched_candidates = 0
        matched_source = "none"
        matched_messages: list[dict[str, Any]] | None = None
        with self._websocket_sessions_lock:
            if explicit_key is not None:
                session = self._websocket_sessions.get(explicit_key)
                if session is None:
                    session = ResponsesWebSocketSessionState(
                        session_id=explicit_key,
                        explicit_session_key=explicit,
                    )
                    self._websocket_sessions[explicit_key] = session
                    new_session_created = True
                    selection_reason = "explicit_new"
                else:
                    selection_reason = "explicit_reuse"
                selected_session = session
            else:
                (
                    matched_session,
                    matched_prefix_len,
                    matched_candidates,
                    matched_source,
                    matched_messages,
                ) = (
                    self._find_matching_websocket_session(
                        current_messages=current_messages
                    )
                )
                if matched_session is not None:
                    selected_session = matched_session
                    if matched_messages is not None:
                        selected_session.conversation_messages = matched_messages
                    selection_reason = (
                        "matched_history_prefix"
                        if matched_source == "history_index"
                        else "matched_prefix"
                    )
                else:
                    session = ResponsesWebSocketSessionState(
                        session_id=self._new_websocket_session_id(),
                        explicit_session_key=False,
                    )
                    self._websocket_sessions[session.session_id] = session
                    selected_session = session
                    new_session_created = True
                    selection_reason = (
                        "ambiguous_prefix_new"
                        if matched_candidates > 1
                        else "new_anonymous"
                    )

        if new_session_created:
            self._enforce_websocket_session_capacity()
        selected_session.last_used_at = time.time()
        self._register_websocket_session_history(
            session_id=selected_session.session_id,
            messages=current_messages,
        )
        self._log(
            "OpenAI Responses WebSocket 会话选择: "
            f"session={selected_session.session_id} "
            f"reason={selection_reason} "
            f"explicit={selected_session.explicit_session_key} "
            f"message_count={len(current_messages)} "
            f"matched_prefix_len={max(matched_prefix_len, 0)} "
            f"matched_candidates={matched_candidates} "
            f"matched_source={matched_source}"
        )
        return selected_session

    @classmethod
    def _build_assistant_response_message(
        cls,
        *,
        content: str,
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not content and not tool_calls:
            return None
        message: dict[str, Any] = {"role": "assistant"}
        if content:
            message["content"] = content
        if tool_calls:
            message["tool_calls"] = tool_calls
            message.setdefault("content", "")
        return message

    @classmethod
    def _normalize_strict_schema_node(cls, node: Any) -> Any:
        return cls._normalize_strict_schema_node_with_keys(
            node=node,
            unsupported_schema_keys=OPENAI_UNSUPPORTED_STRICT_SCHEMA_KEYS,
        )

    @classmethod
    def _normalize_strict_schema_node_with_keys(
        cls,
        *,
        node: Any,
        unsupported_schema_keys: frozenset[str],
    ) -> Any:
        if isinstance(node, list):
            return [
                cls._normalize_strict_schema_node_with_keys(
                    node=item,
                    unsupported_schema_keys=unsupported_schema_keys,
                )
                for item in cast(list[Any], node)
            ]
        if not isinstance(node, dict):
            return node

        normalized: dict[str, Any] = {}
        mapping = cast(dict[str, Any], node)
        for key in sorted(mapping.keys()):
            value = mapping[key]
            if key in unsupported_schema_keys:
                continue
            normalized[key] = cls._normalize_strict_schema_node_with_keys(
                node=value,
                unsupported_schema_keys=unsupported_schema_keys,
            )
        type_obj = normalized.get("type")
        is_object_type = type_obj == "object" or (
            isinstance(type_obj, list) and "object" in type_obj
        )
        properties_obj = normalized.get("properties")

        if is_object_type:
            normalized["additionalProperties"] = False
            if isinstance(properties_obj, dict):
                sorted_properties = {
                    key: cast(dict[str, Any], properties_obj)[key]
                    for key in sorted(cast(dict[str, Any], properties_obj).keys())
                }
                normalized["properties"] = sorted_properties
                normalized["required"] = sorted(sorted_properties.keys())

        required_obj = normalized.get("required")
        required_items = cast(list[Any], required_obj) if isinstance(required_obj, list) else None
        if required_items is not None and all(isinstance(item, str) for item in required_items):
            required = cast(list[str], required_items)
            normalized["required"] = sorted(required)

        return normalized

    @classmethod
    def _normalize_strict_function_parameters(
        cls,
        parameters: dict[str, Any] | None,
        *,
        extra_unsupported_schema_keys: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        if parameters is None:
            return {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
        unsupported_schema_keys = frozenset(
            {*OPENAI_UNSUPPORTED_STRICT_SCHEMA_KEYS, *extra_unsupported_schema_keys}
        )
        return cast(
            dict[str, Any],
            cls._normalize_strict_schema_node_with_keys(
                node=parameters,
                unsupported_schema_keys=unsupported_schema_keys,
            ),
        )

    @classmethod
    def _convert_function_tool_definition(
        cls,
        definition: dict[str, Any],
        *,
        extra_unsupported_schema_keys: frozenset[str] = frozenset(),
    ) -> dict[str, Any] | None:
        name_obj = definition.get("name")
        if not isinstance(name_obj, str) or not name_obj.strip():
            return None
        parameters_obj = definition.get("parameters")
        parameters = cast(
            dict[str, object] | None, parameters_obj if isinstance(parameters_obj, dict) else None
        )
        strict_obj = definition.get("strict")
        strict = strict_obj if isinstance(strict_obj, bool) else True
        if strict:
            parameters = cls._normalize_strict_function_parameters(
                cast(dict[str, Any] | None, parameters),
                extra_unsupported_schema_keys=extra_unsupported_schema_keys,
            )
        tool: dict[str, Any] = {
            "type": "function",
            "name": name_obj,
            "parameters": parameters,
            "strict": strict,
        }
        description_obj = definition.get("description")
        if isinstance(description_obj, str) and description_obj.strip():
            tool["description"] = description_obj
        return tool

    @classmethod
    def _convert_chat_tools_to_responses_tools(  # noqa: PLR0911
        cls,
        request_data: dict[str, Any],
        *,
        extra_unsupported_schema_keys: frozenset[str] = frozenset(),
    ) -> list[dict[str, Any]] | None:
        converted_tools: list[dict[str, Any]] = []
        tools_obj = request_data.get("tools")
        if isinstance(tools_obj, list):
            for tool_obj in cast(list[Any], tools_obj):
                if not isinstance(tool_obj, dict):
                    return None
                tool = cast(dict[str, Any], tool_obj)
                tool_type_obj = tool.get("type")
                tool_type = tool_type_obj if isinstance(tool_type_obj, str) else ""
                if tool_type != "function":
                    converted_tools.append(tool)
                    continue
                function_obj = tool.get("function")
                if not isinstance(function_obj, dict):
                    return None
                converted_tool = cls._convert_function_tool_definition(
                    cast(dict[str, Any], function_obj),
                    extra_unsupported_schema_keys=extra_unsupported_schema_keys,
                )
                if converted_tool is None:
                    return None
                converted_tools.append(converted_tool)

        functions_obj = request_data.get("functions")
        if isinstance(functions_obj, list):
            for function_obj in cast(list[Any], functions_obj):
                if not isinstance(function_obj, dict):
                    return None
                converted_tool = cls._convert_function_tool_definition(
                    cast(dict[str, Any], function_obj),
                    extra_unsupported_schema_keys=extra_unsupported_schema_keys,
                )
                if converted_tool is None:
                    return None
                converted_tools.append(converted_tool)

        if not converted_tools:
            return None
        converted_tools.sort(
            key=lambda tool: (
                cast(str, tool.get("type", "")),
                cast(str, tool.get("name", tool.get("server_label", ""))),
                json.dumps(tool, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            )
        )
        return converted_tools

    @staticmethod
    def _convert_chat_tool_choice(request_data: dict[str, Any]) -> Any:
        tool_choice = request_data.get("tool_choice")
        if isinstance(tool_choice, str):
            return tool_choice
        if isinstance(tool_choice, dict):
            choice = cast(dict[str, Any], tool_choice)
            function_obj = choice.get("function")
            if isinstance(function_obj, dict):
                function_dict = cast(dict[str, Any], function_obj)
                name_obj = function_dict.get("name")
                if isinstance(name_obj, str) and name_obj.strip():
                    return {"type": "function", "name": name_obj}
            choice_type_obj = choice.get("type")
            choice_type = choice_type_obj if isinstance(choice_type_obj, str) else ""
            if choice_type in {"auto", "required", "none"}:
                return choice_type

        function_call = request_data.get("function_call")
        if isinstance(function_call, str):
            return function_call
        if isinstance(function_call, dict):
            function_choice = cast(dict[str, Any], function_call)
            name_obj = function_choice.get("name")
            if isinstance(name_obj, str) and name_obj.strip():
                return {"type": "function", "name": name_obj}
        return None

    @classmethod
    def _append_assistant_tool_calls(  # noqa: PLR0911
        cls,
        *,
        message: dict[str, Any],
        input_items: list[dict[str, Any]],
        pending_legacy_call_ids: dict[str, list[str]],
        legacy_call_counter: int,
    ) -> tuple[int, str | None]:
        tool_calls_obj = message.get("tool_calls")
        if isinstance(tool_calls_obj, list):
            for tool_call_obj in cast(list[Any], tool_calls_obj):
                if not isinstance(tool_call_obj, dict):
                    return legacy_call_counter, "assistant.tool_calls 存在非对象项"
                tool_call = cast(dict[str, Any], tool_call_obj)
                tool_type_obj = tool_call.get("type")
                tool_type = tool_type_obj if isinstance(tool_type_obj, str) else ""
                if tool_type and tool_type != "function":
                    return legacy_call_counter, f"暂不支持 tool_calls.type={tool_type}"
                function_obj = tool_call.get("function")
                if not isinstance(function_obj, dict):
                    return legacy_call_counter, "assistant.tool_calls.function 缺失"
                function_call = cast(dict[str, Any], function_obj)
                name_obj = function_call.get("name")
                arguments = cls._normalize_function_arguments(function_call.get("arguments"))
                if not isinstance(name_obj, str) or not name_obj.strip() or arguments is None:
                    return legacy_call_counter, "assistant.tool_calls.function 内容无效"
                call_id_obj = tool_call.get("id")
                call_id = (
                    call_id_obj
                    if isinstance(call_id_obj, str) and call_id_obj.strip()
                    else cls._new_legacy_function_call_id(legacy_call_counter)
                )
                legacy_call_counter += 1
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name_obj,
                        "arguments": arguments,
                        "status": "completed",
                    }
                )
                function_call_item = input_items[-1]
                tool_item_id_obj = tool_call.get("id")
                if (
                    isinstance(tool_item_id_obj, str)
                    and tool_item_id_obj.strip()
                    and tool_item_id_obj.startswith("fc_")
                ):
                    function_call_item["id"] = tool_item_id_obj
            return legacy_call_counter, None

        function_call_obj = message.get("function_call")
        if function_call_obj is None:
            return legacy_call_counter, None
        if not isinstance(function_call_obj, dict):
            return legacy_call_counter, "assistant.function_call 不是对象"
        function_call = cast(dict[str, Any], function_call_obj)
        name_obj = function_call.get("name")
        arguments = cls._normalize_function_arguments(function_call.get("arguments"))
        if not isinstance(name_obj, str) or not name_obj.strip() or arguments is None:
            return legacy_call_counter, "assistant.function_call 内容无效"
        call_id = cls._new_legacy_function_call_id(legacy_call_counter)
        legacy_call_counter += 1
        pending_legacy_call_ids.setdefault(name_obj, []).append(call_id)
        input_items.append(
            {
                "type": "function_call",
                "id": call_id,
                "call_id": call_id,
                "name": name_obj,
                "arguments": arguments,
                "status": "completed",
            }
        )
        return legacy_call_counter, None

    @classmethod
    def _append_assistant_message_items(
        cls,
        *,
        message: dict[str, Any],
        input_items: list[dict[str, Any]],
        pending_legacy_call_ids: dict[str, list[str]],
        legacy_call_counter: int,
    ) -> tuple[int, str | None]:
        output_content = cls._convert_assistant_content_to_output(message.get("content"))
        if output_content is None:
            return legacy_call_counter, "assistant 历史消息包含无法转换的内容"
        if output_content:
            input_items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": output_content,
                    "id": f"msg_assistant_{legacy_call_counter}",
                }
            )
        return cls._append_assistant_tool_calls(
            message=message,
            input_items=input_items,
            pending_legacy_call_ids=pending_legacy_call_ids,
            legacy_call_counter=legacy_call_counter,
        )

    @classmethod
    def _resolve_legacy_function_output_call_id(
        cls,
        *,
        message: dict[str, Any],
        pending_legacy_call_ids: dict[str, list[str]],
    ) -> str | None:
        tool_call_id_obj = message.get("tool_call_id")
        if isinstance(tool_call_id_obj, str) and tool_call_id_obj.strip():
            return tool_call_id_obj
        name_obj = message.get("name")
        if not isinstance(name_obj, str) or not name_obj.strip():
            return None
        pending_ids = pending_legacy_call_ids.get(name_obj)
        if not pending_ids:
            return None
        call_id = pending_ids.pop(0)
        if not pending_ids:
            pending_legacy_call_ids.pop(name_obj, None)
        return call_id

    @classmethod
    def _convert_chat_messages_to_responses_input(  # noqa: PLR0911, PLR0912
        cls, messages: list[Any]
    ) -> tuple[list[dict[str, Any]], str | None, str | None]:
        input_items: list[dict[str, Any]] = []
        instructions_parts: list[str] = []
        pending_legacy_call_ids: dict[str, list[str]] = {}
        legacy_call_counter = 0
        for raw_message in messages:
            if not isinstance(raw_message, dict):
                return [], None, "messages 中存在非对象项"
            message = cast(dict[str, Any], raw_message)
            role_obj = message.get("role")
            role = role_obj if isinstance(role_obj, str) else ""
            if role == "system":
                instruction = cls._extract_system_instruction(message.get("content"))
                if instruction is None:
                    return [], None, "system 消息包含无法转换的内容"
                if instruction:
                    instructions_parts.append(instruction)
                continue
            if role in {"user", "developer"}:
                content_obj = message.get("content")
                normalized_content: list[dict[str, Any]] = []
                if isinstance(content_obj, list):
                    for part_obj in cast(list[Any], content_obj):
                        normalized_part = cls._convert_message_content_part(part_obj)
                        if normalized_part is None:
                            return [], None, "消息内容存在无法转换的多模态片段"
                        normalized_content.append(normalized_part)
                elif isinstance(content_obj, str):
                    normalized_content.append({"type": "input_text", "text": content_obj})
                else:
                    return [], None, "消息内容格式暂不支持 WebSocket 桥接"
                input_items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": normalized_content,
                    }
                )
                continue
            if role == "assistant":
                legacy_call_counter, error = cls._append_assistant_message_items(
                    message=message,
                    input_items=input_items,
                    pending_legacy_call_ids=pending_legacy_call_ids,
                    legacy_call_counter=legacy_call_counter,
                )
                if error is not None:
                    return [], None, error
                continue
            if role in {"tool", "function"}:
                call_id = cls._resolve_legacy_function_output_call_id(
                    message=message,
                    pending_legacy_call_ids=pending_legacy_call_ids,
                )
                if call_id is None:
                    return [], None, f"无法为 {role} 消息找到对应的 function call"
                output = cls._convert_tool_output_content(message.get("content"))
                if output is None:
                    return [], None, f"{role} 消息输出内容无法转换"
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output,
                    }
                )
                continue
            return [], None, f"暂不支持 role={role or '<unknown>'} 的 WebSocket 桥接"
        instructions = "\n\n".join(part for part in instructions_parts if part) or None
        return input_items, instructions, None

    @classmethod
    def _build_openai_responses_websocket_request(  # noqa: PLR0912
        cls,
        *,
        route: UpstreamRoute,
        request_data: dict[str, Any],
        extra_unsupported_schema_keys: frozenset[str] = frozenset(),
        stripped_top_level_keys: frozenset[str] = frozenset(),
    ) -> tuple[dict[str, Any] | None, str | None]:
        messages_obj = request_data.get("messages")
        if not isinstance(messages_obj, list):
            return None, "缺少可转换的 messages 数组"
        input_items, instructions, error = cls._convert_chat_messages_to_responses_input(
            cast(list[Any], messages_obj)
        )
        if error is not None:
            return None, error

        payload: dict[str, Any] = {
            "model": route.litellm_model,
            "input": input_items,
        }
        if instructions:
            payload["instructions"] = instructions

        converted_tools = cls._convert_chat_tools_to_responses_tools(
            request_data,
            extra_unsupported_schema_keys=extra_unsupported_schema_keys,
        )
        if request_data.get("tools") is not None or request_data.get("functions") is not None:
            if converted_tools is None:
                return None, "顶层 tools/functions 无法转换为 Responses tools"
            payload["tools"] = converted_tools

        converted_tool_choice = cls._convert_chat_tool_choice(request_data)
        if converted_tool_choice is not None:
            payload["tool_choice"] = converted_tool_choice

        passthrough_keys = (
            "parallel_tool_calls",
            "prompt_cache_retention",
            "service_tier",
            "store",
            "temperature",
            "top_p",
            "truncation",
            "user",
        )
        for key in passthrough_keys:
            if key in request_data:
                payload[key] = request_data[key]

        metadata_obj = request_data.get("metadata")
        metadata = (
            cast(dict[str, Any], metadata_obj) if isinstance(metadata_obj, dict) else None
        )
        sanitized_metadata = cls._sanitize_websocket_metadata_for_upstream(metadata)
        if sanitized_metadata is not None:
            payload["metadata"] = sanitized_metadata

        prompt_cache_key = cls._build_request_prompt_cache_key(
            route=route,
            request_data=request_data,
        )
        if prompt_cache_key is not None:
            payload["prompt_cache_key"] = prompt_cache_key

        if "reasoning_effort" in request_data:
            payload["reasoning"] = {"effort": request_data["reasoning_effort"]}

        max_tokens = request_data.get("max_completion_tokens", request_data.get("max_tokens"))
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

        for key in stripped_top_level_keys:
            payload.pop(key, None)

        return payload, None

    @staticmethod
    def _build_openai_chat_delta_chunk(
        *,
        response_id: str,
        created: int,
        model: str,
        delta: dict[str, Any],
        finish_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "id": response_id,
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }

    def _build_openai_responses_websocket_turn_payload(  # noqa: PLR0912, PLR0913
        self,
        *,
        route: UpstreamRoute,
        request_data: dict[str, Any],
        session: ResponsesWebSocketSessionState,
        current_messages: list[dict[str, Any]],
        extra_unsupported_schema_keys: frozenset[str] = frozenset(),
        stripped_top_level_keys: frozenset[str] = frozenset(),
    ) -> tuple[dict[str, Any] | None, bool, str | None]:
        continue_with_previous = False
        turn_request_data = copy.deepcopy(request_data)
        previous_messages = session.conversation_messages
        current_history_hash = self._build_messages_history_hash(current_messages)
        if (
            session.last_response_id
            and session.last_response_connection_started_at > 0
            and session.last_response_connection_started_at == session.connection_started_at
            and previous_messages
            and (
                self._messages_extend_prefix(previous_messages, current_messages)
                or (
                    session.prewarmed_messages_hash == current_history_hash
                    and previous_messages == current_messages
                )
            )
        ):
            delta_messages = current_messages[len(previous_messages) :]
            if delta_messages and not self._requires_full_history_websocket_turn(delta_messages):
                turn_request_data["messages"] = delta_messages
                continue_with_previous = True
            elif (
                not delta_messages
                and session.prewarmed_messages_hash == current_history_hash
            ):
                turn_request_data["messages"] = []
                continue_with_previous = True
                session.prewarmed_messages_hash = None
            elif delta_messages:
                self._log(
                    "OpenAI Responses WebSocket 工具续链回退为全量历史: "
                    f"session={session.session_id}"
                )
            else:
                self._log(
                    "OpenAI Responses WebSocket 检测到重复历史，改为新链路: "
                    f"session={session.session_id}"
                )

        payload, error = self._build_openai_responses_websocket_request(
            route=route,
            request_data=turn_request_data,
            extra_unsupported_schema_keys=extra_unsupported_schema_keys,
            stripped_top_level_keys=stripped_top_level_keys,
        )
        if payload is None:
            return None, False, error

        prompt_cache_key = self._build_request_prompt_cache_key(
            route=route,
            request_data=turn_request_data,
            session=session,
        )
        if prompt_cache_key is not None:
            payload["prompt_cache_key"] = prompt_cache_key
        request_signature = self._build_responses_request_signature(payload)
        tools_fingerprint = self._build_tools_cache_fingerprint(
            cast(list[dict[str, Any]] | None, payload.get("tools"))
        )
        if tools_fingerprint is not None:
            previous_tools_fingerprint = session.last_tools_fingerprint
            session.last_tools_fingerprint = tools_fingerprint
            change_state = (
                "unchanged"
                if previous_tools_fingerprint == tools_fingerprint
                else "changed"
                if previous_tools_fingerprint
                else "initial"
            )
            self._log(
                "OpenAI Responses WebSocket 工具指纹: "
                f"session={session.session_id} "
                f"fingerprint={tools_fingerprint} "
                f"state={change_state}"
            )

        if (
            continue_with_previous
            and session.last_response_id
            and session.last_request_signature is not None
            and session.last_request_signature != request_signature
        ):
            continue_with_previous = False
            turn_request_data = copy.deepcopy(request_data)
            payload, error = self._build_openai_responses_websocket_request(
                route=route,
                request_data=turn_request_data,
                extra_unsupported_schema_keys=extra_unsupported_schema_keys,
                stripped_top_level_keys=stripped_top_level_keys,
            )
            if payload is None:
                return None, False, error
            if prompt_cache_key is not None:
                payload["prompt_cache_key"] = prompt_cache_key
            request_signature = self._build_responses_request_signature(payload)
            self._log(
                "OpenAI Responses WebSocket 续链签名不一致，改为新链路: "
                f"session={session.session_id}"
            )

        if continue_with_previous and session.last_response_id:
            payload["previous_response_id"] = session.last_response_id
            self._log(
                "OpenAI Responses WebSocket 续链: "
                f"session={session.session_id} "
                f"previous_response_id={session.last_response_id}"
            )
        elif session.last_response_id:
            self._log(
                "OpenAI Responses WebSocket 改为新链路: "
                f"session={session.session_id} reason=history_reset"
            )
        session.last_request_signature = request_signature
        return payload, continue_with_previous, None

    def _should_prewarm_websocket_turn(
        self,
        *,
        route: UpstreamRoute,
        session: ResponsesWebSocketSessionState,
        payload: dict[str, Any],
        current_messages: list[dict[str, Any]],
    ) -> bool:
        if not route.prompt_cache_enabled or not route.prompt_cache_key:
            return False
        if session.last_response_id is not None:
            return False
        if session.conversation_messages:
            return False
        if not current_messages:
            return False
        if not payload.get("input"):
            return False
        return (
            session.connection_started_at > 0
            and session.last_response_connection_started_at <= 0
        )

    def _prewarm_websocket_session_for_turn(
        self,
        *,
        route: UpstreamRoute,
        session: ResponsesWebSocketSessionState,
        payload: dict[str, Any],
        current_messages: list[dict[str, Any]],
    ) -> None:
        if not self._websocket_response_create_supports_generate(session):
            self._log(
                "OpenAI Responses WebSocket 跳过预热: "
                f"session={session.session_id} reason=sdk_missing_generate"
            )
            return
        prewarm_payload = copy.deepcopy(payload)
        prewarm_payload["generate"] = False
        self._log(
            "OpenAI Responses WebSocket 预热: "
            f"session={session.session_id} "
            f"prompt_cache_key={prewarm_payload.get('prompt_cache_key', '')}"
        )
        for _ in self._stream_openai_responses_websocket_turn(
            route=route,
            session=session,
            payload=prewarm_payload,
            current_messages=current_messages,
        ):
            pass
        session.prewarmed_messages_hash = self._build_messages_history_hash(current_messages)

    @staticmethod
    def _websocket_response_create_supports_generate(
        session: ResponsesWebSocketSessionState,
    ) -> bool:
        connection = session.connection
        if connection is None:
            return False
        response_resource = getattr(connection, "response", None)
        if response_resource is None:
            return False
        create = getattr(response_resource, "create", None)
        if not callable(create):
            return False
        try:
            signature = inspect.signature(create)
        except (TypeError, ValueError):
            return False
        parameters = signature.parameters.values()
        if any(parameter.name == "generate" for parameter in parameters):
            return True
        return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)

    @staticmethod
    def _extract_unsupported_schema_keyword(error: Exception) -> str | None:
        message = str(error)
        for pattern in (
            r"'([^']+)' is not permitted",
            r'"([^"]+)" is not permitted',
        ):
            match = re.search(pattern, message)
            if match is None:
                continue
            keyword = match.group(1).strip()
            if keyword:
                return keyword
        return None

    @staticmethod
    def _extract_unsupported_top_level_param(error: Exception) -> str | None:
        message = str(error)
        for pattern in (
            r"Unsupported parameter:\s*'([^']+)'",
            r'Unsupported parameter:\s*"([^"]+)"',
            r"Unsupported parameter:\s*([A-Za-z0-9_\.]+)",
        ):
            match = re.search(pattern, message)
            if match is None:
                continue
            param = match.group(1).strip()
            if param:
                return param.split(".", 1)[0]
        return None

    def _stream_openai_responses_websocket_turn(  # noqa: PLR0912, PLR0915
        self,
        *,
        route: UpstreamRoute,
        session: ResponsesWebSocketSessionState,
        payload: dict[str, Any],
        current_messages: list[dict[str, Any]],
        create_request: bool = True,
    ) -> Any:
        connection = session.connection
        if connection is None:
            raise RuntimeError("OpenAI Responses WebSocket 连接未建立")

        response_id = "chatcmpl_websocket_pending"
        created = 0
        function_calls: dict[str, ResponsesWebSocketFunctionCallState] = {}
        saw_function_call = False
        saw_any_event = False
        assistant_content_parts: list[str] = []
        assistant_tool_calls: list[dict[str, Any]] = []

        def now_ts() -> int:
            return int(time.time())

        if create_request:
            connection.response.create(**payload)
        for event in connection:
            saw_any_event = True
            event_payload = _coerce_model_dump(event)
            if event_payload is None:
                continue
            event_type_obj = event_payload.get("type")
            event_type = event_type_obj if isinstance(event_type_obj, str) else ""

            if event_type == "response.created":
                response_obj = event_payload.get("response")
                if isinstance(response_obj, dict):
                    response = cast(dict[str, Any], response_obj)
                    response_id_obj = response.get("id")
                    if isinstance(response_id_obj, str) and response_id_obj:
                        response_id = response_id_obj
                    created_obj = response.get("created_at")
                    if isinstance(created_obj, int):
                        created = created_obj
                if created <= 0:
                    created = now_ts()
                continue

            if event_type in {"response.output_text.delta", "response.refusal.delta"}:
                delta_obj = event_payload.get("delta")
                if isinstance(delta_obj, str) and delta_obj:
                    assistant_content_parts.append(delta_obj)
                    yield self._build_openai_chat_delta_chunk(
                        response_id=response_id,
                        created=created or now_ts(),
                        model=route.litellm_model,
                        delta={"content": delta_obj},
                    )
                continue

            if event_type in {
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
            }:
                delta_obj = event_payload.get("delta")
                if isinstance(delta_obj, str) and delta_obj:
                    yield self._build_openai_chat_delta_chunk(
                        response_id=response_id,
                        created=created or now_ts(),
                        model=route.litellm_model,
                        delta={"reasoning_content": delta_obj},
                    )
                continue

            if event_type == "response.output_item.added":
                item_obj = event_payload.get("item")
                item = cast(dict[str, Any], item_obj) if isinstance(item_obj, dict) else None
                if not item or item.get("type") != "function_call":
                    continue
                item_id_obj = item.get("id")
                item_id = item_id_obj if isinstance(item_id_obj, str) else ""
                call_id_obj = item.get("call_id")
                call_id = call_id_obj if isinstance(call_id_obj, str) and call_id_obj else item_id
                name_obj = item.get("name")
                name = name_obj if isinstance(name_obj, str) else ""
                function_calls[item_id] = ResponsesWebSocketFunctionCallState(
                    index=len(function_calls),
                    call_id=call_id,
                    name=name,
                )
                assistant_tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }
                )
                saw_function_call = True
                yield self._build_openai_chat_delta_chunk(
                    response_id=response_id,
                    created=created or now_ts(),
                    model=route.litellm_model,
                    delta={"tool_calls": [copy.deepcopy(assistant_tool_calls[-1])]},
                )
                continue

            if event_type in {
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            }:
                item_id_obj = event_payload.get("item_id")
                item_id = item_id_obj if isinstance(item_id_obj, str) else ""
                payload_key = "delta" if event_type.endswith(".delta") else "arguments"
                delta_obj = event_payload.get(payload_key)
                if not item_id or not isinstance(delta_obj, str) or not delta_obj:
                    continue
                function_call = function_calls.get(item_id)
                if function_call is None:
                    continue
                if event_type.endswith(".done") and function_call.arguments_streamed:
                    continue
                function_call.arguments_streamed = True
                for tool_call in assistant_tool_calls:
                    if tool_call.get("id") == function_call.call_id:
                        function_obj = tool_call.get("function")
                        if isinstance(function_obj, dict):
                            function_data = cast(dict[str, Any], function_obj)
                            previous_arguments = function_data.get("arguments")
                            arguments = (
                                previous_arguments
                                if isinstance(previous_arguments, str)
                                else ""
                            )
                            function_data["arguments"] = arguments + delta_obj
                        break
                yield self._build_openai_chat_delta_chunk(
                    response_id=response_id,
                    created=created or now_ts(),
                    model=route.litellm_model,
                    delta={
                        "tool_calls": [
                            {
                                "index": function_call.index,
                                "id": function_call.call_id,
                                "type": "function",
                                "function": {"arguments": delta_obj},
                            }
                        ]
                    },
                )
                continue

            if event_type == "response.completed":
                assistant_message = self._build_assistant_response_message(
                    content="".join(assistant_content_parts),
                    tool_calls=assistant_tool_calls,
                )
                updated_messages = self._clone_message_list(current_messages)
                if assistant_message is not None:
                    updated_messages.append(assistant_message)
                session.last_response_id = response_id
                session.last_response_connection_started_at = session.connection_started_at
                session.conversation_messages = updated_messages
                session.last_used_at = time.time()
                self._register_websocket_session_history(
                    session_id=session.session_id,
                    messages=updated_messages,
                )
                yield self._build_openai_chat_delta_chunk(
                    response_id=response_id,
                    created=created or now_ts(),
                    model=route.litellm_model,
                    delta={},
                    finish_reason="tool_calls" if saw_function_call else "stop",
                )
                return

            if event_type in {"response.failed", "error"}:
                status_obj = event_payload.get("status")
                status = status_obj if isinstance(status_obj, int) else None
                error_obj = event_payload.get("error")
                error_payload = (
                    cast(dict[str, Any], error_obj) if isinstance(error_obj, dict) else {}
                )
                message = ""
                for key in ("message", "detail"):
                    message_obj = error_payload.get(key)
                    if isinstance(message_obj, str) and message_obj.strip():
                        message = message_obj.strip()
                        break
                if not message:
                    message = f"OpenAI Responses WebSocket 返回错误事件: {event_type}"
                code_obj = error_payload.get("code")
                code = code_obj.strip() if isinstance(code_obj, str) and code_obj.strip() else None
                raise ResponsesWebSocketSessionError(message, code=code, status=status)
        if not saw_any_event:
            raise ResponsesWebSocketSessionError(
                "OpenAI Responses WebSocket 在没有任何事件输出前关闭",
                code="websocket_closed_before_events",
            )
        raise ResponsesWebSocketSessionError(
            "OpenAI Responses WebSocket 在响应完成前关闭",
            code="websocket_closed_mid_stream",
        )

    def _create_openai_responses_websocket_stream(  # noqa: PLR0915
        self,
        *,
        route: UpstreamRoute,
        request_data: dict[str, Any],
    ) -> Any:
        messages_obj = request_data.get("messages")
        if not isinstance(messages_obj, list):
            raise ValueError("缺少可转换的 messages 数组")
        current_messages = self._clone_message_list(cast(list[Any], messages_obj))
        session = self._get_or_create_websocket_session(
            request_data=request_data,
            current_messages=current_messages,
        )

        def generate() -> Any:  # noqa: PLR0912, PLR0915
            dynamic_unsupported_schema_keys: set[str] = set()
            dynamic_stripped_top_level_keys: set[str] = set()
            schema_self_heal_attempts = 0
            unsupported_param_self_heal_attempts = 0
            recoverable_session_attempts = 0
            empty_close_sampling_retry_attempts = 0
            with session.lock:
                while True:
                    emitted_output = False
                    self._ensure_websocket_session_connection(session=session, route=route)
                    payload, _continued, error = (
                        self._build_openai_responses_websocket_turn_payload(
                            route=route,
                            request_data=request_data,
                            session=session,
                            current_messages=current_messages,
                            extra_unsupported_schema_keys=frozenset(
                                dynamic_unsupported_schema_keys
                            ),
                            stripped_top_level_keys=frozenset(
                                dynamic_stripped_top_level_keys
                            ),
                        )
                    )
                    if payload is None:
                        raise ValueError(error or "无法构造 Responses WebSocket 请求")

                    try:
                        if self._should_prewarm_websocket_turn(
                            route=route,
                            session=session,
                            payload=payload,
                            current_messages=current_messages,
                        ):
                            self._prewarm_websocket_session_for_turn(
                                route=route,
                                session=session,
                                payload=payload,
                                current_messages=current_messages,
                            )
                            payload, _continued, error = (
                                self._build_openai_responses_websocket_turn_payload(
                                    route=route,
                                    request_data=request_data,
                                    session=session,
                                    current_messages=current_messages,
                                    extra_unsupported_schema_keys=frozenset(
                                        dynamic_unsupported_schema_keys
                                    ),
                                    stripped_top_level_keys=frozenset(
                                        dynamic_stripped_top_level_keys
                                    ),
                                )
                            )
                            if payload is None:
                                raise ValueError(error or "无法构造 Responses WebSocket 请求")
                        for chunk in self._stream_openai_responses_websocket_turn(
                            route=route,
                            session=session,
                            payload=payload,
                            current_messages=current_messages,
                        ):
                            emitted_output = True
                            yield chunk
                        return
                    except ResponsesWebSocketSessionError as exc:
                        session.last_used_at = time.time()
                        unsupported_top_level_param = self._extract_unsupported_top_level_param(exc)
                        can_self_heal_param = (
                            not emitted_output
                            and unsupported_top_level_param is not None
                            and unsupported_top_level_param not in dynamic_stripped_top_level_keys
                            and unsupported_param_self_heal_attempts
                            < OPENAI_UNSUPPORTED_PARAM_SELF_HEAL_MAX_ATTEMPTS
                        )
                        if can_self_heal_param:
                            assert unsupported_top_level_param is not None
                            dynamic_stripped_top_level_keys.add(unsupported_top_level_param)
                            unsupported_param_self_heal_attempts += 1
                            self._log(
                                "OpenAI Responses WebSocket 参数自愈: "
                                f"session={session.session_id} "
                                f"drop_param={unsupported_top_level_param}"
                            )
                            self._close_websocket_session(session)
                            self._open_websocket_session_connection(
                                session=session,
                                route=route,
                                reconnect_reason=f"unsupported_param:{unsupported_top_level_param}",
                            )
                            continue
                        recoverable_codes = {
                            "previous_response_not_found",
                            "websocket_connection_limit_reached",
                        }
                        can_retry_without_sampling = (
                            not emitted_output
                            and exc.code == "websocket_closed_before_events"
                            and empty_close_sampling_retry_attempts < 1
                            and self._should_retry_without_sampling(
                                request_data=request_data,
                                stripped_top_level_keys=dynamic_stripped_top_level_keys,
                            )
                        )
                        if can_retry_without_sampling:
                            empty_close_sampling_retry_attempts += 1
                            sampling_keys = self._get_sampling_retry_keys(
                                request_data=request_data,
                                stripped_top_level_keys=dynamic_stripped_top_level_keys,
                            )
                            dynamic_stripped_top_level_keys.update(sampling_keys)
                            self._log(
                                "OpenAI Responses WebSocket 空响应关闭，移除采样参数重试: "
                                f"session={session.session_id} "
                                f"drop_keys={','.join(sorted(sampling_keys))} "
                                f"params={self._build_sampling_param_log(request_data)}"
                            )
                            self._close_websocket_session(session)
                            self._open_websocket_session_connection(
                                session=session,
                                route=route,
                                reconnect_reason="empty_close_sampling_retry",
                            )
                            continue
                        if (
                            emitted_output
                            or exc.code not in recoverable_codes
                            or recoverable_session_attempts >= 1
                        ):
                            raise
                        recoverable_session_attempts += 1
                        self._log(
                            "OpenAI Responses WebSocket 会话恢复: "
                            f"session={session.session_id} code={exc.code}"
                        )
                        self._reset_websocket_session_chain(session)
                        self._open_websocket_session_connection(
                            session=session,
                            route=route,
                            reconnect_reason=exc.code,
                        )
                        continue
                    except Exception as exc:
                        session.last_used_at = time.time()
                        unsupported_schema_key = self._extract_unsupported_schema_keyword(exc)
                        can_self_heal_schema = (
                            not emitted_output
                            and unsupported_schema_key is not None
                            and unsupported_schema_key
                            not in OPENAI_UNSUPPORTED_STRICT_SCHEMA_KEYS
                            and unsupported_schema_key not in dynamic_unsupported_schema_keys
                            and schema_self_heal_attempts
                            < OPENAI_STRICT_SCHEMA_SELF_HEAL_MAX_ATTEMPTS
                        )
                        if can_self_heal_schema:
                            assert unsupported_schema_key is not None
                            dynamic_unsupported_schema_keys.add(unsupported_schema_key)
                            schema_self_heal_attempts += 1
                            self._log(
                                "OpenAI Responses WebSocket strict schema 自愈: "
                                f"session={session.session_id} "
                                f"strip_key={unsupported_schema_key}"
                            )
                            self._close_websocket_session(session)
                            self._open_websocket_session_connection(
                                session=session,
                                route=route,
                                reconnect_reason=f"strict_schema:{unsupported_schema_key}",
                            )
                            continue
                        self._close_websocket_session(session)
                        self._reset_websocket_session_chain(session)
                        raise

        return generate()

    @staticmethod
    def _extract_websocket_status_code(exc: Exception) -> int | None:
        for attr_name in ("status_code", "status"):
            value = getattr(exc, attr_name, None)
            if isinstance(value, int):
                return value
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        match = re.search(r"\b(4\d{2}|5\d{2})\b", str(exc))
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _is_official_openai_route(route: UpstreamRoute) -> bool:
        try:
            parsed = urlparse(route.base_url)
        except Exception:
            return False
        host = (parsed.hostname or "").lower()
        return host in {"api.openai.com", "api.openai.azure.com"}

    def _should_retry_websocket_with_model_query(
        self,
        *,
        route: UpstreamRoute,
        exc: Exception,
    ) -> bool:
        if self._is_official_openai_route(route):
            return False
        status_code = self._extract_websocket_status_code(exc)
        message = str(exc).lower()
        if status_code == HTTP_FORBIDDEN:
            return True
        return "handshake" in message and str(HTTP_FORBIDDEN) in message

    @staticmethod
    def _get_sampling_retry_keys(
        *,
        request_data: dict[str, Any],
        stripped_top_level_keys: set[str],
    ) -> set[str]:
        retryable_keys = {"temperature", "top_p", "stream_options"}
        result: set[str] = set()
        for key in retryable_keys:
            if key in stripped_top_level_keys:
                continue
            if key in request_data:
                result.add(key)
        return result

    def _should_retry_without_sampling(
        self,
        *,
        request_data: dict[str, Any],
        stripped_top_level_keys: set[str],
    ) -> bool:
        return bool(
            self._get_sampling_retry_keys(
                request_data=request_data,
                stripped_top_level_keys=stripped_top_level_keys,
            )
        )

    @staticmethod
    def _build_sampling_param_log(request_data: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in ("temperature", "top_p", "stream_options"):
            if key in request_data:
                parts.append(f"{key}={request_data[key]}")
        return ", ".join(parts) if parts else "none"

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

    def _create_http_chat_completion(
        self,
        *,
        route: UpstreamRoute,
        request_data: dict[str, Any],
    ) -> Any:
        if route.provider == OPENAI_RESPONSE_PROVIDER:
            return self._create_openai_responses_http_stream(
                route=route,
                request_data=request_data,
            )
        call_kwargs = self._normalize_provider_chat_request(
            route=route,
            request_data=request_data,
        )
        call_kwargs["model"] = self._resolve_chat_completion_model(route)
        call_kwargs["base_url"] = route.base_url
        call_kwargs["api_key"] = route.api_key
        call_kwargs["ssl_verify"] = self._build_request_ssl_verify(self._disable_ssl_strict_mode)
        return _create_litellm_completion(**call_kwargs)

    def _create_openai_responses_http_stream(
        self,
        *,
        route: UpstreamRoute,
        request_data: dict[str, Any],
    ) -> Any:
        messages_obj = request_data.get("messages")
        if not isinstance(messages_obj, list):
            raise ValueError("缺少可转换的 messages 数组")
        current_messages = self._clone_message_list(cast(list[Any], messages_obj))
        dynamic_unsupported_schema_keys: set[str] = set()
        dynamic_stripped_top_level_keys: set[str] = set()
        schema_self_heal_attempts = 0
        unsupported_param_self_heal_attempts = 0
        client = OpenAI(api_key=route.api_key, base_url=route.base_url)
        raw_stream: Any = None
        payload: dict[str, Any] | None = None
        while True:
            payload, error = self._build_openai_responses_websocket_request(
                route=route,
                request_data=request_data,
                extra_unsupported_schema_keys=frozenset(dynamic_unsupported_schema_keys),
                stripped_top_level_keys=frozenset(dynamic_stripped_top_level_keys),
            )
            if payload is None:
                raise ValueError(error or "无法构造 Responses HTTP 请求")
            try:
                raw_stream = client.responses.create(stream=True, **payload)
                break
            except Exception as exc:
                unsupported_top_level_param = self._extract_unsupported_top_level_param(exc)
                can_self_heal_param = (
                    unsupported_top_level_param is not None
                    and unsupported_top_level_param not in dynamic_stripped_top_level_keys
                    and unsupported_param_self_heal_attempts
                    < OPENAI_UNSUPPORTED_PARAM_SELF_HEAL_MAX_ATTEMPTS
                )
                if can_self_heal_param:
                    assert unsupported_top_level_param is not None
                    dynamic_stripped_top_level_keys.add(unsupported_top_level_param)
                    unsupported_param_self_heal_attempts += 1
                    self._log(
                        "OpenAI Responses HTTP 参数自愈: "
                        f"drop_param={unsupported_top_level_param}"
                    )
                    continue
                unsupported_schema_key = self._extract_unsupported_schema_keyword(exc)
                can_self_heal_schema = (
                    unsupported_schema_key is not None
                    and unsupported_schema_key not in OPENAI_UNSUPPORTED_STRICT_SCHEMA_KEYS
                    and unsupported_schema_key not in dynamic_unsupported_schema_keys
                    and schema_self_heal_attempts < OPENAI_STRICT_SCHEMA_SELF_HEAL_MAX_ATTEMPTS
                )
                if can_self_heal_schema:
                    assert unsupported_schema_key is not None
                    dynamic_unsupported_schema_keys.add(unsupported_schema_key)
                    schema_self_heal_attempts += 1
                    self._log(
                        "OpenAI Responses HTTP strict schema 自愈: "
                        f"strip_key={unsupported_schema_key}"
                    )
                    continue
                raise

        class _NoOpResponseResource:
            def create(self, **kwargs: Any) -> None:
                return None

        class _HTTPConnection:
            def __init__(self, stream: Any) -> None:
                self.response = _NoOpResponseResource()
                self._stream = stream

            def __iter__(self) -> Any:
                return iter(self._stream)

        temp_session = ResponsesWebSocketSessionState(
            session_id="http_fallback",
            explicit_session_key=True,
        )
        temp_session.connection = _HTTPConnection(raw_stream)

        def generate() -> Any:
            try:
                yield from self._stream_openai_responses_websocket_turn(
                    route=route,
                    session=temp_session,
                    payload=payload,
                    current_messages=current_messages,
                    create_request=False,
                )
            finally:
                close = getattr(raw_stream, "close", None)
                if callable(close):
                    with contextlib.suppress(Exception):
                        close()

        return generate()

    def _create_chat_completion_with_websocket_fallback(
        self,
        *,
        route: UpstreamRoute,
        request_data: dict[str, Any],
    ) -> Any:
        def generate() -> Any:
            websocket_started = False
            try:
                websocket_stream = self._create_openai_responses_websocket_stream(
                    route=route,
                    request_data=request_data,
                )
                for chunk in websocket_stream:
                    websocket_started = True
                    yield chunk
                return
            except ValueError as exc:
                self._mark_websocket_route_fallback(route, str(exc))
                self._log(f"OpenAI Responses WebSocket 已回退到 HTTP: {exc}")
            except Exception as exc:  # noqa: BLE001
                if websocket_started:
                    raise
                self._mark_websocket_route_fallback(route, str(exc))
                self._log(
                    f"OpenAI Responses WebSocket 模式出问题了，没连接上，已回退到普通 HTTP: {exc}"
                )

            http_stream = self._create_http_chat_completion(
                route=route,
                request_data=request_data,
            )
            if hasattr(http_stream, "__iter__") and not isinstance(
                http_stream, (dict, str, bytes, bytearray)
            ):
                for chunk in http_stream:
                    yield chunk
                return
            raise RuntimeError("WebSocket 回退到 HTTP 后未拿到流式响应")

        return generate()

    def create_chat_completion(
        self,
        *,
        route: UpstreamRoute,
        request_data: dict[str, Any],
    ) -> Any:
        if route.websocket_mode_enabled and self._is_websocket_route_fallback(route):
            self._log("OpenAI Responses WebSocket 已锁定回退，当前请求直接走 HTTP")
            return self._create_http_chat_completion(
                route=route,
                request_data=request_data,
            )
        use_websocket_mode, reason = self._supports_openai_responses_websocket(route, request_data)
        if use_websocket_mode:
            self._log("流式请求优先使用 OpenAI Responses WebSocket 模式")
            return self._create_chat_completion_with_websocket_fallback(
                route=route,
                request_data=request_data,
            )
        elif reason is not None and route.websocket_mode_enabled:
            self._log(f"OpenAI Responses WebSocket 未启用: {reason}")

        return self._create_http_chat_completion(
            route=route,
            request_data=request_data,
        )

    def close(self) -> None:
        with self._websocket_sessions_lock:
            sessions = list(self._websocket_sessions.values())
            self._websocket_sessions = {}
        for session in sessions:
            with session.lock:
                self._close_websocket_session(session)
                self._reset_websocket_session_chain(session)


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
