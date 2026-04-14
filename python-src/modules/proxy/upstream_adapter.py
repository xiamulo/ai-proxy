from __future__ import annotations

import contextlib
import copy
import json
import os
import ssl
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import litellm
from openai import OpenAI

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
LITELLM_CONNECT_RETRY_COUNT = 2
UPSTREAM_PARAM_SELF_HEAL_MAX_ATTEMPTS = 3
WEBSOCKET_SESSION_IDLE_TIMEOUT_SECONDS = 15 * 60
WEBSOCKET_SESSION_MAX_COUNT = 64
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
            return content
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
            return arguments
        if arguments is None:
            return ""
        try:
            return json.dumps(arguments, ensure_ascii=False)
        except TypeError:
            return None

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
    def _messages_extend_prefix(
        previous_messages: list[dict[str, Any]],
        current_messages: list[dict[str, Any]],
    ) -> bool:
        previous_count = len(previous_messages)
        if previous_count == 0 or len(current_messages) <= previous_count:
            return False
        return current_messages[:previous_count] == previous_messages

    @staticmethod
    def _extract_websocket_session_key(request_data: dict[str, Any]) -> tuple[str | None, bool]:
        metadata_obj = request_data.get("metadata")
        if isinstance(metadata_obj, dict):
            metadata = cast(dict[str, Any], metadata_obj)
            for key in (
                "mtga_ws_session_id",
                "mtga_session_id",
                "session_id",
                "conversation_id",
                "chat_id",
                "thread_id",
            ):
                value_obj = metadata.get(key)
                if isinstance(value_obj, str) and value_obj.strip():
                    return value_obj.strip(), True
        return None, False

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

    def _prune_websocket_sessions(self) -> None:
        now = time.time()
        with self._websocket_sessions_lock:
            stale_sessions = self._collect_stale_websocket_sessions_locked(now=now)
        for session in stale_sessions:
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
        session.conversation_messages = []

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
        connection_context = client.responses.connect()
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
    ) -> ResponsesWebSocketSessionState | None:
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
            return best_session
        return None

    def _get_or_create_websocket_session(
        self,
        *,
        request_data: dict[str, Any],
        current_messages: list[dict[str, Any]],
    ) -> ResponsesWebSocketSessionState:
        self._prune_websocket_sessions()
        explicit_key, explicit = self._extract_websocket_session_key(request_data)
        new_session_created = False
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
                selected_session = session
            else:
                matched_session = self._find_matching_websocket_session(
                    current_messages=current_messages
                )
                if matched_session is not None:
                    selected_session = matched_session
                else:
                    session = ResponsesWebSocketSessionState(
                        session_id=self._new_websocket_session_id(),
                        explicit_session_key=False,
                    )
                    self._websocket_sessions[session.session_id] = session
                    selected_session = session
                    new_session_created = True

        if new_session_created:
            self._enforce_websocket_session_capacity()
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
        if isinstance(node, list):
            return [cls._normalize_strict_schema_node(item) for item in cast(list[Any], node)]
        if not isinstance(node, dict):
            return node

        normalized = {
            key: cls._normalize_strict_schema_node(value)
            for key, value in cast(dict[str, Any], node).items()
        }
        type_obj = normalized.get("type")
        is_object_type = type_obj == "object" or (
            isinstance(type_obj, list) and "object" in type_obj
        )
        properties_obj = normalized.get("properties")

        if is_object_type:
            normalized["additionalProperties"] = False
            if isinstance(properties_obj, dict):
                normalized["required"] = list(cast(dict[str, Any], properties_obj).keys())

        return normalized

    @classmethod
    def _normalize_strict_function_parameters(
        cls, parameters: dict[str, Any] | None
    ) -> dict[str, Any]:
        if parameters is None:
            return {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
        return cast(dict[str, Any], cls._normalize_strict_schema_node(parameters))

    @classmethod
    def _convert_function_tool_definition(cls, definition: dict[str, Any]) -> dict[str, Any] | None:
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
                cast(dict[str, Any] | None, parameters)
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
    def _convert_chat_tools_to_responses_tools(
        cls, request_data: dict[str, Any]
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
                    cast(dict[str, Any], function_obj)
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
                    cast(dict[str, Any], function_obj)
                )
                if converted_tool is None:
                    return None
                converted_tools.append(converted_tool)

        return converted_tools or None

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
                        "id": call_id,
                        "call_id": call_id,
                        "name": name_obj,
                        "arguments": arguments,
                        "status": "completed",
                    }
                )
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
    def _build_openai_responses_websocket_request(
        cls, *, route: UpstreamRoute, request_data: dict[str, Any]
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

        converted_tools = cls._convert_chat_tools_to_responses_tools(request_data)
        if request_data.get("tools") is not None or request_data.get("functions") is not None:
            if converted_tools is None:
                return None, "顶层 tools/functions 无法转换为 Responses tools"
            payload["tools"] = converted_tools

        converted_tool_choice = cls._convert_chat_tool_choice(request_data)
        if converted_tool_choice is not None:
            payload["tool_choice"] = converted_tool_choice

        passthrough_keys = (
            "metadata",
            "parallel_tool_calls",
            "service_tier",
            "store",
            "temperature",
            "top_p",
            "user",
        )
        for key in passthrough_keys:
            if key in request_data:
                payload[key] = request_data[key]

        if "reasoning_effort" in request_data:
            payload["reasoning"] = {"effort": request_data["reasoning_effort"]}

        max_tokens = request_data.get("max_completion_tokens", request_data.get("max_tokens"))
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

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

    def _build_openai_responses_websocket_turn_payload(
        self,
        *,
        route: UpstreamRoute,
        request_data: dict[str, Any],
        session: ResponsesWebSocketSessionState,
        current_messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, bool, str | None]:
        continue_with_previous = False
        turn_request_data = copy.deepcopy(request_data)
        previous_messages = session.conversation_messages
        if (
            session.last_response_id
            and previous_messages
            and self._messages_extend_prefix(previous_messages, current_messages)
        ):
            delta_messages = current_messages[len(previous_messages) :]
            if delta_messages:
                turn_request_data["messages"] = delta_messages
                continue_with_previous = True
            else:
                self._log(
                    "OpenAI Responses WebSocket 检测到重复历史，改为新链路: "
                    f"session={session.session_id}"
                )

        payload, error = self._build_openai_responses_websocket_request(
            route=route,
            request_data=turn_request_data,
        )
        if payload is None:
            return None, False, error

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
        return payload, continue_with_previous, None

    def _stream_openai_responses_websocket_turn(  # noqa: PLR0912, PLR0915
        self,
        *,
        route: UpstreamRoute,
        session: ResponsesWebSocketSessionState,
        payload: dict[str, Any],
        current_messages: list[dict[str, Any]],
    ) -> Any:
        connection = session.connection
        if connection is None:
            raise RuntimeError("OpenAI Responses WebSocket 连接未建立")

        response_id = "chatcmpl_websocket_pending"
        created = 0
        function_calls: dict[str, ResponsesWebSocketFunctionCallState] = {}
        saw_function_call = False
        assistant_content_parts: list[str] = []
        assistant_tool_calls: list[dict[str, Any]] = []

        def now_ts() -> int:
            return int(time.time())

        connection.response.create(**payload)
        for event in connection:
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
                    delta={"tool_calls": [assistant_tool_calls[-1]]},
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
                session.conversation_messages = updated_messages
                session.last_used_at = time.time()
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
                message_obj = error_payload.get("message")
                message = (
                    message_obj.strip()
                    if isinstance(message_obj, str) and message_obj.strip()
                    else f"OpenAI Responses WebSocket 返回错误事件: {event_type}"
                )
                code_obj = error_payload.get("code")
                code = code_obj.strip() if isinstance(code_obj, str) and code_obj.strip() else None
                raise ResponsesWebSocketSessionError(message, code=code, status=status)

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
            with session.lock:
                for attempt in range(2):
                    emitted_output = False
                    self._ensure_websocket_session_connection(session=session, route=route)
                    payload, _continued, error = (
                        self._build_openai_responses_websocket_turn_payload(
                            route=route,
                            request_data=request_data,
                            session=session,
                            current_messages=current_messages,
                        )
                    )
                    if payload is None:
                        raise ValueError(error or "无法构造 Responses WebSocket 请求")

                    try:
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
                        recoverable_codes = {
                            "previous_response_not_found",
                            "websocket_connection_limit_reached",
                        }
                        if emitted_output or exc.code not in recoverable_codes or attempt >= 1:
                            raise
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
                    except Exception:
                        self._close_websocket_session(session)
                        self._reset_websocket_session_chain(session)
                        raise

        return generate()

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
        call_kwargs = self._normalize_provider_chat_request(
            route=route,
            request_data=request_data,
        )
        call_kwargs["model"] = self._resolve_chat_completion_model(route)
        call_kwargs["base_url"] = route.base_url
        call_kwargs["api_key"] = route.api_key
        call_kwargs["ssl_verify"] = self._build_request_ssl_verify(self._disable_ssl_strict_mode)
        return _create_litellm_completion(**call_kwargs)

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
                self._log(f"OpenAI Responses WebSocket 已回退到 HTTP: {exc}")
            except Exception as exc:  # noqa: BLE001
                if websocket_started:
                    raise
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
