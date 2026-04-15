from __future__ import annotations

import tempfile
import time
import unittest
from typing import Any, get_origin
from unittest.mock import patch

import httpx
from litellm.exceptions import BadRequestError

from modules.proxy.proxy_config import (
    ANTHROPIC_PROVIDER,
    GEMINI_NATIVE_X_GOOG_API_KEY_MODEL_DISCOVERY,
    OPENAI_CHAT_COMPLETION_PROVIDER,
    OPENAI_RESPONSE_PROVIDER,
    ProxyConfig,
)
from modules.proxy.proxy_config import (
    build_proxy_config as build_runtime_proxy_config,
)
from modules.proxy.upstream_adapter import (
    CHAT_COMPLETIONS_REQUEST_API,
    LiteLLMUpstreamAdapter,
    OpenAIWebSocketConnectionOptions,
    ResponsesWebSocketSessionError,
    ResponsesWebSocketSessionState,
    build_upstream_route,
)


class DummyResourceManager:
    def __init__(self, *, user_data_dir: str, program_resource_dir: str) -> None:
        self._user_data_dir = user_data_dir
        self._program_resource_dir = program_resource_dir

    def get_user_data_dir(self) -> str:
        return self._user_data_dir

    def get_program_resource_dir(self) -> str:
        return self._program_resource_dir

    def get_user_config_file(self) -> str:
        return f"{self._user_data_dir}/config.yml"


def _build_proxy_config(  # noqa: PLR0913
    *,
    provider: str = OPENAI_CHAT_COMPLETION_PROVIDER,
    target_api_base_url: str,
    target_model_id: str,
    middle_route: str | None = None,
    api_key: str = "test-key",
    model_discovery_strategy: str | None = None,
    prompt_cache_bucket_id: str = "",
    prompt_cache_enabled: bool = True,
    request_params_enabled: bool = True,
    websocket_mode_enabled: bool = False,
) -> ProxyConfig:
    return ProxyConfig(
        provider=provider,
        target_api_base_url=target_api_base_url,
        middle_route=middle_route or "",
        custom_model_id="gpt-5",
        target_model_id=target_model_id,
        stream_mode=None,
        debug_mode=False,
        disable_ssl_strict_mode=False,
        api_key=api_key,
        mtga_auth_key="mtga-auth",
        model_discovery_strategy=model_discovery_strategy,
        prompt_cache_bucket_id=prompt_cache_bucket_id,
        prompt_cache_enabled=prompt_cache_enabled,
        request_params_enabled=request_params_enabled,
        websocket_mode_enabled=websocket_mode_enabled,
    )


def _build_bad_request_error(
    *,
    message: str,
    body: dict[str, Any] | None = None,
    model: str = "gpt-5",
) -> BadRequestError:
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response_body = body or {
        "error": {
            "code": None,
            "message": message,
            "param": None,
            "type": "invalid_request_error",
        }
    }
    response = httpx.Response(400, request=request, json=response_body)
    return BadRequestError(
        f"OpenAIException - {message}",
        model=model,
        llm_provider="openai",
        response=response,
        body=response_body,
    )


class UpstreamRouteTests(unittest.TestCase):
    def test_openai_websocket_connection_options_type_is_runtime_safe(self) -> None:
        self.assertIs(get_origin(OpenAIWebSocketConnectionOptions), dict)

    def test_build_proxy_config_ignores_legacy_group_mapped_model_id(self) -> None:
        temp_dir = tempfile.mkdtemp(prefix="mtga-proxy-config-")
        resource_manager = DummyResourceManager(
            user_data_dir=temp_dir,
            program_resource_dir=temp_dir,
        )

        proxy_config = build_runtime_proxy_config(
            {
                "api_url": "https://api.openai.com",
                "model_id": "gpt-4o-mini",
                "api_key": "test-key",
                "mapped_model_id": "legacy-group-model",
            },
            resource_manager=resource_manager,  # type: ignore[arg-type]
            log_func=lambda _message: None,
        )

        self.assertIsNotNone(proxy_config)
        assert proxy_config is not None
        self.assertEqual(proxy_config.custom_model_id, "")
        self.assertEqual(proxy_config.target_model_id, "gpt-4o-mini")

    def test_openai_chat_completion_route_keeps_middle_route(self) -> None:
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_CHAT_COMPLETION_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-4o-mini",
                middle_route="/v1",
            )
        )

        self.assertEqual(route.provider, OPENAI_CHAT_COMPLETION_PROVIDER)
        self.assertEqual(route.request_api, CHAT_COMPLETIONS_REQUEST_API)
        self.assertEqual(route.litellm_model, "gpt-4o-mini")
        self.assertEqual(route.base_url, "https://api.openai.com/v1")
        self.assertEqual(route.litellm_base_url, "https://api.openai.com/v1")
        self.assertTrue(route.middle_route_applied)
        self.assertFalse(route.middle_route_ignored)


class UpstreamAdapterTests(unittest.TestCase):
    def test_openai_response_builds_responses_payload_for_http_path(self) -> None:
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://example.com",
                target_model_id="gpt-5",
            )
        )

        payload, error = LiteLLMUpstreamAdapter._build_openai_responses_websocket_request(
            route=route,
            request_data={
                "messages": [{"role": "user", "content": "你好"}],
                "temperature": 0,
                "service_tier": "priority",
                "store": True,
                "reasoning_effort": "medium",
            },
        )

        self.assertIsNone(error)
        assert payload is not None
        self.assertEqual(payload["model"], "gpt-5")
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["service_tier"], "priority")
        self.assertEqual(payload["store"], True)
        self.assertEqual(payload["reasoning"], {"effort": "medium"})

    def test_openai_response_http_fallback_uses_direct_responses_stream(self) -> None:
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=lambda _message: None,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://example.com",
                target_model_id="gpt-5.4",
            )
        )

        with (
            patch.object(
                adapter,
                "_create_openai_responses_http_stream",
                return_value=iter([{"id": "chatcmpl_123", "choices": []}]),
            ) as responses_http_mock,
            patch("modules.proxy.upstream_adapter._create_litellm_completion") as completion_mock,
        ):
            chunks = list(
                adapter.create_chat_completion(
                    route=route,
                    request_data={
                        "messages": [{"role": "user", "content": "你好"}],
                        "stream": True,
                    },
                )
            )

        self.assertEqual(len(chunks), 1)
        responses_http_mock.assert_called_once()
        completion_mock.assert_not_called()

    def test_websocket_connection_uses_long_response_safe_options(self) -> None:
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=lambda _message: None,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        session = ResponsesWebSocketSessionState(
            session_id="session-1",
            explicit_session_key=True,
        )
        fake_connection = object()
        fake_client = type(
            "FakeClient",
            (),
            {
                "responses": type(
                    "FakeResponses",
                    (),
                    {
                        "connect": lambda self, **kwargs: connect_mock(**kwargs),
                    },
                )()
            },
        )()
        connect_kwargs: dict[str, Any] = {}

        class FakeConnectionContext:
            def __enter__(self) -> Any:
                return fake_connection

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

        def connect_mock(**kwargs: Any) -> Any:
            connect_kwargs.update(kwargs)
            return FakeConnectionContext()

        with patch("modules.proxy.upstream_adapter.OpenAI", return_value=fake_client):
            adapter._open_websocket_session_connection(
                session=session,
                route=route,
                reconnect_reason="initial_connect",
            )

        self.assertEqual(
            connect_kwargs["websocket_connection_options"]["ping_interval"],
            None,
        )
        self.assertEqual(
            connect_kwargs["websocket_connection_options"]["max_size"],
            None,
        )

    def test_openai_compatible_request_drops_optional_params_when_disabled(self) -> None:
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=lambda _message: None,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_CHAT_COMPLETION_PROVIDER,
                target_api_base_url="https://example.com",
                target_model_id="gpt-5.3-codex",
                request_params_enabled=False,
            )
        )

        with patch(
            "modules.proxy.upstream_adapter._create_litellm_completion",
            return_value={"id": "chatcmpl_123", "choices": []},
        ) as completion_mock:
            adapter.create_chat_completion(
                route=route,
                request_data={
                    "messages": [{"role": "user", "content": "你好"}],
                    "temperature": 0.1,
                    "top_p": 0.5,
                    "presence_penalty": 1,
                    "frequency_penalty": 1,
                    "reasoning_effort": "medium",
                    "metadata": {"source": "test"},
                },
            )

        call_kwargs = completion_mock.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gpt-5.3-codex")
        self.assertNotIn("temperature", call_kwargs)
        self.assertNotIn("top_p", call_kwargs)
        self.assertNotIn("presence_penalty", call_kwargs)
        self.assertNotIn("frequency_penalty", call_kwargs)
        self.assertNotIn("reasoning_effort", call_kwargs)
        self.assertNotIn("metadata", call_kwargs)

    def test_anthropic_uses_custom_base_url_without_openai_provider_override(self) -> None:
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=lambda _message: None,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=ANTHROPIC_PROVIDER,
                target_api_base_url="https://anthropic-proxy.example.com",
                target_model_id="claude-3-7-sonnet-latest",
                model_discovery_strategy=GEMINI_NATIVE_X_GOOG_API_KEY_MODEL_DISCOVERY,
            )
        )

        with patch(
            "modules.proxy.upstream_adapter._create_litellm_completion",
            return_value={"id": "chatcmpl_123", "choices": []},
        ) as completion_mock:
            adapter.create_chat_completion(
                route=route,
                request_data={"messages": [{"role": "user", "content": "你好"}]},
            )

        call_kwargs = completion_mock.call_args.kwargs
        self.assertEqual(call_kwargs["base_url"], "https://anthropic-proxy.example.com/v1")
        self.assertNotIn("api_base", call_kwargs)
        self.assertNotIn("custom_llm_provider", call_kwargs)

    def test_openai_response_gpt_5_4_stream_uses_websocket_mode(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )

        with (
            patch.object(
                adapter,
                "_create_openai_responses_websocket_stream",
                return_value=iter([]),
            ) as websocket_mock,
            patch(
                "modules.proxy.upstream_adapter._create_litellm_completion",
            ) as completion_mock,
        ):
            stream = adapter.create_chat_completion(
                route=route,
                request_data={
                    "messages": [{"role": "user", "content": "你好"}],
                    "stream": True,
                },
            )
            chunks = list(stream)

        self.assertEqual(chunks, [])
        websocket_mock.assert_called_once()
        completion_mock.assert_not_called()
        self.assertTrue(
            any("流式请求优先使用 OpenAI Responses WebSocket 模式" in item for item in logs)
        )

    def test_openai_chat_completion_gpt_5_4_stream_also_uses_websocket_mode(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_CHAT_COMPLETION_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )

        with (
            patch.object(
                adapter,
                "_create_openai_responses_websocket_stream",
                return_value=iter([]),
            ) as websocket_mock,
            patch(
                "modules.proxy.upstream_adapter._create_litellm_completion",
            ) as completion_mock,
        ):
            stream = adapter.create_chat_completion(
                route=route,
                request_data={
                    "messages": [{"role": "user", "content": "你好"}],
                    "stream": True,
                },
            )
            chunks = list(stream)

        self.assertEqual(chunks, [])
        websocket_mock.assert_called_once()
        completion_mock.assert_not_called()
        self.assertTrue(
            any("流式请求优先使用 OpenAI Responses WebSocket 模式" in item for item in logs)
        )

    def test_openai_response_gpt_5_4_stream_falls_back_to_http_when_websocket_connection_fails(
        self,
    ) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )

        def broken_websocket_stream() -> Any:
            raise RuntimeError("dial tcp timeout")
            yield  # pragma: no cover

        with (
            patch.object(
                adapter,
                "_create_openai_responses_websocket_stream",
                return_value=broken_websocket_stream(),
            ),
            patch.object(
                adapter,
                "_create_openai_responses_http_stream",
                return_value=iter([{"id": "chatcmpl_123", "choices": []}]),
            ) as http_fallback_mock,
        ):
            stream = adapter.create_chat_completion(
                route=route,
                request_data={
                    "messages": [{"role": "user", "content": "你好"}],
                    "stream": True,
                },
            )
            chunks = list(stream)

        self.assertEqual(len(chunks), 1)
        http_fallback_mock.assert_called_once()
        self.assertTrue(
            any("WebSocket 模式出问题了，没连接上，已回退到普通 HTTP" in item for item in logs)
        )

    def test_websocket_request_converts_tool_history_and_legacy_function_fields(self) -> None:
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        payload, error = LiteLLMUpstreamAdapter._build_openai_responses_websocket_request(
            route=route,
            request_data={
                "messages": [
                    {"role": "system", "content": "系统提示"},
                    {"role": "user", "content": "先查天气"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_weather",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city":"上海"}'},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_weather",
                        "content": "晴天",
                    },
                    {
                        "role": "assistant",
                        "function_call": {"name": "summarize", "arguments": {"result": "晴天"}},
                        "content": "我来整理",
                    },
                    {
                        "role": "function",
                        "name": "summarize",
                        "content": "总结完成",
                    },
                ],
                "functions": [
                    {
                        "name": "summarize",
                        "description": "总结天气结果",
                        "parameters": {
                            "type": "object",
                            "properties": {"result": {"type": "string"}},
                        },
                    }
                ],
                "function_call": {"name": "summarize"},
            },
        )

        self.assertIsNone(error)
        assert payload is not None
        self.assertEqual(payload["instructions"], "系统提示")
        self.assertEqual(payload["tool_choice"], {"type": "function", "name": "summarize"})
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertEqual(payload["tools"][0]["parameters"]["additionalProperties"], False)
        self.assertEqual(payload["tools"][0]["parameters"]["required"], ["result"])
        input_items = payload["input"]
        self.assertEqual(input_items[0]["role"], "user")
        self.assertEqual(input_items[1]["type"], "function_call")
        self.assertEqual(input_items[1]["call_id"], "call_weather")
        self.assertNotIn("id", input_items[1])
        self.assertEqual(input_items[2]["type"], "function_call_output")
        self.assertEqual(input_items[2]["call_id"], "call_weather")
        self.assertEqual(input_items[3]["type"], "message")
        self.assertEqual(input_items[3]["role"], "assistant")
        self.assertEqual(input_items[4]["type"], "function_call")
        self.assertEqual(input_items[4]["name"], "summarize")
        self.assertEqual(input_items[5]["type"], "function_call_output")
        self.assertEqual(input_items[5]["output"], "总结完成")

    def test_websocket_stream_emits_tool_calls_for_responses_events(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        captured_payload: dict[str, Any] = {}
        events = [
            {"type": "response.created", "response": {"id": "resp_123", "created_at": 123}},
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "lookup_weather",
                },
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_1",
                "arguments": '{"city":"上海"}',
            },
            {"type": "response.completed"},
        ]

        class FakeResponseResource:
            def create(self, **kwargs: Any) -> None:
                captured_payload.update(kwargs)

        class FakeConnection:
            def __init__(self) -> None:
                self.response = FakeResponseResource()

            def __enter__(self) -> FakeConnection:
                return self

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

            def __iter__(self) -> Any:
                return iter(events)

        class FakeResponses:
            def connect(self, **kwargs: Any) -> FakeConnection:
                return FakeConnection()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        with patch("modules.proxy.upstream_adapter.OpenAI", return_value=FakeClient()):
            stream = adapter.create_chat_completion(
                route=route,
                request_data={
                    "messages": [{"role": "user", "content": "查天气"}],
                    "stream": True,
                },
            )
            chunks = list(stream)

        self.assertEqual(captured_payload["model"], "gpt-5.4")
        self.assertEqual(
            chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "lookup_weather"
        )
        self.assertEqual(
            chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"], ""
        )
        self.assertEqual(
            chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            '{"city":"上海"}',
        )
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_websocket_session_reuses_connection_and_previous_response_id(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        created_payloads: list[dict[str, Any]] = []
        connect_calls = 0
        event_batches = [
            [
                {"type": "response.created", "response": {"id": "resp_1", "created_at": 1}},
                {"type": "response.output_text.delta", "delta": "你好"},
                {"type": "response.completed"},
            ],
            [
                {"type": "response.created", "response": {"id": "resp_2", "created_at": 2}},
                {"type": "response.output_text.delta", "delta": "继续"},
                {"type": "response.completed"},
            ],
        ]

        class FakeResponseResource:
            def create(self, **kwargs: Any) -> None:
                created_payloads.append(kwargs)

        class FakeConnection:
            def __init__(self) -> None:
                self.response = FakeResponseResource()

            def __iter__(self) -> Any:
                return iter(event_batches.pop(0))

        class FakeConnectionContext:
            def __init__(self) -> None:
                self._connection = FakeConnection()

            def __enter__(self) -> FakeConnection:
                nonlocal connect_calls
                connect_calls += 1
                return self._connection

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

        class FakeResponses:
            def connect(self, **kwargs: Any) -> FakeConnectionContext:
                return FakeConnectionContext()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        with patch("modules.proxy.upstream_adapter.OpenAI", return_value=FakeClient()):
            first_stream = adapter.create_chat_completion(
                route=route,
                request_data={
                    "messages": [{"role": "user", "content": "你好"}],
                    "stream": True,
                },
            )
            list(first_stream)
            second_stream = adapter.create_chat_completion(
                route=route,
                request_data={
                    "messages": [
                        {"role": "user", "content": "你好"},
                        {"role": "assistant", "content": "你好"},
                        {"role": "user", "content": "继续"},
                    ],
                    "stream": True,
                },
            )
            list(second_stream)

        self.assertEqual(connect_calls, 1)
        self.assertEqual(created_payloads[0]["input"][0]["role"], "user")
        self.assertEqual(created_payloads[1]["previous_response_id"], "resp_1")
        self.assertEqual(len(created_payloads[1]["input"]), 1)
        self.assertEqual(created_payloads[1]["input"][0]["role"], "user")

    def test_websocket_session_prewarms_new_cached_connection(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
                prompt_cache_enabled=True,
                prompt_cache_bucket_id="bucket-123",
            )
        )
        created_payloads: list[dict[str, Any]] = []
        connect_calls = 0
        event_batches = [
            [
                {"type": "response.created", "response": {"id": "resp_prewarm", "created_at": 1}},
                {"type": "response.completed"},
            ],
            [
                {"type": "response.created", "response": {"id": "resp_1", "created_at": 2}},
                {"type": "response.output_text.delta", "delta": "你好"},
                {"type": "response.completed"},
            ],
        ]

        class FakeResponseResource:
            def create(self, **kwargs: Any) -> None:
                created_payloads.append(kwargs)

        class FakeConnection:
            def __init__(self) -> None:
                self.response = FakeResponseResource()

            def __iter__(self) -> Any:
                return iter(event_batches.pop(0))

        class FakeConnectionContext:
            def __init__(self) -> None:
                self._connection = FakeConnection()

            def __enter__(self) -> FakeConnection:
                nonlocal connect_calls
                connect_calls += 1
                return self._connection

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

        class FakeResponses:
            def connect(self, **kwargs: Any) -> FakeConnectionContext:
                return FakeConnectionContext()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        with patch("modules.proxy.upstream_adapter.OpenAI", return_value=FakeClient()):
            stream = adapter.create_chat_completion(
                route=route,
                request_data={
                    "messages": [{"role": "user", "content": "你好"}],
                    "stream": True,
                },
            )
            chunks = list(stream)

        self.assertEqual(connect_calls, 1)
        self.assertEqual(len(created_payloads), 2)
        self.assertFalse(created_payloads[0]["generate"])
        self.assertEqual(
            created_payloads[0]["prompt_cache_key"],
            created_payloads[1]["prompt_cache_key"],
        )
        self.assertEqual(created_payloads[1]["previous_response_id"], "resp_prewarm")
        self.assertEqual(created_payloads[1]["input"], [])
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "你好")
        self.assertTrue(any("WebSocket 预热" in item for item in logs))

    def test_websocket_session_skips_prewarm_when_sdk_create_lacks_generate(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
                prompt_cache_enabled=True,
                prompt_cache_bucket_id="bucket-123",
            )
        )
        created_payloads: list[dict[str, Any]] = []
        connect_calls = 0
        events = [
            {"type": "response.created", "response": {"id": "resp_1", "created_at": 2}},
            {"type": "response.output_text.delta", "delta": "你好"},
            {"type": "response.completed"},
        ]

        class FakeResponseResource:
            def create(self, **kwargs: Any) -> None:
                created_payloads.append(kwargs)

        class FakeConnection:
            def __init__(self) -> None:
                self.response = FakeResponseResource()

            def __iter__(self) -> Any:
                return iter(events)

        class FakeConnectionContext:
            def __init__(self) -> None:
                self._connection = FakeConnection()

            def __enter__(self) -> FakeConnection:
                nonlocal connect_calls
                connect_calls += 1
                return self._connection

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

        class FakeResponses:
            def connect(self, **kwargs: Any) -> FakeConnectionContext:
                return FakeConnectionContext()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        with (
            patch("modules.proxy.upstream_adapter.OpenAI", return_value=FakeClient()),
            patch.object(
                adapter,
                "_websocket_response_create_supports_generate",
                return_value=False,
            ),
        ):
            stream = adapter.create_chat_completion(
                route=route,
                request_data={
                    "messages": [{"role": "user", "content": "你好"}],
                    "stream": True,
                },
            )
            chunks = list(stream)

        self.assertEqual(connect_calls, 1)
        self.assertEqual(len(created_payloads), 1)
        self.assertNotIn("generate", created_payloads[0])
        self.assertNotEqual(created_payloads[0]["input"], [])
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "你好")
        self.assertTrue(any("跳过预热" in item for item in logs))
        self.assertFalse(any("已锁定回退 HTTP" in item for item in logs))

    def test_websocket_session_skips_previous_response_id_when_request_signature_changes(
        self,
    ) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        created_payloads: list[dict[str, Any]] = []
        event_batches = [
            [
                {"type": "response.created", "response": {"id": "resp_1", "created_at": 1}},
                {"type": "response.output_text.delta", "delta": "你好"},
                {"type": "response.completed"},
            ],
            [
                {"type": "response.created", "response": {"id": "resp_2", "created_at": 2}},
                {"type": "response.output_text.delta", "delta": "继续"},
                {"type": "response.completed"},
            ],
        ]

        class FakeResponseResource:
            def create(self, **kwargs: Any) -> None:
                created_payloads.append(kwargs)

        class FakeConnection:
            def __init__(self) -> None:
                self.response = FakeResponseResource()

            def __iter__(self) -> Any:
                return iter(event_batches.pop(0))

        class FakeConnectionContext:
            def __init__(self) -> None:
                self._connection = FakeConnection()

            def __enter__(self) -> FakeConnection:
                return self._connection

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

        class FakeResponses:
            def connect(self, **kwargs: Any) -> FakeConnectionContext:
                return FakeConnectionContext()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        with patch("modules.proxy.upstream_adapter.OpenAI", return_value=FakeClient()):
            list(
                adapter.create_chat_completion(
                    route=route,
                    request_data={
                        "messages": [{"role": "user", "content": "你好"}],
                        "stream": True,
                    },
                )
            )
            list(
                adapter.create_chat_completion(
                    route=route,
                    request_data={
                        "messages": [
                            {"role": "user", "content": "你好"},
                            {"role": "assistant", "content": "你好"},
                            {"role": "user", "content": "继续"},
                        ],
                        "stream": True,
                        "parallel_tool_calls": False,
                    },
                )
            )

        self.assertNotIn("previous_response_id", created_payloads[1])
        self.assertTrue(any("续链签名不一致" in item for item in logs))

    def test_websocket_session_long_chain_reuses_single_connection(self) -> None:
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=lambda _message: None,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        created_payloads: list[dict[str, Any]] = []
        connect_calls = 0
        total_turns = 8
        event_batches = [
            [
                {
                    "type": "response.created",
                    "response": {"id": f"resp_{turn + 1}", "created_at": turn + 1},
                },
                {"type": "response.output_text.delta", "delta": f"assistant-{turn + 1}"},
                {"type": "response.completed"},
            ]
            for turn in range(total_turns)
        ]

        class FakeResponseResource:
            def create(self, **kwargs: Any) -> None:
                created_payloads.append(kwargs)

        class FakeConnection:
            def __init__(self) -> None:
                self.response = FakeResponseResource()
                self._turn = 0

            def __iter__(self) -> Any:
                batch = event_batches[self._turn]
                self._turn += 1
                return iter(batch)

        class FakeConnectionContext:
            def __init__(self) -> None:
                self._connection = FakeConnection()

            def __enter__(self) -> FakeConnection:
                nonlocal connect_calls
                connect_calls += 1
                return self._connection

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

        class FakeResponses:
            def connect(self, **kwargs: Any) -> FakeConnectionContext:
                return FakeConnectionContext()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        with patch("modules.proxy.upstream_adapter.OpenAI", return_value=FakeClient()):
            messages: list[dict[str, str]] = [{"role": "user", "content": "turn-1"}]
            for turn in range(total_turns):
                list(
                    adapter.create_chat_completion(
                        route=route,
                        request_data={"messages": messages, "stream": True},
                    )
                )
                if turn < total_turns - 1:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": f"assistant-{turn + 1}"},
                        {"role": "user", "content": f"turn-{turn + 2}"},
                    ]

        self.assertEqual(connect_calls, 1)
        self.assertNotIn("previous_response_id", created_payloads[0])
        for turn in range(1, total_turns):
            self.assertEqual(created_payloads[turn]["previous_response_id"], f"resp_{turn}")
            self.assertEqual(len(created_payloads[turn]["input"]), 1)

    def test_websocket_session_reconnects_after_connection_limit(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        created_payloads: list[dict[str, Any]] = []
        connect_calls = 0
        current_batch = 0
        event_batches = [
            [
                {"type": "response.created", "response": {"id": "resp_1", "created_at": 1}},
                {"type": "response.output_text.delta", "delta": "首轮"},
                {"type": "response.completed"},
            ],
            [
                {"type": "response.created", "response": {"id": "resp_2", "created_at": 2}},
                {"type": "response.output_text.delta", "delta": "重连后"},
                {"type": "response.completed"},
            ],
        ]

        class FakeResponseResource:
            def create(self, **kwargs: Any) -> None:
                created_payloads.append(kwargs)

        class FakeConnection:
            def __init__(self) -> None:
                self.response = FakeResponseResource()

            def __iter__(self) -> Any:
                nonlocal current_batch
                batch = event_batches[current_batch]
                current_batch += 1
                return iter(batch)

        class FakeConnectionContext:
            def __init__(self) -> None:
                self._connection = FakeConnection()

            def __enter__(self) -> FakeConnection:
                nonlocal connect_calls
                connect_calls += 1
                return self._connection

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

        class FakeResponses:
            def connect(self, **kwargs: Any) -> FakeConnectionContext:
                return FakeConnectionContext()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        with patch("modules.proxy.upstream_adapter.OpenAI", return_value=FakeClient()):
            list(
                adapter.create_chat_completion(
                    route=route,
                    request_data={
                        "messages": [{"role": "user", "content": "首轮"}],
                        "stream": True,
                    },
                )
            )
            assert adapter._websocket_sessions
            session = next(iter(adapter._websocket_sessions.values()))
            session.connection_started_at = time.time() - 56 * 60
            list(
                adapter.create_chat_completion(
                    route=route,
                    request_data={
                        "messages": [
                            {"role": "user", "content": "首轮"},
                            {"role": "assistant", "content": "首轮"},
                            {"role": "user", "content": "第二轮"},
                        ],
                        "stream": True,
                    },
                )
            )

        self.assertEqual(connect_calls, 2)
        self.assertNotIn("previous_response_id", created_payloads[1])
        self.assertTrue(any("reason=connection_limit_guard" in item for item in logs))

    def test_websocket_session_recovers_from_previous_response_not_found(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        created_payloads: list[dict[str, Any]] = []
        connect_calls = 0
        current_batch = 0
        event_batches = [
            [
                {"type": "response.created", "response": {"id": "resp_1", "created_at": 1}},
                {"type": "response.output_text.delta", "delta": "第一轮"},
                {"type": "response.completed"},
            ],
            [
                {
                    "type": "error",
                    "status": 400,
                    "error": {
                        "type": "invalid_request_error",
                        "code": "previous_response_not_found",
                        "message": "missing response",
                    },
                }
            ],
            [
                {"type": "response.created", "response": {"id": "resp_2", "created_at": 2}},
                {"type": "response.output_text.delta", "delta": "恢复成功"},
                {"type": "response.completed"},
            ],
        ]

        class FakeResponseResource:
            def create(self, **kwargs: Any) -> None:
                created_payloads.append(kwargs)

        class FakeConnection:
            def __init__(self) -> None:
                self.response = FakeResponseResource()

            def __iter__(self) -> Any:
                nonlocal current_batch
                batch = event_batches[current_batch]
                current_batch += 1
                return iter(batch)

        class FakeConnectionContext:
            def __init__(self) -> None:
                self._connection = FakeConnection()

            def __enter__(self) -> FakeConnection:
                nonlocal connect_calls
                connect_calls += 1
                return self._connection

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

        class FakeResponses:
            def connect(self, **kwargs: Any) -> FakeConnectionContext:
                return FakeConnectionContext()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        with patch("modules.proxy.upstream_adapter.OpenAI", return_value=FakeClient()):
            list(
                adapter.create_chat_completion(
                    route=route,
                    request_data={
                        "messages": [{"role": "user", "content": "第一轮"}],
                        "stream": True,
                    },
                )
            )
            list(
                adapter.create_chat_completion(
                    route=route,
                    request_data={
                        "messages": [
                            {"role": "user", "content": "第一轮"},
                            {"role": "assistant", "content": "第一轮"},
                            {"role": "user", "content": "继续"},
                        ],
                        "stream": True,
                    },
                )
            )

        self.assertEqual(connect_calls, 2)
        self.assertEqual(created_payloads[1]["previous_response_id"], "resp_1")
        self.assertNotIn("previous_response_id", created_payloads[2])
        self.assertTrue(any("code=previous_response_not_found" in item for item in logs))

    def test_websocket_session_retries_third_party_handshake_with_model_query(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://example-proxy.test",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        connect_kwargs_list: list[dict[str, Any]] = []
        created_payloads: list[dict[str, Any]] = []

        class FakeHandshakeError(RuntimeError):
            def __init__(self) -> None:
                super().__init__("Handshake failed with status 403")
                self.status_code = 403

        class FakeResponseResource:
            def create(self, **kwargs: Any) -> None:
                created_payloads.append(kwargs)

        class FakeConnection:
            def __init__(self) -> None:
                self.response = FakeResponseResource()

            def __iter__(self) -> Any:
                return iter(
                    [
                        {
                            "type": "response.created",
                            "response": {"id": "resp_ok", "created_at": 1},
                        },
                        {"type": "response.output_text.delta", "delta": "完成"},
                        {"type": "response.completed"},
                    ]
                )

        class FakeConnectionContext:
            def __enter__(self) -> FakeConnection:
                return FakeConnection()

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

        class FakeResponses:
            def __init__(self) -> None:
                self.calls = 0

            def connect(self, **kwargs: Any) -> FakeConnectionContext:
                self.calls += 1
                connect_kwargs_list.append(kwargs)
                if self.calls == 1:
                    raise FakeHandshakeError()
                return FakeConnectionContext()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        with patch("modules.proxy.upstream_adapter.OpenAI", return_value=FakeClient()):
            chunks = list(
                adapter.create_chat_completion(
                    route=route,
                    request_data={
                        "messages": [{"role": "user", "content": "你好"}],
                        "stream": True,
                    },
                )
            )

        self.assertEqual(len(connect_kwargs_list), 2)
        self.assertNotIn("extra_query", connect_kwargs_list[0])
        self.assertEqual(connect_kwargs_list[1]["extra_query"], {"model": "gpt-5.4"})
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "完成")
        self.assertTrue(any("建连失败" in item for item in logs))
        self.assertTrue(any("代理兼容重试" in item for item in logs))

    def test_websocket_session_retries_without_sampling_after_empty_close(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        created_payloads: list[dict[str, Any]] = []
        current_batch = 0
        event_batches = [
            [],
            [
                {"type": "response.created", "response": {"id": "resp_ok", "created_at": 2}},
                {"type": "response.output_text.delta", "delta": "恢复成功"},
                {"type": "response.completed"},
            ],
        ]

        class FakeResponseResource:
            def create(self, **kwargs: Any) -> None:
                created_payloads.append(kwargs)

        class FakeConnection:
            def __init__(self) -> None:
                self.response = FakeResponseResource()

            def __iter__(self) -> Any:
                nonlocal current_batch
                batch = event_batches[current_batch]
                current_batch += 1
                return iter(batch)

        class FakeConnectionContext:
            def __enter__(self) -> FakeConnection:
                return FakeConnection()

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

        class FakeResponses:
            def connect(self, **kwargs: Any) -> FakeConnectionContext:
                return FakeConnectionContext()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        with patch("modules.proxy.upstream_adapter.OpenAI", return_value=FakeClient()):
            chunks = list(
                adapter.create_chat_completion(
                    route=route,
                    request_data={
                        "messages": [{"role": "user", "content": "你好"}],
                        "stream": True,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "stream_options": {"include_usage": True},
                    },
                )
            )

        self.assertEqual(len(created_payloads), 2)
        self.assertIn("temperature", created_payloads[0])
        self.assertIn("top_p", created_payloads[0])
        self.assertNotIn("temperature", created_payloads[1])
        self.assertNotIn("top_p", created_payloads[1])
        self.assertNotIn("stream_options", created_payloads[1])
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "恢复成功")
        self.assertTrue(any("空响应关闭，移除采样参数重试" in item for item in logs))

    def test_websocket_route_sticky_fallback_skips_second_websocket_attempt(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        websocket_attempts = 0
        http_attempts = 0

        def broken_websocket_stream() -> Any:
            nonlocal websocket_attempts
            websocket_attempts += 1
            raise RuntimeError("ws down")
            yield  # pragma: no cover

        def fake_http_stream(*, route: Any, request_data: Any) -> Any:
            nonlocal http_attempts
            http_attempts += 1
            return iter(
                [
                    {
                        "id": "resp-http",
                        "created": 1,
                        "model": route.litellm_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "ok"},
                                "finish_reason": None,
                            }
                        ],
                    }
                ]
            )

        with (
            patch.object(
                adapter,
                "_create_openai_responses_websocket_stream",
                return_value=broken_websocket_stream(),
            ),
            patch.object(
                adapter,
                "_create_http_chat_completion",
                side_effect=fake_http_stream,
            ),
        ):
            list(
                adapter.create_chat_completion(
                    route=route,
                    request_data={
                        "messages": [{"role": "user", "content": "first"}],
                        "stream": True,
                    },
                )
            )
            list(
                adapter.create_chat_completion(
                    route=route,
                    request_data={
                        "messages": [{"role": "user", "content": "second"}],
                        "stream": True,
                    },
                )
            )

        self.assertEqual(websocket_attempts, 1)
        self.assertEqual(http_attempts, 2)
        self.assertTrue(any("锁定回退 HTTP" in item for item in logs))

    def test_websocket_session_does_not_retry_after_partial_stream_error(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        created_payloads: list[dict[str, Any]] = []
        connect_calls = 0
        event_batches = [
            [
                {"type": "response.created", "response": {"id": "resp_1", "created_at": 1}},
                {"type": "response.output_text.delta", "delta": "部分"},
                {
                    "type": "error",
                    "status": 400,
                    "error": {
                        "type": "invalid_request_error",
                        "code": "previous_response_not_found",
                        "message": "missing response",
                    },
                },
            ]
        ]

        class FakeResponseResource:
            def create(self, **kwargs: Any) -> None:
                created_payloads.append(kwargs)

        class FakeConnection:
            def __init__(self) -> None:
                self.response = FakeResponseResource()

            def __iter__(self) -> Any:
                return iter(event_batches[0])

        class FakeConnectionContext:
            def __init__(self) -> None:
                self._connection = FakeConnection()

            def __enter__(self) -> FakeConnection:
                nonlocal connect_calls
                connect_calls += 1
                return self._connection

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

        class FakeResponses:
            def connect(self, **kwargs: Any) -> FakeConnectionContext:
                return FakeConnectionContext()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        with patch("modules.proxy.upstream_adapter.OpenAI", return_value=FakeClient()):
            stream = adapter.create_chat_completion(
                route=route,
                request_data={
                    "messages": [{"role": "user", "content": "第一轮"}],
                    "stream": True,
                },
            )
            iterator = iter(stream)
            first_chunk = next(iterator)
            self.assertEqual(first_chunk["choices"][0]["delta"]["content"], "部分")
            with self.assertRaises(ResponsesWebSocketSessionError):
                next(iterator)

        self.assertEqual(connect_calls, 1)
        self.assertEqual(len(created_payloads), 1)

    def test_websocket_session_does_not_use_user_as_session_key(self) -> None:
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=lambda _message: None,
        )
        session_key, explicit = adapter._extract_websocket_session_key(
            {"user": "same-user", "messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertIsNone(session_key)
        self.assertFalse(explicit)

    def test_websocket_session_prunes_idle_sessions(self) -> None:
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=lambda _message: None,
        )
        stale_session = ResponsesWebSocketSessionState(
            session_id="stale",
            explicit_session_key=True,
            last_used_at=time.time() - (16 * 60),
        )
        fresh_session = ResponsesWebSocketSessionState(
            session_id="fresh",
            explicit_session_key=True,
            last_used_at=time.time(),
        )
        adapter._websocket_sessions = {
            stale_session.session_id: stale_session,
            fresh_session.session_id: fresh_session,
        }

        adapter._prune_websocket_sessions()

        self.assertNotIn("stale", adapter._websocket_sessions)
        self.assertIn("fresh", adapter._websocket_sessions)

    def test_websocket_session_ambiguous_prefix_creates_new_session(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        shared_messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        session_a = ResponsesWebSocketSessionState(
            session_id="session-a",
            explicit_session_key=False,
            last_used_at=time.time(),
            conversation_messages=shared_messages,
        )
        session_b = ResponsesWebSocketSessionState(
            session_id="session-b",
            explicit_session_key=False,
            last_used_at=time.time(),
            conversation_messages=shared_messages,
        )
        adapter._websocket_sessions = {
            session_a.session_id: session_a,
            session_b.session_id: session_b,
        }

        selected_session = adapter._get_or_create_websocket_session(
            request_data={
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "continue"},
                ]
            },
            current_messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "continue"},
            ],
        )

        self.assertNotIn(selected_session.session_id, {"session-a", "session-b"})
        self.assertTrue(any("reason=ambiguous_prefix_new" in item for item in logs))

    def test_websocket_session_logs_explicit_and_matched_selection_reason(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )

        explicit_session = adapter._get_or_create_websocket_session(
            request_data={
                "metadata": {"session_id": "session-explicit"},
                "messages": [{"role": "user", "content": "explicit"}],
            },
            current_messages=[{"role": "user", "content": "explicit"}],
        )
        self.assertEqual(explicit_session.session_id, "session-explicit")

        matched_session = ResponsesWebSocketSessionState(
            session_id="session-matched",
            explicit_session_key=False,
            last_used_at=time.time(),
            conversation_messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        )
        adapter._websocket_sessions[matched_session.session_id] = matched_session
        reused_session = adapter._get_or_create_websocket_session(
            request_data={
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "continue"},
                ]
            },
            current_messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "continue"},
            ],
        )

        self.assertEqual(reused_session.session_id, "session-matched")
        self.assertTrue(any("reason=explicit_new" in item for item in logs))
        self.assertTrue(any("reason=matched_prefix" in item for item in logs))

    def test_websocket_session_history_index_reuses_anonymous_session(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )

        first_session = adapter._get_or_create_websocket_session(
            request_data={
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
            },
            current_messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        )
        self.assertFalse(first_session.explicit_session_key)

        second_session = adapter._get_or_create_websocket_session(
            request_data={
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "continue"},
                ]
            },
            current_messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "continue"},
            ],
        )

        self.assertEqual(second_session.session_id, first_session.session_id)
        self.assertTrue(any("reason=matched_history_prefix" in item for item in logs))

    def test_websocket_request_normalizes_strict_function_schema_recursively(self) -> None:
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        payload, error = LiteLLMUpstreamAdapter._build_openai_responses_websocket_request(
            route=route,
            request_data={
                "messages": [{"role": "user", "content": "创建任务"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Task",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "meta": {
                                        "type": "object",
                                        "properties": {
                                            "priority": {"type": "string"},
                                            "tags": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "name": {"type": "string"},
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    }
                ],
            },
        )

        self.assertIsNone(error)
        assert payload is not None
        parameters = payload["tools"][0]["parameters"]
        self.assertEqual(parameters["additionalProperties"], False)
        self.assertEqual(parameters["required"], ["meta", "title"])
        self.assertEqual(parameters["properties"]["meta"]["additionalProperties"], False)
        self.assertEqual(parameters["properties"]["meta"]["required"], ["priority", "tags"])
        self.assertEqual(
            parameters["properties"]["meta"]["properties"]["tags"]["items"]["additionalProperties"],
            False,
        )
        self.assertEqual(
            parameters["properties"]["meta"]["properties"]["tags"]["items"]["required"],
            ["name"],
        )

    def test_websocket_request_keeps_non_strict_function_schema_unchanged(self) -> None:
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        payload, error = LiteLLMUpstreamAdapter._build_openai_responses_websocket_request(
            route=route,
            request_data={
                "messages": [{"role": "user", "content": "创建任务"}],
                "functions": [
                    {
                        "name": "Task",
                        "strict": False,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                            },
                        },
                    }
                ],
            },
        )

        self.assertIsNone(error)
        assert payload is not None
        parameters = payload["tools"][0]["parameters"]
        self.assertNotIn("additionalProperties", parameters)
        self.assertNotIn("required", parameters)

    def test_websocket_request_strips_unsupported_strict_schema_keywords(self) -> None:
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        payload, error = LiteLLMUpstreamAdapter._build_openai_responses_websocket_request(
            route=route,
            request_data={
                "messages": [{"role": "user", "content": "读取日志"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_preview_console_logs",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "log_levels": {
                                        "type": "array",
                                        "uniqueItems": True,
                                        "minItems": 1,
                                        "maxItems": 5,
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                            },
                                            "allOf": [{"type": "object"}],
                                        },
                                    }
                                    ,
                                    "url": {
                                        "type": "string",
                                        "format": "uri",
                                    }
                                },
                            },
                        },
                    }
                ],
            },
        )

        self.assertIsNone(error)
        assert payload is not None
        log_levels = payload["tools"][0]["parameters"]["properties"]["log_levels"]
        self.assertNotIn("uniqueItems", log_levels)
        items = log_levels["items"]
        self.assertNotIn("allOf", items)
        self.assertEqual(items["additionalProperties"], False)
        self.assertNotIn(
            "format",
            payload["tools"][0]["parameters"]["properties"]["url"],
        )

    def test_websocket_request_does_not_forward_internal_session_metadata_keys(self) -> None:
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        payload, error = LiteLLMUpstreamAdapter._build_openai_responses_websocket_request(
            route=route,
            request_data={
                "messages": [{"role": "user", "content": "你好"}],
                "metadata": {
                    "session_id": "abc",
                    "conversation_id": "conv-1",
                    "foo": "bar",
                },
            },
        )

        self.assertIsNone(error)
        assert payload is not None
        self.assertEqual(payload["metadata"], {"foo": "bar"})

    def test_websocket_request_includes_prompt_cache_key(self) -> None:
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
                prompt_cache_enabled=True,
                prompt_cache_bucket_id="bucket-123",
            )
        )
        payload, error = LiteLLMUpstreamAdapter._build_openai_responses_websocket_request(
            route=route,
            request_data={
                "messages": [{"role": "user", "content": "你好"}],
                "metadata": {"session_id": "conversation-123"},
            },
        )

        self.assertIsNone(error)
        assert payload is not None
        self.assertEqual(
            payload["prompt_cache_key"],
            f"{route.prompt_cache_key}:meta:conversation-123",
        )

    def test_websocket_request_uses_top_level_conversation_anchor_for_prompt_cache_key(
        self,
    ) -> None:
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
                prompt_cache_enabled=True,
                prompt_cache_bucket_id="bucket-123",
            )
        )
        payload, error = LiteLLMUpstreamAdapter._build_openai_responses_websocket_request(
            route=route,
            request_data={
                "messages": [{"role": "user", "content": "你好"}],
                "conversation_id": "conversation-top-level-123",
            },
        )

        self.assertIsNone(error)
        assert payload is not None
        self.assertEqual(
            payload["prompt_cache_key"],
            f"{route.prompt_cache_key}:meta:conversation-top-level-123",
        )

    def test_websocket_request_passthroughs_cache_related_fields(self) -> None:
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        payload, error = LiteLLMUpstreamAdapter._build_openai_responses_websocket_request(
            route=route,
            request_data={
                "messages": [{"role": "user", "content": "你好"}],
                "parallel_tool_calls": False,
                "prompt_cache_retention": {"type": "ephemeral"},
                "truncation": "disabled",
            },
        )

        self.assertIsNone(error)
        assert payload is not None
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertEqual(payload["prompt_cache_retention"], {"type": "ephemeral"})
        self.assertEqual(payload["truncation"], "disabled")

    def test_websocket_turn_with_tool_output_uses_full_history_new_chain(self) -> None:
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=lambda _message: None,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
                prompt_cache_bucket_id="bucket-123",
            )
        )
        session = ResponsesWebSocketSessionState(
            session_id="session-1",
            explicit_session_key=True,
            last_response_id="resp_1",
            conversation_messages=[
                {"role": "user", "content": "第一轮"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "record_event",
                                "arguments": "{\"value\":\"round1\"}",
                            },
                        }
                    ],
                },
            ],
        )
        current_messages = [
            {"role": "user", "content": "第一轮"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "record_event",
                            "arguments": "{\"value\":\"round1\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "{\"ok\":true,\"value\":\"round1\"}",
            },
            {"role": "user", "content": "继续"},
        ]

        payload, continued, error = adapter._build_openai_responses_websocket_turn_payload(
            route=route,
            request_data={"messages": current_messages},
            session=session,
            current_messages=current_messages,
        )

        self.assertIsNone(error)
        assert payload is not None
        self.assertFalse(continued)
        self.assertNotIn("previous_response_id", payload)
        self.assertEqual(payload["input"][1]["type"], "function_call")
        self.assertEqual(payload["input"][2]["type"], "function_call_output")
        self.assertEqual(
            payload["prompt_cache_key"],
            f"{route.prompt_cache_key}:meta:{session.session_id}",
        )
        self.assertEqual(
            payload["input"][1]["arguments"],
            "{\"value\":\"round1\"}",
        )
        self.assertEqual(
            payload["input"][2]["output"],
            "{\"ok\":true,\"value\":\"round1\"}",
        )

    def test_websocket_turn_reuses_metadata_prompt_cache_scope_across_turns(self) -> None:
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=lambda _message: None,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
                prompt_cache_enabled=True,
                prompt_cache_bucket_id="bucket-123",
            )
        )
        session = ResponsesWebSocketSessionState(
            session_id="session-randomized",
            explicit_session_key=False,
        )
        current_messages = [{"role": "user", "content": "继续"}]

        payload, continued, error = adapter._build_openai_responses_websocket_turn_payload(
            route=route,
            request_data={
                "messages": current_messages,
                "metadata": {"conversation_id": "conversation-123"},
            },
            session=session,
            current_messages=current_messages,
        )

        self.assertIsNone(error)
        assert payload is not None
        self.assertFalse(continued)
        self.assertEqual(session.prompt_cache_scope, "meta:conversation-123")
        self.assertEqual(
            payload["prompt_cache_key"],
            f"{route.prompt_cache_key}:meta:conversation-123",
        )

    def test_websocket_request_canonicalizes_tool_payload_for_cache_stability(self) -> None:
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
                prompt_cache_enabled=True,
                prompt_cache_bucket_id="bucket-123",
            )
        )
        payload, error = LiteLLMUpstreamAdapter._build_openai_responses_websocket_request(
            route=route,
            request_data={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_b",
                                "type": "function",
                                "function": {
                                    "name": "z_tool",
                                    "arguments": '{\n  "b": 2,\n  "a": 1\n}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_b",
                        "content": '{\n  "z": 2,\n  "a": 1\n}',
                    },
                    {"role": "user", "content": "继续"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "z_tool",
                            "strict": True,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "z": {"type": "string"},
                                    "a": {"type": "string"},
                                },
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "a_tool",
                            "strict": True,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "b": {"type": "string"},
                                    "a": {"type": "string"},
                                },
                            },
                        },
                    },
                ],
            },
        )

        self.assertIsNone(error)
        assert payload is not None
        self.assertEqual(
            [tool["name"] for tool in payload["tools"]],
            ["a_tool", "z_tool"],
        )
        self.assertEqual(
            payload["tools"][0]["parameters"]["required"],
            ["a", "b"],
        )
        self.assertEqual(
            payload["input"][0]["arguments"],
            "{\"a\":1,\"b\":2}",
        )
        self.assertEqual(
            payload["input"][1]["output"],
            "{\"a\":1,\"z\":2}",
        )

    def test_websocket_stream_self_heals_unsupported_schema_keyword_before_output(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        created_payloads: list[dict[str, Any]] = []
        connect_calls = 0
        events = [
            {"type": "response.created", "response": {"id": "resp_ok", "created_at": 1}},
            {"type": "response.output_text.delta", "delta": "完成"},
            {"type": "response.completed"},
        ]

        class FakeResponseResource:
            def create(self, **kwargs: Any) -> None:
                created_payloads.append(kwargs)
                query_schema = kwargs["tools"][0]["parameters"]["properties"]["query"]
                if "customKeyword" in query_schema:
                    raise RuntimeError(
                        "Invalid schema for function 'search_logs': "
                        "In context=('properties', 'query'), 'customKeyword' is not permitted."
                    )

        class FakeConnection:
            def __init__(self) -> None:
                self.response = FakeResponseResource()

            def __iter__(self) -> Any:
                return iter(events)

        class FakeConnectionContext:
            def __init__(self) -> None:
                self._connection = FakeConnection()

            def __enter__(self) -> FakeConnection:
                nonlocal connect_calls
                connect_calls += 1
                return self._connection

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

        class FakeResponses:
            def connect(self, **kwargs: Any) -> FakeConnectionContext:
                return FakeConnectionContext()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        with patch("modules.proxy.upstream_adapter.OpenAI", return_value=FakeClient()):
            chunks = list(
                adapter.create_chat_completion(
                    route=route,
                    request_data={
                        "messages": [{"role": "user", "content": "查日志"}],
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "search_logs",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "query": {
                                                "type": "string",
                                                "customKeyword": True,
                                            }
                                        },
                                    },
                                },
                            }
                        ],
                        "stream": True,
                    },
                )
            )

        self.assertEqual(connect_calls, 2)
        self.assertEqual(len(created_payloads), 2)
        self.assertIn(
            "customKeyword",
            created_payloads[0]["tools"][0]["parameters"]["properties"]["query"],
        )
        self.assertNotIn(
            "customKeyword",
            created_payloads[1]["tools"][0]["parameters"]["properties"]["query"],
        )
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "完成")
        self.assertTrue(any("strict schema 自愈" in item for item in logs))

    def test_websocket_stream_self_heals_unsupported_top_level_param_before_output(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=True,
            )
        )
        created_payloads: list[dict[str, Any]] = []
        connect_calls = 0
        batches = [
            [
                {
                    "type": "error",
                    "status": 400,
                    "error": {"detail": "Unsupported parameter: metadata"},
                }
            ],
            [
                {"type": "response.created", "response": {"id": "resp_ok", "created_at": 1}},
                {"type": "response.output_text.delta", "delta": "完成"},
                {"type": "response.completed"},
            ],
        ]
        batch_index = 0

        class FakeResponseResource:
            def create(self, **kwargs: Any) -> None:
                created_payloads.append(kwargs)

        class FakeConnection:
            def __init__(self) -> None:
                self.response = FakeResponseResource()

            def __iter__(self) -> Any:
                nonlocal batch_index
                batch = batches[batch_index]
                batch_index += 1
                return iter(batch)

        class FakeConnectionContext:
            def __init__(self) -> None:
                self._connection = FakeConnection()

            def __enter__(self) -> FakeConnection:
                nonlocal connect_calls
                connect_calls += 1
                return self._connection

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

        class FakeResponses:
            def connect(self, **kwargs: Any) -> FakeConnectionContext:
                return FakeConnectionContext()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        with patch("modules.proxy.upstream_adapter.OpenAI", return_value=FakeClient()):
            chunks = list(
                adapter.create_chat_completion(
                    route=route,
                    request_data={
                        "messages": [{"role": "user", "content": "查日志"}],
                        "stream": True,
                        "metadata": {"foo": "bar"},
                    },
                )
            )

        self.assertEqual(connect_calls, 2)
        self.assertEqual(len(created_payloads), 2)
        self.assertIn("metadata", created_payloads[0])
        self.assertNotIn("metadata", created_payloads[1])
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "完成")
        self.assertTrue(any("参数自愈" in item for item in logs))

    def test_openai_responses_http_stream_self_heals_unsupported_metadata(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://api.openai.com",
                target_model_id="gpt-5.4",
                websocket_mode_enabled=False,
            )
        )
        create_payloads: list[dict[str, Any]] = []

        class FakeRawStream:
            def __iter__(self) -> Any:
                return iter(
                    [
                        {
                            "type": "response.created",
                            "response": {"id": "resp_http_ok", "created_at": 1},
                        },
                        {"type": "response.output_text.delta", "delta": "完成"},
                        {"type": "response.completed"},
                    ]
                )

            def close(self) -> None:
                return None

        class FakeResponses:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **kwargs: Any) -> FakeRawStream:
                self.calls += 1
                create_payloads.append(kwargs)
                if self.calls == 1:
                    raise RuntimeError("Unsupported parameter: metadata")
                return FakeRawStream()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        with patch("modules.proxy.upstream_adapter.OpenAI", return_value=FakeClient()):
            chunks = list(
                adapter._create_openai_responses_http_stream(
                    route=route,
                    request_data={
                        "messages": [{"role": "user", "content": "查日志"}],
                        "stream": True,
                        "metadata": {"foo": "bar"},
                    },
                )
            )

        self.assertEqual(len(create_payloads), 2)
        self.assertIn("metadata", create_payloads[0])
        self.assertNotIn("metadata", create_payloads[1])
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "完成")
        self.assertTrue(any("HTTP 参数自愈" in item for item in logs))


if __name__ == "__main__":
    unittest.main()
