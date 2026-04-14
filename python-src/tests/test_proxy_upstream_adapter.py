from __future__ import annotations

import tempfile
import time
import unittest
from typing import Any
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
    def test_openai_response_drops_unsupported_standard_params_before_litellm(self) -> None:
        logs: list[str] = []
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=logs.append,
        )
        route = build_upstream_route(
            _build_proxy_config(
                provider=OPENAI_RESPONSE_PROVIDER,
                target_api_base_url="https://example.com",
                target_model_id="gpt-5",
            )
        )

        with (
            patch(
                "modules.proxy.upstream_adapter._get_supported_openai_params",
                return_value=["reasoning_effort", "stream"],
            ),
            patch(
                "modules.proxy.upstream_adapter._create_litellm_completion",
                return_value={"id": "chatcmpl_123", "choices": []},
            ) as completion_mock,
        ):
            adapter.create_chat_completion(
                route=route,
                request_data={
                    "messages": [{"role": "user", "content": "你好"}],
                    "temperature": 0,
                    "service_tier": "priority",
                    "store": True,
                    "reasoning_effort": "medium",
                },
            )

        call_kwargs = completion_mock.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "responses/gpt-5")
        self.assertEqual(call_kwargs["reasoning_effort"], "medium")
        self.assertNotIn("temperature", call_kwargs)
        self.assertNotIn("service_tier", call_kwargs)
        self.assertNotIn("store", call_kwargs)

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
            patch(
                "modules.proxy.upstream_adapter._create_litellm_completion",
                return_value=iter([{"id": "chatcmpl_123", "choices": []}]),
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

        self.assertEqual(len(chunks), 1)
        completion_mock.assert_called_once()
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
            def connect(self) -> FakeConnection:
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
            def connect(self) -> FakeConnectionContext:
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
                {
                    "type": "error",
                    "status": 400,
                    "error": {
                        "type": "invalid_request_error",
                        "code": "websocket_connection_limit_reached",
                        "message": "limit reached",
                    },
                }
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
            def connect(self) -> FakeConnectionContext:
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

        self.assertEqual(connect_calls, 3)
        self.assertEqual(created_payloads[1]["previous_response_id"], "resp_1")
        self.assertNotIn("previous_response_id", created_payloads[2])
        self.assertTrue(any("code=websocket_connection_limit_reached" in item for item in logs))

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
            def connect(self) -> FakeConnectionContext:
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
            def connect(self) -> FakeConnectionContext:
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
        adapter = LiteLLMUpstreamAdapter(
            disable_ssl_strict_mode=False,
            log_func=lambda _message: None,
        )
        shared_messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        session_a = ResponsesWebSocketSessionState(
            session_id="session-a",
            explicit_session_key=False,
            conversation_messages=shared_messages,
        )
        session_b = ResponsesWebSocketSessionState(
            session_id="session-b",
            explicit_session_key=False,
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
        self.assertEqual(parameters["required"], ["title", "meta"])
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


if __name__ == "__main__":
    unittest.main()
