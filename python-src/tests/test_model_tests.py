from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import cast
from unittest.mock import MagicMock, patch

from modules.actions import model_tests
from modules.proxy.proxy_config import (
    ANTHROPIC_NATIVE_MODEL_DISCOVERY,
    ANTHROPIC_PROVIDER,
    GEMINI_NATIVE_BEARER_MODEL_DISCOVERY,
    GEMINI_NATIVE_X_GOOG_API_KEY_MODEL_DISCOVERY,
    GEMINI_PROVIDER,
)
from modules.proxy.upstream_adapter import CHAT_COMPLETIONS_REQUEST_API, UpstreamRoute

RunGenerationTest = Callable[[dict[str, object], Callable[[str], None]], None]


def _build_config_group(
    *,
    provider: str,
    model_id: str,
    middle_route: str | None = "/v1",
    model_discovery_strategy: str | None = None,
) -> dict[str, object]:
    config_group: dict[str, object] = {
        "provider": provider,
        "api_url": "https://provider.example.com",
        "model_id": model_id,
        "api_key": "test-key",
    }
    if middle_route is not None:
        config_group["middle_route"] = middle_route
    if model_discovery_strategy is not None:
        config_group["model_discovery_strategy"] = model_discovery_strategy
    return config_group


def _run_generation_test(config_group: dict[str, object], log_func: Callable[[str], None]) -> None:
    run_generation_test = cast(
        RunGenerationTest,
        model_tests._run_generation_test_with_litellm,
    )
    run_generation_test(config_group, log_func)


class GenerationTestViaLiteLLMTests(unittest.TestCase):
    def test_non_openai_generation_tests_use_litellm_adapter(self) -> None:
        cases = (
            (ANTHROPIC_PROVIDER, "claude-3-5-haiku-20241022"),
            (GEMINI_PROVIDER, "gemini-2.5-pro"),
        )

        for provider, model_id in cases:
            with self.subTest(provider=provider):
                logs: list[str] = []
                adapter = MagicMock()
                adapter.build_route.return_value = UpstreamRoute(
                    provider=provider,
                    request_api=CHAT_COMPLETIONS_REQUEST_API,
                    litellm_model=f"{provider}/{model_id}",
                    base_url="https://provider.example.com",
                    api_key="test-key",
                    prompt_cache_enabled=True,
                    request_params_enabled=True,
                    middle_route_applied=False,
                    middle_route_ignored=False,
                )
                adapter.create_chat_completion.return_value = {
                    "id": "chatcmpl_123",
                    "object": "chat.completion",
                    "created": 123,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"total_tokens": 3},
                }

                with patch(
                    "modules.actions.model_tests.LiteLLMUpstreamAdapter",
                    return_value=adapter,
                ):
                    _run_generation_test(
                        _build_config_group(provider=provider, model_id=model_id),
                        logs.append,
                    )

                adapter.build_route.assert_called_once()
                adapter.create_chat_completion.assert_called_once()
                call_kwargs = adapter.create_chat_completion.call_args.kwargs
                self.assertEqual(
                    call_kwargs["request_data"],
                    {
                        "model": model_id,
                        "messages": [{"role": "user", "content": "你是谁"}],
                        "max_tokens": 1,
                        "temperature": 0,
                    },
                )
                self.assertTrue(any(f"provider={provider}" in item for item in logs))
                self.assertTrue(any("✅ 模型测活成功" in item for item in logs))
                adapter.close.assert_called_once()

    def test_generation_test_omits_optional_request_params_when_disabled(self) -> None:
        logs: list[str] = []
        adapter = MagicMock()
        adapter.build_route.return_value = UpstreamRoute(
            provider=ANTHROPIC_PROVIDER,
            request_api=CHAT_COMPLETIONS_REQUEST_API,
            litellm_model="anthropic/claude-3-5-haiku-20241022",
            base_url="https://provider.example.com",
            api_key="test-key",
            prompt_cache_enabled=True,
            request_params_enabled=False,
            middle_route_applied=False,
            middle_route_ignored=False,
        )
        adapter.create_chat_completion.return_value = {
            "id": "chatcmpl_123",
            "object": "chat.completion",
            "created": 123,
            "model": "claude-3-5-haiku-20241022",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 3},
        }

        with patch(
            "modules.actions.model_tests.LiteLLMUpstreamAdapter",
            return_value=adapter,
        ):
            _run_generation_test(
                {
                    **_build_config_group(
                        provider=ANTHROPIC_PROVIDER,
                        model_id="claude-3-5-haiku-20241022",
                    ),
                    "request_params_enabled": False,
                },
                logs.append,
            )

        call_kwargs = adapter.create_chat_completion.call_args.kwargs
        self.assertEqual(
            call_kwargs["request_data"],
            {
                "model": "claude-3-5-haiku-20241022",
                "messages": [{"role": "user", "content": "你是谁"}],
                "max_tokens": 1,
            },
        )

    def test_gemini_generation_test_preserves_cached_model_discovery_strategy(self) -> None:
        logs: list[str] = []
        adapter = MagicMock()
        adapter.build_route.return_value = UpstreamRoute(
            provider=GEMINI_PROVIDER,
            request_api=CHAT_COMPLETIONS_REQUEST_API,
            litellm_model="gemini/gemini-2.5-pro",
            base_url="https://provider.example.com",
            api_key="test-key",
            prompt_cache_enabled=True,
            request_params_enabled=True,
            middle_route_applied=False,
            middle_route_ignored=False,
            model_discovery_strategy=GEMINI_NATIVE_X_GOOG_API_KEY_MODEL_DISCOVERY,
        )
        adapter.create_chat_completion.return_value = {
            "id": "chatcmpl_123",
            "object": "chat.completion",
            "created": 123,
            "model": "gemini-2.5-pro",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 3},
        }

        with patch(
            "modules.actions.model_tests.LiteLLMUpstreamAdapter",
            return_value=adapter,
        ):
            _run_generation_test(
                _build_config_group(
                    provider=GEMINI_PROVIDER,
                    model_id="gemini-2.5-pro",
                    middle_route=None,
                    model_discovery_strategy=GEMINI_NATIVE_X_GOOG_API_KEY_MODEL_DISCOVERY,
                ),
                logs.append,
            )

        adapter.build_route.assert_called_once()
        proxy_config = adapter.build_route.call_args.args[0]
        self.assertEqual(
            proxy_config.model_discovery_strategy,
            GEMINI_NATIVE_X_GOOG_API_KEY_MODEL_DISCOVERY,
        )
        adapter.close.assert_called_once()


class ModelDiscoveryTests(unittest.TestCase):
    def test_fetch_model_list_uses_anthropic_native_models_endpoint(self) -> None:
        logs: list[str] = []
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "object": "list",
            "data": [
                {"id": "glm-4.7-flash"},
                {"id": "glm-4.7-air"},
            ],
        }

        with patch("modules.actions.model_tests.requests.get", return_value=response) as get_mock:
            result = model_tests.fetch_model_list_result(
                _build_config_group(
                    provider=ANTHROPIC_PROVIDER,
                    model_id="glm-4.7-flash",
                ),
                log_func=logs.append,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.model_ids, ["glm-4.7-air", "glm-4.7-flash"])
        self.assertEqual(result.strategy_id, ANTHROPIC_NATIVE_MODEL_DISCOVERY)
        get_mock.assert_called_once_with(
            "https://provider.example.com/v1/models",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
            },
            timeout=10,
        )
        self.assertTrue(any("✅ 模型列表获取成功" in item for item in logs))

    def test_fetch_model_list_falls_back_to_gemini_native_bearer(self) -> None:
        logs: list[str] = []
        upstream_401 = MagicMock()
        upstream_401.status_code = 401
        upstream_401.text = '{"error":"invalid api key"}'
        success = MagicMock()
        success.status_code = 200
        success.json.return_value = {
            "data": [
                {"id": "gemini-2.5-pro"},
                {"id": "gemini-2.5-flash"},
            ]
        }

        with patch(
            "modules.actions.model_tests.requests.get",
            side_effect=[upstream_401, success],
        ) as get_mock:
            result = model_tests.fetch_model_list_result(
                _build_config_group(
                    provider=GEMINI_PROVIDER,
                    model_id="gemini-2.5-pro",
                    middle_route=None,
                ),
                log_func=logs.append,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.model_ids, ["gemini-2.5-flash", "gemini-2.5-pro"])
        self.assertEqual(result.strategy_id, GEMINI_NATIVE_BEARER_MODEL_DISCOVERY)
        self.assertEqual(get_mock.call_count, 2)
        first_call = get_mock.call_args_list[0]
        second_call = get_mock.call_args_list[1]
        self.assertEqual(
            first_call.args[0],
            "https://provider.example.com/v1beta/models?key=test-key",
        )
        self.assertEqual(first_call.kwargs["headers"], {})
        self.assertEqual(
            second_call.args[0],
            "https://provider.example.com/v1beta/models",
        )
        self.assertEqual(
            second_call.kwargs["headers"],
            {"Authorization": "Bearer test-key"},
        )
        self.assertTrue(any("尝试降级到下一种模型发现策略" in item for item in logs))


if __name__ == "__main__":
    unittest.main()
