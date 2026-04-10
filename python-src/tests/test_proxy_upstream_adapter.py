from __future__ import annotations

import tempfile
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

        with patch(
            "modules.proxy.upstream_adapter.litellm.get_supported_openai_params",
            return_value=["reasoning_effort", "stream"],
        ), patch(
            "modules.proxy.upstream_adapter.litellm.completion",
            return_value={"id": "chatcmpl_123", "choices": []},
        ) as completion_mock:
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
            "modules.proxy.upstream_adapter.litellm.completion",
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
            "modules.proxy.upstream_adapter.litellm.completion",
            return_value={"id": "chatcmpl_123", "choices": []},
        ) as completion_mock:
            adapter.create_chat_completion(
                route=route,
                request_data={"messages": [{"role": "user", "content": "你好"}]},
            )

        call_kwargs = completion_mock.call_args.kwargs
        self.assertEqual(call_kwargs["base_url"], "https://anthropic-proxy.example.com")
        self.assertNotIn("api_base", call_kwargs)
        self.assertNotIn("custom_llm_provider", call_kwargs)


if __name__ == "__main__":
    unittest.main()
