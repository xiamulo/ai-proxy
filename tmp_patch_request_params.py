from pathlib import Path

root = Path(r"c:\Users\ADMIN\Downloads\mtga-tauri")


def replace_once(path_str: str, old: str, new: str) -> None:
    path = root / path_str
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"pattern not found: {path_str}\n---\n{old}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def insert_before_once(path_str: str, marker: str, block: str) -> None:
    path = root / path_str
    text = path.read_text(encoding="utf-8")
    if block in text:
        return
    if marker not in text:
        raise SystemExit(f"marker not found: {path_str}\n---\n{marker}")
    path.write_text(text.replace(marker, block + marker, 1), encoding="utf-8")


replace_once(
    "python-src/modules/services/config_service.py",
    '        "model_discovery_strategy",\n        "prompt_cache_enabled",\n',
    '        "model_discovery_strategy",\n        "prompt_cache_enabled",\n        "request_params_enabled",\n',
)
replace_once(
    "python-src/modules/services/config_service.py",
    '    prompt_cache_enabled = normalized.get("prompt_cache_enabled")\n    if isinstance(prompt_cache_enabled, str):\n        normalized["prompt_cache_enabled"] = prompt_cache_enabled.strip().lower() not in {\n            "false",\n            "0",\n            "off",\n            "no",\n        }\n    elif isinstance(prompt_cache_enabled, bool):\n        normalized["prompt_cache_enabled"] = prompt_cache_enabled\n    else:\n        normalized["prompt_cache_enabled"] = False\n    return normalized\n',
    '    prompt_cache_enabled = normalized.get("prompt_cache_enabled")\n    if isinstance(prompt_cache_enabled, str):\n        normalized["prompt_cache_enabled"] = prompt_cache_enabled.strip().lower() not in {\n            "false",\n            "0",\n            "off",\n            "no",\n        }\n    elif isinstance(prompt_cache_enabled, bool):\n        normalized["prompt_cache_enabled"] = prompt_cache_enabled\n    else:\n        normalized["prompt_cache_enabled"] = False\n\n    request_params_enabled = normalized.get("request_params_enabled")\n    if isinstance(request_params_enabled, str):\n        normalized["request_params_enabled"] = request_params_enabled.strip().lower() not in {\n            "false",\n            "0",\n            "off",\n            "no",\n        }\n    elif isinstance(request_params_enabled, bool):\n        normalized["request_params_enabled"] = request_params_enabled\n    else:\n        normalized["request_params_enabled"] = True\n    return normalized\n',
)

replace_once(
    "python-src/modules/proxy/proxy_config.py",
    '    model_discovery_strategy: str | None = None\n    prompt_cache_bucket_id: str = ""\n    prompt_cache_enabled: bool = True\n',
    '    model_discovery_strategy: str | None = None\n    prompt_cache_bucket_id: str = ""\n    prompt_cache_enabled: bool = True\n    request_params_enabled: bool = True\n',
)
replace_once(
    "python-src/modules/proxy/proxy_config.py",
    'def normalize_prompt_cache_enabled(value: Any) -> bool:\n    if isinstance(value, bool):\n        return value\n    if isinstance(value, str):\n        return value.strip().lower() not in {"false", "0", "off", "no"}\n    return bool(value) if value is not None else False\n\n\ndef provider_supports_model_discovery(value: str | None) -> bool:\n',
    'def normalize_prompt_cache_enabled(value: Any) -> bool:\n    if isinstance(value, bool):\n        return value\n    if isinstance(value, str):\n        return value.strip().lower() not in {"false", "0", "off", "no"}\n    return bool(value) if value is not None else False\n\n\ndef normalize_request_params_enabled(value: Any) -> bool:\n    if isinstance(value, bool):\n        return value\n    if isinstance(value, str):\n        return value.strip().lower() not in {"false", "0", "off", "no"}\n    return bool(value) if value is not None else True\n\n\ndef provider_supports_model_discovery(value: str | None) -> bool:\n',
)
replace_once(
    "python-src/modules/proxy/proxy_config.py",
    '        prompt_cache_enabled=normalize_prompt_cache_enabled(\n            raw_config.get("prompt_cache_enabled")\n        ),\n    )\n',
    '        prompt_cache_enabled=normalize_prompt_cache_enabled(\n            raw_config.get("prompt_cache_enabled")\n        ),\n        request_params_enabled=normalize_request_params_enabled(\n            raw_config.get("request_params_enabled")\n        ),\n    )\n',
)
replace_once(
    "python-src/modules/proxy/proxy_config.py",
    '    "normalize_prompt_cache_enabled",\n    "normalize_provider",\n    "provider_supports_model_discovery",\n]',
    '    "normalize_prompt_cache_enabled",\n    "normalize_provider",\n    "normalize_request_params_enabled",\n    "provider_supports_model_discovery",\n]',
)

replace_once(
    "python-src/modules/proxy/upstream_adapter.py",
    '    prompt_cache_enabled: bool\n    middle_route_applied: bool\n    middle_route_ignored: bool\n    litellm_base_url: str = ""\n',
    '    prompt_cache_enabled: bool\n    request_params_enabled: bool\n    middle_route_applied: bool\n    middle_route_ignored: bool\n    litellm_base_url: str = ""\n',
)
replace_once(
    "python-src/modules/proxy/upstream_adapter.py",
    '        call_kwargs: dict[str, Any] = dict(request_data)\n        extra_body_obj = call_kwargs.get("extra_body")\n',
    '        call_kwargs: dict[str, Any] = dict(request_data)\n        if not route.request_params_enabled:\n            self._strip_optional_request_params(call_kwargs)\n            return call_kwargs\n        extra_body_obj = call_kwargs.get("extra_body")\n',
)
replace_once(
    "python-src/modules/proxy/upstream_adapter.py",
    '        call_kwargs = dict(request_data)\n        supported_params = self._get_supported_openai_params(route)\n',
    '        call_kwargs = dict(request_data)\n        if not route.request_params_enabled:\n            self._strip_optional_request_params(call_kwargs)\n            return call_kwargs\n        supported_params = self._get_supported_openai_params(route)\n',
)
replace_once(
    "python-src/modules/proxy/upstream_adapter.py",
    '    def _drop_unsupported_standard_params(\n',
    '    @staticmethod\n    def _strip_optional_request_params(call_kwargs: dict[str, Any]) -> None:\n        for key in (\n            "temperature",\n            "top_p",\n            "presence_penalty",\n            "frequency_penalty",\n            "logprobs",\n            "top_logprobs",\n            "seed",\n            "service_tier",\n            "reasoning_effort",\n            "prediction",\n            "modalities",\n            "audio",\n            "metadata",\n            "store",\n            "extra_body",\n            "allowed_openai_params",\n        ):\n            call_kwargs.pop(key, None)\n\n    def _drop_unsupported_standard_params(\n',
)
replace_once(
    "python-src/modules/proxy/upstream_adapter.py",
    '        prompt_cache_enabled=proxy_config.prompt_cache_enabled,\n        middle_route_applied=middle_route_applied,\n',
    '        prompt_cache_enabled=proxy_config.prompt_cache_enabled,\n        request_params_enabled=proxy_config.request_params_enabled,\n        middle_route_applied=middle_route_applied,\n',
)

replace_once(
    "python-src/modules/actions/model_tests.py",
    '        test_data = {\n            "model": model_id,\n            "messages": [{"role": "user", "content": GENERATION_TEST_PROMPT}],\n            "max_tokens": 1,\n        }\n',
    '        test_data = {\n            "model": model_id,\n            "messages": [{"role": "user", "content": GENERATION_TEST_PROMPT}],\n            "max_tokens": 1,\n        }\n        if bool(config_group.get("request_params_enabled", True)):\n            test_data["temperature"] = 0\n',
)

replace_once(
    "python-src/tests/test_model_tests.py",
    '                        "messages": [{"role": "user", "content": "1"}],\n',
    '                        "messages": [{"role": "user", "content": "你是谁"}],\n',
)
insert_before_once(
    "python-src/tests/test_model_tests.py",
    '    def test_gemini_generation_test_preserves_cached_model_discovery_strategy(self) -> None:\n',
    '    def test_generation_test_omits_optional_request_params_when_disabled(self) -> None:\n        logs: list[str] = []\n        adapter = MagicMock()\n        adapter.build_route.return_value = UpstreamRoute(\n            provider=ANTHROPIC_PROVIDER,\n            request_api=CHAT_COMPLETIONS_REQUEST_API,\n            litellm_model="anthropic/claude-3-5-haiku-20241022",\n            base_url="https://provider.example.com",\n            api_key="test-key",\n            prompt_cache_enabled=True,\n            request_params_enabled=False,\n            middle_route_applied=False,\n            middle_route_ignored=False,\n        )\n        adapter.create_chat_completion.return_value = {\n            "id": "chatcmpl_123",\n            "object": "chat.completion",\n            "created": 123,\n            "model": "claude-3-5-haiku-20241022",\n            "choices": [\n                {\n                    "index": 0,\n                    "message": {"role": "assistant", "content": "ok"},\n                    "finish_reason": "stop",\n                }\n            ],\n            "usage": {"total_tokens": 3},\n        }\n\n        with patch(\n            "modules.actions.model_tests.LiteLLMUpstreamAdapter",\n            return_value=adapter,\n        ):\n            model_tests._run_generation_test_with_litellm(\n                {\n                    **_build_config_group(\n                        provider=ANTHROPIC_PROVIDER,\n                        model_id="claude-3-5-haiku-20241022",\n                    ),\n                    "request_params_enabled": False,\n                },\n                logs.append,\n            )\n\n        call_kwargs = adapter.create_chat_completion.call_args.kwargs\n        self.assertEqual(\n            call_kwargs["request_data"],\n            {\n                "model": "claude-3-5-haiku-20241022",\n                "messages": [{"role": "user", "content": "你是谁"}],\n                "max_tokens": 1,\n            },\n        )\n\n',
)

replace_once(
    "python-src/tests/test_config_service.py",
    '                "model_discovery_strategy": "gemini_native_bearer",\n                "prompt_cache_enabled": False,\n',
    '                "model_discovery_strategy": "gemini_native_bearer",\n                "prompt_cache_enabled": False,\n                "request_params_enabled": False,\n',
)
replace_once(
    "python-src/tests/test_config_service.py",
    '        self.assertFalse(normalized["prompt_cache_enabled"])\n',
    '        self.assertFalse(normalized["prompt_cache_enabled"])\n        self.assertTrue(normalized["request_params_enabled"])\n',
)

replace_once(
    "python-src/tests/test_proxy_upstream_adapter.py",
    '    prompt_cache_bucket_id: str = "",\n    prompt_cache_enabled: bool = True,\n) -> ProxyConfig:\n',
    '    prompt_cache_bucket_id: str = "",\n    prompt_cache_enabled: bool = True,\n    request_params_enabled: bool = True,\n) -> ProxyConfig:\n',
)
replace_once(
    "python-src/tests/test_proxy_upstream_adapter.py",
    '        model_discovery_strategy=model_discovery_strategy,\n        prompt_cache_bucket_id=prompt_cache_bucket_id,\n        prompt_cache_enabled=prompt_cache_enabled,\n    )\n',
    '        model_discovery_strategy=model_discovery_strategy,\n        prompt_cache_bucket_id=prompt_cache_bucket_id,\n        prompt_cache_enabled=prompt_cache_enabled,\n        request_params_enabled=request_params_enabled,\n    )\n',
)
insert_before_once(
    "python-src/tests/test_proxy_upstream_adapter.py",
    '    def test_anthropic_uses_custom_base_url_without_openai_provider_override(self) -> None:\n',
    '    def test_openai_compatible_request_drops_optional_params_when_disabled(self) -> None:\n        adapter = LiteLLMUpstreamAdapter(\n            disable_ssl_strict_mode=False,\n            log_func=lambda _message: None,\n        )\n        route = build_upstream_route(\n            _build_proxy_config(\n                provider=OPENAI_CHAT_COMPLETION_PROVIDER,\n                target_api_base_url="https://example.com",\n                target_model_id="gpt-5.3-codex",\n                request_params_enabled=False,\n            )\n        )\n\n        with patch(\n            "modules.proxy.upstream_adapter.litellm.completion",\n            return_value={"id": "chatcmpl_123", "choices": []},\n        ) as completion_mock:\n            adapter.create_chat_completion(\n                route=route,\n                request_data={\n                    "messages": [{"role": "user", "content": "你好"}],\n                    "temperature": 0.1,\n                    "top_p": 0.5,\n                    "presence_penalty": 1,\n                    "frequency_penalty": 1,\n                    "reasoning_effort": "medium",\n                    "metadata": {"source": "test"},\n                },\n            )\n\n        call_kwargs = completion_mock.call_args.kwargs\n        self.assertEqual(call_kwargs["model"], "gpt-5.3-codex")\n        self.assertNotIn("temperature", call_kwargs)\n        self.assertNotIn("top_p", call_kwargs)\n        self.assertNotIn("presence_penalty", call_kwargs)\n        self.assertNotIn("frequency_penalty", call_kwargs)\n        self.assertNotIn("reasoning_effort", call_kwargs)\n        self.assertNotIn("metadata", call_kwargs)\n\n',
)
