$ErrorActionPreference = 'Stop'
$root = 'c:\Users\ADMIN\Downloads\mtga-tauri'

function Replace-Once {
  param(
    [string]$RelativePath,
    [string]$Old,
    [string]$New
  )
  $path = Join-Path $root $RelativePath
  $text = Get-Content -Raw -Path $path
  if (-not $text.Contains($Old)) {
    throw "Pattern not found in $RelativePath`n---`n$Old"
  }
  $updated = $text.Replace($Old, $New)
  Set-Content -Path $path -Value $updated -Encoding UTF8NoBOM
}

function Insert-Before-Once {
  param(
    [string]$RelativePath,
    [string]$Marker,
    [string]$Block
  )
  $path = Join-Path $root $RelativePath
  $text = Get-Content -Raw -Path $path
  if ($text.Contains($Block)) {
    return
  }
  if (-not $text.Contains($Marker)) {
    throw "Marker not found in $RelativePath`n---`n$Marker"
  }
  $updated = $text.Replace($Marker, $Block + $Marker)
  Set-Content -Path $path -Value $updated -Encoding UTF8NoBOM
}

Replace-Once 'python-src\modules\services\config_service.py' @'
        "model_discovery_strategy",
        "prompt_cache_enabled",
'@ @'
        "model_discovery_strategy",
        "prompt_cache_enabled",
        "request_params_enabled",
'@

Replace-Once 'python-src\modules\services\config_service.py' @'
    prompt_cache_enabled = normalized.get("prompt_cache_enabled")
    if isinstance(prompt_cache_enabled, str):
        normalized["prompt_cache_enabled"] = prompt_cache_enabled.strip().lower() not in {
            "false",
            "0",
            "off",
            "no",
        }
    elif isinstance(prompt_cache_enabled, bool):
        normalized["prompt_cache_enabled"] = prompt_cache_enabled
    else:
        normalized["prompt_cache_enabled"] = False
    return normalized
'@ @'
    prompt_cache_enabled = normalized.get("prompt_cache_enabled")
    if isinstance(prompt_cache_enabled, str):
        normalized["prompt_cache_enabled"] = prompt_cache_enabled.strip().lower() not in {
            "false",
            "0",
            "off",
            "no",
        }
    elif isinstance(prompt_cache_enabled, bool):
        normalized["prompt_cache_enabled"] = prompt_cache_enabled
    else:
        normalized["prompt_cache_enabled"] = False

    request_params_enabled = normalized.get("request_params_enabled")
    if isinstance(request_params_enabled, str):
        normalized["request_params_enabled"] = request_params_enabled.strip().lower() not in {
            "false",
            "0",
            "off",
            "no",
        }
    elif isinstance(request_params_enabled, bool):
        normalized["request_params_enabled"] = request_params_enabled
    else:
        normalized["request_params_enabled"] = True
    return normalized
'@

Replace-Once 'python-src\modules\proxy\proxy_config.py' @'
    model_discovery_strategy: str | None = None
    prompt_cache_bucket_id: str = ""
    prompt_cache_enabled: bool = True
'@ @'
    model_discovery_strategy: str | None = None
    prompt_cache_bucket_id: str = ""
    prompt_cache_enabled: bool = True
    request_params_enabled: bool = True
'@

Replace-Once 'python-src\modules\proxy\proxy_config.py' @'
def normalize_prompt_cache_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "off", "no"}
    return bool(value) if value is not None else False


def provider_supports_model_discovery(value: str | None) -> bool:
'@ @'
def normalize_prompt_cache_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "off", "no"}
    return bool(value) if value is not None else False


def normalize_request_params_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "off", "no"}
    return bool(value) if value is not None else True


def provider_supports_model_discovery(value: str | None) -> bool:
'@

Replace-Once 'python-src\modules\proxy\proxy_config.py' @'
        prompt_cache_enabled=normalize_prompt_cache_enabled(
            raw_config.get("prompt_cache_enabled")
        ),
    )
'@ @'
        prompt_cache_enabled=normalize_prompt_cache_enabled(
            raw_config.get("prompt_cache_enabled")
        ),
        request_params_enabled=normalize_request_params_enabled(
            raw_config.get("request_params_enabled")
        ),
    )
'@

Replace-Once 'python-src\modules\proxy\proxy_config.py' @'
    "normalize_prompt_cache_enabled",
    "normalize_provider",
    "provider_supports_model_discovery",
]
'@ @'
    "normalize_prompt_cache_enabled",
    "normalize_provider",
    "normalize_request_params_enabled",
    "provider_supports_model_discovery",
]
'@

Replace-Once 'python-src\modules\proxy\upstream_adapter.py' @'
    prompt_cache_enabled: bool
    middle_route_applied: bool
    middle_route_ignored: bool
    litellm_base_url: str = ""
'@ @'
    prompt_cache_enabled: bool
    request_params_enabled: bool
    middle_route_applied: bool
    middle_route_ignored: bool
    litellm_base_url: str = ""
'@

Replace-Once 'python-src\modules\proxy\upstream_adapter.py' @'
        call_kwargs: dict[str, Any] = dict(request_data)
        extra_body_obj = call_kwargs.get("extra_body")
'@ @'
        call_kwargs: dict[str, Any] = dict(request_data)
        if not route.request_params_enabled:
            self._strip_optional_request_params(call_kwargs)
            return call_kwargs
        extra_body_obj = call_kwargs.get("extra_body")
'@

Replace-Once 'python-src\modules\proxy\upstream_adapter.py' @'
        call_kwargs = dict(request_data)
        supported_params = self._get_supported_openai_params(route)
'@ @'
        call_kwargs = dict(request_data)
        if not route.request_params_enabled:
            self._strip_optional_request_params(call_kwargs)
            return call_kwargs
        supported_params = self._get_supported_openai_params(route)
'@

Replace-Once 'python-src\modules\proxy\upstream_adapter.py' @'
    def _drop_unsupported_standard_params(
'@ @'
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
'@

Replace-Once 'python-src\modules\proxy\upstream_adapter.py' @'
        prompt_cache_enabled=proxy_config.prompt_cache_enabled,
        middle_route_applied=middle_route_applied,
'@ @'
        prompt_cache_enabled=proxy_config.prompt_cache_enabled,
        request_params_enabled=proxy_config.request_params_enabled,
        middle_route_applied=middle_route_applied,
'@

Replace-Once 'python-src\modules\actions\model_tests.py' @'
        test_data = {
            "model": model_id,
            "messages": [{"role": "user", "content": GENERATION_TEST_PROMPT}],
            "max_tokens": 1,
        }
'@ @'
        test_data = {
            "model": model_id,
            "messages": [{"role": "user", "content": GENERATION_TEST_PROMPT}],
            "max_tokens": 1,
        }
        if bool(config_group.get("request_params_enabled", True)):
            test_data["temperature"] = 0
'@

Replace-Once 'python-src\tests\test_model_tests.py' @'
                        "messages": [{"role": "user", "content": "1"}],
'@ @'
                        "messages": [{"role": "user", "content": "你是谁"}],
'@

Insert-Before-Once 'python-src\tests\test_model_tests.py' @'
    def test_gemini_generation_test_preserves_cached_model_discovery_strategy(self) -> None:
'@ @'
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
            model_tests._run_generation_test_with_litellm(
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

'@

Replace-Once 'python-src\tests\test_config_service.py' @'
                "model_discovery_strategy": "gemini_native_bearer",
                "prompt_cache_enabled": False,
'@ @'
                "model_discovery_strategy": "gemini_native_bearer",
                "prompt_cache_enabled": False,
                "request_params_enabled": False,
'@

Replace-Once 'python-src\tests\test_config_service.py' @'
        self.assertFalse(normalized["prompt_cache_enabled"])
'@ @'
        self.assertFalse(normalized["prompt_cache_enabled"])
        self.assertTrue(normalized["request_params_enabled"])
'@

Replace-Once 'python-src\tests\test_proxy_upstream_adapter.py' @'
    prompt_cache_bucket_id: str = "",
    prompt_cache_enabled: bool = True,
) -> ProxyConfig:
'@ @'
    prompt_cache_bucket_id: str = "",
    prompt_cache_enabled: bool = True,
    request_params_enabled: bool = True,
) -> ProxyConfig:
'@

Replace-Once 'python-src\tests\test_proxy_upstream_adapter.py' @'
        model_discovery_strategy=model_discovery_strategy,
        prompt_cache_bucket_id=prompt_cache_bucket_id,
        prompt_cache_enabled=prompt_cache_enabled,
    )
'@ @'
        model_discovery_strategy=model_discovery_strategy,
        prompt_cache_bucket_id=prompt_cache_bucket_id,
        prompt_cache_enabled=prompt_cache_enabled,
        request_params_enabled=request_params_enabled,
    )
'@

Insert-Before-Once 'python-src\tests\test_proxy_upstream_adapter.py' @'
    def test_anthropic_uses_custom_base_url_without_openai_provider_override(self) -> None:
'@ @'
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

'@
