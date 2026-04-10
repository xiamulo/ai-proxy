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

Replace-Once -RelativePath 'python-src\modules\services\config_service.py' -Old @'
        "prompt_cache_enabled",
'@ -New @'
        "prompt_cache_enabled",
        "request_params_enabled",
'@

Replace-Once -RelativePath 'python-src\modules\services\config_service.py' -Old @'
    else:
        normalized["prompt_cache_enabled"] = False
    return normalized
'@ -New @'
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

Replace-Once -RelativePath 'python-src\modules\actions\model_tests.py' -Old @'
        prompt_cache_bucket_id="",
        api_key=(config_group.get("api_key") or ""),
'@ -New @'
        prompt_cache_bucket_id="",
        request_params_enabled=bool(config_group.get("request_params_enabled", True)),
        api_key=(config_group.get("api_key") or ""),
'@

Replace-Once -RelativePath 'python-src\modules\actions\model_tests.py' -Old @'
        test_data = {
            "model": model_id,
            "messages": [{"role": "user", "content": GENERATION_TEST_PROMPT}],
            "max_tokens": 1,
        }
'@ -New @'
        test_data = {
            "model": model_id,
            "messages": [{"role": "user", "content": GENERATION_TEST_PROMPT}],
            "max_tokens": 1,
        }
        if bool(config_group.get("request_params_enabled", True)):
            test_data["temperature"] = 0
'@

Replace-Once -RelativePath 'python-src\tests\test_model_tests.py' -Old @'
            prompt_cache_enabled=True,
            middle_route_applied=False,
'@ -New @'
            prompt_cache_enabled=True,
            request_params_enabled=True,
            middle_route_applied=False,
'@

Replace-Once -RelativePath 'python-src\tests\test_model_tests.py' -Old @'
            prompt_cache_enabled=True,
            middle_route_applied=False,
'@ -New @'
            prompt_cache_enabled=True,
            request_params_enabled=False,
            middle_route_applied=False,
'@

Replace-Once -RelativePath 'python-src\tests\test_model_tests.py' -Old @'
                        "max_tokens": 1,
                    },
'@ -New @'
                        "max_tokens": 1,
                        "temperature": 0,
                    },
'@

Replace-Once -RelativePath 'python-src\tests\test_proxy_upstream_adapter.py' -Old @'
    prompt_cache_enabled: bool = True,
) -> ProxyConfig:
'@ -New @'
    prompt_cache_enabled: bool = True,
    request_params_enabled: bool = True,
) -> ProxyConfig:
'@

Replace-Once -RelativePath 'python-src\tests\test_proxy_upstream_adapter.py' -Old @'
        prompt_cache_enabled=prompt_cache_enabled,
    )
'@ -New @'
        prompt_cache_enabled=prompt_cache_enabled,
        request_params_enabled=request_params_enabled,
    )
'@

Replace-Once -RelativePath 'python-src\tests\test_config_service.py' -Old @'
                "prompt_cache_enabled": False,
            },
'@ -New @'
                "prompt_cache_enabled": False,
                "request_params_enabled": False,
            },
'@

Replace-Once -RelativePath 'python-src\tests\test_config_service.py' -Old @'
        self.assertFalse(normalized["prompt_cache_enabled"])
'@ -New @'
        self.assertFalse(normalized["prompt_cache_enabled"])
        self.assertTrue(normalized["request_params_enabled"])
'@
