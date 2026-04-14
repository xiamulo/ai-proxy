from __future__ import annotations

import tempfile
import unittest
from typing import Any

from modules.proxy.proxy_transport import ProxyTransport


class DummyResourceManager:
    def __init__(self, *, user_data_dir: str, program_resource_dir: str) -> None:
        self.user_data_dir = user_data_dir
        self.program_resource_dir = program_resource_dir


class ProxyTransportTests(unittest.TestCase):
    def test_iter_coalesced_openai_text_chunks_merges_small_content_deltas(self) -> None:
        temp_dir = tempfile.mkdtemp(prefix="mtga-proxy-transport-")
        transport = ProxyTransport(
            resource_manager=DummyResourceManager(
                user_data_dir=temp_dir,
                program_resource_dir=temp_dir,
            ),  # type: ignore[arg-type]
            disable_ssl_strict_mode=False,
            log_func=lambda _message: None,
        )

        chunks = [
            {
                "id": "chatcmpl_123",
                "created": 123,
                "model": "gpt-5.4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "你"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_123",
                "created": 123,
                "model": "gpt-5.4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "好"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_123",
                "created": 123,
                "model": "gpt-5.4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "啊"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_123",
                "created": 123,
                "model": "gpt-5.4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "！"},
                        "finish_reason": "stop",
                    }
                ],
            },
        ]

        merged = list(transport.iter_coalesced_openai_text_chunks(chunks, target_chars=8))

        self.assertEqual(len(merged), 2)
        first_delta = merged[0]["choices"][0]["delta"]
        self.assertEqual(first_delta["role"], "assistant")
        self.assertEqual(first_delta["content"], "你好啊")
        self.assertEqual(merged[1]["choices"][0]["delta"]["content"], "！")
        self.assertEqual(merged[1]["choices"][0]["finish_reason"], "stop")

    def test_iter_coalesced_openai_text_chunks_does_not_merge_tool_call_chunks(self) -> None:
        temp_dir = tempfile.mkdtemp(prefix="mtga-proxy-transport-")
        transport = ProxyTransport(
            resource_manager=DummyResourceManager(
                user_data_dir=temp_dir,
                program_resource_dir=temp_dir,
            ),  # type: ignore[arg-type]
            disable_ssl_strict_mode=False,
            log_func=lambda _message: None,
        )

        chunks: list[dict[str, Any]] = [
            {
                "id": "chatcmpl_123",
                "created": 123,
                "model": "gpt-5.4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "你"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_123",
                "created": 123,
                "model": "gpt-5.4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup"},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
        ]

        merged = list(transport.iter_coalesced_openai_text_chunks(chunks, target_chars=8))

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["choices"][0]["delta"]["content"], "你")
        self.assertIn("tool_calls", merged[1]["choices"][0]["delta"])


if __name__ == "__main__":
    unittest.main()
