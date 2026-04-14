from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from websockets.sync.client import connect


@dataclass(frozen=True)
class ProbeSummary:
    started_at: float
    first_event_at: float | None
    response_completed_at: float | None
    response_usage: dict[str, Any] | None
    response_text: str
    event_count: int


def _read_required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"缺少 {name}")
    return value


def _build_ws_url(base_url: str) -> str:
    if base_url.startswith("ws://") or base_url.startswith("wss://"):
        return base_url
    parsed = urlparse(base_url.rstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/")
    ws_path = path if path.endswith("/responses") else f"{path}/responses" if path else "/responses"
    return f"{scheme}://{parsed.netloc}{ws_path}"


def _build_probe_payload(*, model: str, target_words: str) -> dict[str, Any]:
    prompt = (
        f"请用中文写一篇大约{target_words}字的科幻小说。"
        "要求：有完整开端、冲突、反转和结尾；不要分点；直接输出正文。"
    )
    return {
        "type": "response.create",
        "model": model,
        "store": False,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
    }


def _extract_response(event: dict[str, Any]) -> dict[str, Any] | None:
    response_obj = event.get("response")
    if isinstance(response_obj, dict):
        return response_obj
    return None


def _iter_output_text(response: dict[str, Any]) -> Iterable[str]:
    output_obj = response.get("output")
    if not isinstance(output_obj, list):
        return []

    texts: list[str] = []
    for item_obj in output_obj:
        if not isinstance(item_obj, dict):
            continue
        if item_obj.get("type") != "message":
            continue
        content_obj = item_obj.get("content")
        if not isinstance(content_obj, list):
            continue
        for part_obj in content_obj:
            if not isinstance(part_obj, dict):
                continue
            if part_obj.get("type") != "output_text":
                continue
            text_obj = part_obj.get("text")
            if isinstance(text_obj, str):
                texts.append(text_obj)
    return texts


def _parse_usage_output_tokens(response_usage: dict[str, Any] | None) -> int:
    if response_usage is None:
        return 0
    output_tokens_obj = response_usage.get("output_tokens")
    if isinstance(output_tokens_obj, int):
        return output_tokens_obj
    return 0


def _consume_probe_stream(
    *,
    ws_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    started_at: float,
) -> ProbeSummary:
    first_event_at: float | None = None
    response_completed_at: float | None = None
    response_usage: dict[str, Any] | None = None
    response_text = ""
    event_count = 0

    with connect(ws_url, additional_headers=headers, open_timeout=30, close_timeout=10) as ws:
        connected_at = time.perf_counter()
        print(f"Handshake seconds: {connected_at - started_at:.3f}")

        ws.send(json.dumps(payload, ensure_ascii=False))
        sent_at = time.perf_counter()
        print(f"Sent request after seconds: {sent_at - started_at:.3f}")

        while True:
            raw = ws.recv()
            now = time.perf_counter()
            event_count += 1
            if first_event_at is None:
                first_event_at = now
                print(f"First event seconds: {first_event_at - started_at:.3f}")

            event = json.loads(raw)
            event_type = event.get("type")

            if event_type == "response.output_text.delta":
                delta_obj = event.get("delta")
                if isinstance(delta_obj, str):
                    response_text += delta_obj
                continue

            if event_type == "response.completed":
                response_completed_at = now
                response = _extract_response(event)
                if response is not None:
                    usage_obj = response.get("usage")
                    if isinstance(usage_obj, dict):
                        response_usage = usage_obj
                    text_chunks = list(_iter_output_text(response))
                    if text_chunks:
                        response_text = "".join(text_chunks)
                break

            if event_type in {"response.failed", "error"}:
                print(json.dumps(event, ensure_ascii=False, indent=2))
                raise RuntimeError(f"收到失败事件: {event_type}")

    return ProbeSummary(
        started_at=started_at,
        first_event_at=first_event_at,
        response_completed_at=response_completed_at,
        response_usage=response_usage,
        response_text=response_text,
        event_count=event_count,
    )


def _print_probe_summary(summary: ProbeSummary) -> None:
    finished_at = summary.response_completed_at or time.perf_counter()
    elapsed = finished_at - summary.started_at
    generation_elapsed = finished_at - (summary.first_event_at or summary.started_at)
    output_tokens = _parse_usage_output_tokens(summary.response_usage)
    tokens_per_second = (output_tokens / generation_elapsed) if generation_elapsed > 0 else 0.0

    print(f"Event count: {summary.event_count}")
    print(f"Total seconds: {elapsed:.3f}")
    print(f"Generation seconds: {generation_elapsed:.3f}")
    print(f"Output tokens: {output_tokens}")
    print(f"Tokens/sec: {tokens_per_second:.3f}")
    print(f"Output chars: {len(summary.response_text)}")
    print("Preview:")
    print(summary.response_text[:500])


def main() -> int:
    api_key = _read_required_env("OPENAI_API_KEY")
    base_url = _read_required_env("OPENAI_BASE_URL")
    model = os.environ.get("OPENAI_MODEL", "gpt-5.4").strip() or "gpt-5.4"
    target_words = os.environ.get("OPENAI_TARGET_WORDS", "3000").strip() or "3000"

    ws_url = _build_ws_url(base_url)
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = _build_probe_payload(model=model, target_words=target_words)

    started_at = time.perf_counter()

    print(f"WS URL: {ws_url}")
    print(f"Model: {model}")

    summary = _consume_probe_stream(
        ws_url=ws_url,
        headers=headers,
        payload=payload,
        started_at=started_at,
    )
    _print_probe_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
