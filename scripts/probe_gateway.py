"""Probe the LLM gateway: try several path/verb combos and log timing."""

import json
import time

import httpx

from insightflow.core.config import get_settings


def main() -> None:
    s = get_settings()
    if not s.openai_api_key:
        raise SystemExit("INSIGHTFLOW_OPENAI_API_KEY not set")
    base = s.openai_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {s.openai_api_key}", "Content-Type": "application/json"}

    body = json.dumps(
        {"model": s.default_model, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 64}
    )

    def try_request(client: httpx.Client, method: str, path: str, **kwargs) -> None:
        t0 = time.time()
        try:
            r = client.request(method, base + path, headers=headers, timeout=240, **kwargs)
            dt = time.time() - t0
            text = r.text[:300].replace("\n", " ")
            print(f"{method} {path} -> {r.status_code} in {dt:.1f}s | {text}")
        except Exception as e:  # noqa: BLE001
            dt = time.time() - t0
            print(f"{method} {path} -> FAIL after {dt:.1f}s | {type(e).__name__}: {e}")

    with httpx.Client() as c:
        try_request(c, "POST", "/v1/chat/completions", content=body)
        try_request(c, "POST", "/chat/completions", content=body)
        try_request(c, "GET", "/v1/models")


if __name__ == "__main__":
    main()
